#!/usr/bin/env python3
"""Pick backfill rationale candidates to replace the reviewed-out clips, filtering gore.

Selects the next-best strong candidates from the merged pool (excluding the ones already in the
gold set and the ones deleted in review), balanced by domain, then runs a Gemini gore/disturbing
pass and drops anything a viewer would find distressing (gory game clips were removed in review
for annotator comfort). Writes backfill_set.csv in rationale_set.csv's schema; the cultural / draft /
scene / load steps then run on it and append to the Stage C tasks for review.

  export TINKER/GOOGLE creds; python data_pipeline/06_annotate/backfill_select.py --need 44 --buffer 20
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "common"))
import rvb_config as cfg  # noqa: E402  (loads .env, puts every stage dir on sys.path)

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import select_stage_c as sel  # noqa: E402  (combined_key, balanced_pick, probe_dur)
from build_pilot_frame import default_data_root  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = cfg.work("stimulus_ranking")
    p.add_argument("--pool_csv", type=Path, default=d / "pool_merged.csv")
    p.add_argument("--gemini_jsonl", type=Path, default=d / "gemini_merged.jsonl")
    p.add_argument("--deleted_csv", type=Path, default=d / "rationale_deleted.csv")
    p.add_argument("--out", type=Path, default=d / "backfill_set.csv")
    p.add_argument("--data_root", type=Path, default=default_data_root())
    p.add_argument("--recut_dir", type=Path, default=cfg.RECUT_ROOT / "stimuli")
    p.add_argument("--need", type=int, default=44)
    p.add_argument("--buffer", type=int, default=24, help="extra candidates so deletes during re-review still reach 200")
    p.add_argument("--gore_max", type=int, default=1, help="drop candidates with gore rating above this")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--model", default="gemini-3.6-flash")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


GORE_PROMPT = ("Rate how graphically violent, gory, or disturbing this short stimulus clip is, on a "
               "scale a general viewer would feel: 0 = none, 1 = mild (cartoon/comedic/implied), "
               "2 = strong (real injury, blood, jump-scare horror), 3 = extreme (graphic gore/"
               "torture). Give only the score and a 3-word reason.")
GORE_SCHEMA = {"type": "object", "properties": {"gore": {"type": "integer"}, "why": {"type": "string"}},
               "required": ["gore", "why"]}


def sb(path):
    U, K = os.environ["SUPABASE_URL"], os.environ["SUPABASE_SECRET_KEY"]
    out, step = [], 1000
    for frm in range(0, 10_000_000, step):
        req = urllib.request.Request(f"{U}/rest/v1/{path}",
            headers={"apikey": K, "Authorization": f"Bearer {K}", "Range": f"{frm}-{frm + step - 1}"})
        b = json.load(urllib.request.urlopen(req)); out += b
        if len(b) < step:
            return out
    return out


def build_strong(args):
    pool = {r["candidate_id"]: r for r in csv.DictReader(open(args.pool_csv))}
    rows = []
    for line in args.gemini_jsonl.read_text().splitlines():
        g = json.loads(line)
        p = pool.get(g["candidate_id"])
        if not p or g.get("error"):
            continue
        rows.append({**p, "stim_quality": g["stim_quality"], "event": g["event"], "self_contained": g["self_contained"]})
    # duration gate (recut rows carry dur_diff; probe the rest)
    face_root, stim_root = args.data_root / "face_origin", args.data_root / "stimuli"
    cache = {}
    for r in rows:
        dd = str(r.get("dur_diff", "")).strip()
        if dd:
            r["_ok"] = float(dd) <= 0.5
        else:
            v = r["video_id"]
            if v not in cache:
                df, ds = sel.probe_dur(face_root / f"{v}.mp4"), sel.probe_dur(stim_root / f"{v}.mp4")
                cache[v] = abs(df - ds) if df > 0 and ds > 0 else 999
            r["_ok"] = cache[v] <= 0.5
    return [r for r in rows if r["_ok"] and int(r["stim_quality"]) >= 2 and r["self_contained"]]


def gore_rate(rows, args):
    from google import genai
    from google.genai import types
    cfg.require_env("GOOGLE_CLOUD_PROJECT")
    os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
    client = genai.Client()
    gen_cfg = types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json",
        response_schema=GORE_SCHEMA, media_resolution=types.MediaResolution.MEDIA_RESOLUTION_LOW)
    stim_root = args.data_root / "stimuli"
    lock = threading.Lock(); out = {}
    with tempfile.TemporaryDirectory() as tmp:
        def one(r):
            cid = r["candidate_id"]
            try:
                src = (args.recut_dir if r.get("stim_source") == "recut" else stim_root) / f'{r["video_id"]}.mp4'
                clip = Path(tmp) / f"{cid}.mp4"
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{max(0.0, float(r['stim_s'])-1):.2f}",
                    "-to", f"{float(r['stim_e'])+1:.2f}", "-i", str(src), "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "26", "-an", "-movflags", "+faststart", str(clip)], check=True, capture_output=True, timeout=120)
                v = types.Part(inline_data=types.Blob(data=clip.read_bytes(), mime_type="video/mp4"))
                resp = client.models.generate_content(model=args.model,
                    contents=[types.Content(role="user", parts=[v, types.Part(text=GORE_PROMPT)])], config=gen_cfg)
                clip.unlink(missing_ok=True)
                with lock:
                    out[cid] = int(json.loads(resp.text)["gore"])
            except Exception as ex:  # noqa: BLE001
                with lock:
                    out[cid] = 0  # fail-open; scene/human review still catches it
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(one, rows))
    return out


def main():
    args = parse_args()
    used = {t["candidate_id"] for t in sb("gold_rationale_tasks?select=candidate_id")}
    deleted = {r["candidate_id"] for r in csv.DictReader(open(args.deleted_csv))}
    used_vids = {t["video_id"] for t in sb("gold_rationale_tasks?select=video_id")}
    strong = build_strong(args)
    avail = [r for r in strong if r["candidate_id"] not in used and r["candidate_id"] not in deleted
             and r["video_id"] not in used_vids]  # also avoid same-video near-dupes of kept clips
    print(f"[backfill] strong {len(strong)} -> available (excl used/deleted/used-videos) {len(avail)}")

    want = args.need + args.buffer
    # take a generous shortlist, gore-filter, then balance to `want`
    shortlist = sel.balanced_pick(avail, want * 2, 0.30)
    if args.dry_run:
        print(f"  shortlist {len(shortlist)} (would gore-rate then keep {want})")
        return 0
    gore = gore_rate(shortlist, args)
    clean = [r for r in shortlist if gore.get(r["candidate_id"], 0) <= args.gore_max]
    dropped = len(shortlist) - len(clean)
    picked = sel.balanced_pick(clean, want, 0.30)

    cols = ["candidate_id", "video_id", "domain", "creator", "start_s", "end_s", "reaction_mean",
            "reaction_std", "stim_quality", "self_contained", "match_ready", "stim_s", "stim_e",
            "dur_diff", "stim_source", "event", "stim_text"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(picked)
    from collections import Counter
    print(f"[backfill] gore-dropped {dropped}; wrote {len(picked)} -> {args.out}; by domain {dict(Counter(r['domain'] for r in picked))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
