#!/usr/bin/env python3
"""Fine stimulus-quality pass: Gemini rates how clear the triggering event is.

The coarse pass (rank_stimulus_quality.py) already picked moments whose stimulus timeline has
a novel, uniquely-aligned content segment. It cannot tell whether that segment is actually a
clear, self-contained triggering event (a punchline, a reveal, a crash) versus something vague.
This pass cuts the stimulus window around the aligned segment and asks Gemini to rate it 0-3.

This is SELECTION, not a gold label (LLMs are used for pre-screening only). Its output is combined
with the Stage A reaction strength to pick the final rationale/match set; a human then reviews a
sample and, for match items, does the 1-click ground-truth confirmation.

Reads pool.csv from the coarse pass; cuts stimuli/{video}.mp4 to the event window; low-res,
resumable (already-scored candidate_ids in the output JSONL are skipped).

  python data_pipeline/06_annotate/rank_stimulus_gemini.py --dry_run          # count tokens, project cost
  python data_pipeline/06_annotate/rank_stimulus_gemini.py --limit 3          # smoke test (real calls)
  python data_pipeline/06_annotate/rank_stimulus_gemini.py                    # full pool
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
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from google import genai
from google.genai import types

from build_pilot_frame import default_data_root

PRICES = {"gemini-3.6-flash": (1.50, 7.50)}  # USD / 1M tokens (in, out); see run_gemini_variant.py

# pad the cut around the aligned stimulus segment so the event and the beat after it are visible.
PAD_BEFORE = 1.5
PAD_AFTER = 1.5

PROMPT = """This is a short clip of the STIMULUS a reaction-video reviewer was watching — the
content on their screen, not the reviewer. A human rated the reviewer's reaction to this moment
as strong. Your job is to judge the STIMULUS itself, so we can pick clips where the trigger is
clear enough to annotate.

Rate how clear and self-contained the triggering event is, 0-3:
0 = nothing happens / static shot / no identifiable event a viewer could react to.
1 = something changes but it is vague or needs prior context to make sense.
2 = a clear, identifiable event (a line delivered, an action, a reveal).
3 = a strong, unambiguous, self-contained event (a punchline, a jump scare, a crash, a twist)
    that a first-time viewer would understand on its own.

Give the score, one sentence naming the event, and whether it is self-contained (understandable
without having seen what came before)."""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "stim_quality": {"type": "integer"},
        "event": {"type": "string"},
        "self_contained": {"type": "boolean"},
    },
    "required": ["stim_quality", "event", "self_contained"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", type=Path, default=default_data_root())
    p.add_argument("--pool_csv", type=Path, default=cfg.work("stimulus_ranking/pool.csv"))
    p.add_argument("--out", type=Path, default=cfg.work("stimulus_ranking/gemini.jsonl"))
    p.add_argument("--recut_stim_dir", type=Path,
                   default=cfg.RECUT_ROOT / "stimuli",
                   help="rows with stim_source=recut cut their event window from here (aligned stimulus)")
    p.add_argument("--model", default="gemini-3.6-flash")
    p.add_argument("--media_resolution", default="LOW", choices=["LOW", "MEDIUM", "HIGH"])
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--project", default=os.environ.get("GOOGLE_CLOUD_PROJECT"))
    p.add_argument("--location", default=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"))
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def load_done(path: Path) -> set:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("error") is None:
            done.add(r["candidate_id"])
    return done


def cut_window(src: Path, s: float, e: float, dest: Path) -> None:
    ss = max(0.0, s - PAD_BEFORE)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{ss:.2f}", "-to", f"{e + PAD_AFTER:.2f}",
        "-i", str(src), "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p",
        "-an", "-movflags", "+faststart", str(dest)], check=True, capture_output=True, timeout=300)


def build_request(clip: Path):
    video = types.Part(inline_data=types.Blob(data=clip.read_bytes(), mime_type="video/mp4"))
    return [types.Content(role="user", parts=[video, types.Part(text=PROMPT)])]


def main() -> int:
    args = parse_args()
    started = datetime.now(timezone.utc)
    if not args.project:
        _sys.exit("set GOOGLE_CLOUD_PROJECT (or pass --project); Vertex AI needs a project id")
    os.environ.update(GOOGLE_CLOUD_PROJECT=args.project, GOOGLE_CLOUD_LOCATION=args.location,
                      GOOGLE_GENAI_USE_VERTEXAI="true")
    client = genai.Client()

    rows = [r for r in csv.DictReader(open(args.pool_csv)) if r.get("stim_s") not in ("", None)]
    if args.limit:
        rows = rows[: args.limit]
    done = load_done(args.out)
    todo = [r for r in rows if r["candidate_id"] not in done]
    print(f"[gemini] pool {len(rows)}, {len(done)} done, {len(todo)} to run "
          f"({args.model}, {args.media_resolution})", flush=True)

    config = types.GenerateContentConfig(
        temperature=args.temperature,
        media_resolution=getattr(types.MediaResolution, f"MEDIA_RESOLUTION_{args.media_resolution}"),
        response_mime_type="application/json", response_schema=RESPONSE_SCHEMA)

    stim_root = args.data_root / "stimuli"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    stats = {"ok": 0, "error": 0, "in_tokens": 0, "out_tokens": 0, "q": [0, 0, 0, 0]}

    with tempfile.TemporaryDirectory() as tmp:
        def run_one(r: dict) -> None:
            cid = r["candidate_id"]
            rec = {"candidate_id": cid, "video_id": r["video_id"], "domain": r.get("domain", "")}
            try:
                clip = Path(tmp) / f"{cid}.mp4"
                src_root = args.recut_stim_dir if r.get("stim_source") == "recut" else stim_root
                cut_window(src_root / f'{r["video_id"]}.mp4', float(r["stim_s"]), float(r["stim_e"]), clip)
                if args.dry_run:
                    counted = client.models.count_tokens(model=args.model, contents=build_request(clip),
                        config=types.CountTokensConfig(generation_config=types.GenerationConfig(
                            media_resolution=config.media_resolution)))
                    clip.unlink(missing_ok=True)
                    with lock:
                        stats["in_tokens"] += counted.total_tokens
                    return
                resp = client.models.generate_content(model=args.model, contents=build_request(clip), config=config)
                clip.unlink(missing_ok=True)
                payload = json.loads(resp.text)
                rec.update(stim_quality=int(payload["stim_quality"]), event=payload["event"],
                           self_contained=bool(payload["self_contained"]), error=None)
                u = resp.usage_metadata
                with lock:
                    stats["ok"] += 1
                    stats["q"][max(0, min(3, rec["stim_quality"]))] += 1
                    stats["in_tokens"] += (u.prompt_token_count or 0)
                    stats["out_tokens"] += (u.candidates_token_count or 0)
            except Exception as ex:  # noqa: BLE001 -- log and continue, never silently drop
                rec.update(error=f"{type(ex).__name__}: {ex}")
                with lock:
                    stats["error"] += 1
            if not args.dry_run:
                with lock, open(args.out, "a") as f:
                    f.write(json.dumps(rec) + "\n")

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(run_one, todo))

    price_in, price_out = PRICES.get(args.model, (1.50, 7.50))
    if args.dry_run:
        if stats["error"]:
            print(f"[gemini] dry run FAILED on {stats['error']}/{len(todo)} clips "
                  f"(likely stale ADC -> run `gcloud auth application-default login`)")
            return 1
        est = stats["in_tokens"] / 1e6 * price_in + len(todo) * 60 / 1e6 * price_out
        print(f"[gemini] dry run: {stats['in_tokens']:,} input tokens over {len(todo)} clips "
              f"-> ~${est:.2f} standard, ~${est / 2:.2f} batch")
        return 0
    cost = stats["in_tokens"] / 1e6 * price_in + stats["out_tokens"] / 1e6 * price_out
    print(json.dumps({"ok": stats["ok"], "error": stats["error"], "quality_hist_0_3": stats["q"],
        "in_tokens": stats["in_tokens"], "out_tokens": stats["out_tokens"], "cost_usd": round(cost, 2),
        "seconds": round((datetime.now(timezone.utc) - started).total_seconds(), 1)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
