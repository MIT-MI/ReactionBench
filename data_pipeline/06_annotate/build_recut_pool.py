#!/usr/bin/env python3
"""Rebuild coarse pool rows for the recut Patreon clips, on the ALIGNED timeline.

The recut stimulus is face-aligned (recut_stim[t] == source[face_offset + t]). But
stimuli_description segment times are on the OLD desynced stimulus timeline, so before we can
re-derive which segment triggered a reaction we must translate them:

    aligned_time = old_time + (stim_offset - face_offset)

With that shift the segments live on the recut/face timeline, and align_stimulus (reaction onset
is a face time) picks the correct trigger. Everything else — novelty, distinct-segment count,
match_ready — matches rank_stimulus_quality. Output is pool.csv-schema plus `stim_source=recut`
and a `dur_diff` column, so select_stage_c can union it in and the fine pass / build can cut the
event window from the recut stimulus overlay.

  python data_pipeline/06_annotate/build_recut_pool.py
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "common"))
import rvb_config as cfg  # noqa: E402  (loads .env, puts every stage dir on sys.path)

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, pstdev

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_pilot_frame import default_data_root  # noqa: E402
import rank_stimulus_quality as rq  # noqa: E402  (reuse sb, load_segments, analyze_video, align_stimulus)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = cfg.work("stimulus_ranking")
    p.add_argument("--data_root", type=Path, default=default_data_root())
    p.add_argument("--recut_manifest", type=Path, default=d / "recut/manifest.jsonl")
    p.add_argument("--stim_offsets", type=Path, default=d / "recut/stim_offsets.jsonl")
    p.add_argument("--recut_stim_dir", type=Path,
                   default=cfg.RECUT_ROOT / "stimuli")
    p.add_argument("--out", type=Path, default=d / "recut_pool.csv")
    return p.parse_args()


def load_jsonl(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("ok"):
                out[r["id"]] = r
    return out


def main() -> int:
    args = parse_args()
    face_off = load_jsonl(args.recut_manifest)      # id -> {offset, face_dur, stim_dur}
    stim_off = load_jsonl(args.stim_offsets)        # id -> {stim_offset}
    vids = sorted(set(face_off) & set(stim_off))
    print(f"[recut-pool] {len(vids)} recut videos with both offsets")

    videos = {g["video_id"]: g for g in rq.sb("gold_videos?select=video_id,domain,creator,duration_s")}
    cands = rq.sb("gold_candidates?select=candidate_id,video_id,start_s,end_s")
    scores = rq.sb("gold_scores?select=candidate_id,score")
    by_cand: dict = defaultdict(list)
    for s in scores:
        by_cand[s["candidate_id"]].append(s["score"])
    cand_by_vid: dict = defaultdict(list)
    for c in cands:
        cand_by_vid[c["video_id"]].append(c)

    desc_root = args.data_root / "stimuli_description"
    rows = []
    skipped = Counter()
    for vid in vids:
        shift = stim_off[vid]["stim_offset"] - face_off[vid]["offset"]
        fdur, sdur = face_off[vid]["face_dur"], face_off[vid]["stim_dur"]
        p = desc_root / f"{vid}.csv"
        if not p.exists():
            skipped["no_desc"] += 1; continue
        segs = rq.load_segments(p)
        for s in segs:                              # translate onto the aligned (recut) timeline
            s["s"] += shift; s["e"] += shift
        va = rq.analyze_video(segs)
        gv = videos.get(vid, {})
        for c in cand_by_vid.get(vid, []):
            raters = by_cand.get(c["candidate_id"], [])
            if len(raters) < 3 or median(raters) < 2:
                skipped["weak_or_few"] += 1; continue
            seg, n_win = rq.align_stimulus(va["content"], c["start_s"])
            rows.append({
                "candidate_id": c["candidate_id"], "video_id": vid,
                "domain": gv.get("domain", ""), "creator": gv.get("creator", ""),
                "start_s": round(c["start_s"], 2), "end_s": round(c["end_s"], 2),
                "n_raters": len(raters),
                "reaction_mean": round(mean(raters), 3), "reaction_std": round(pstdev(raters), 3),
                "stim_aligned": bool(seg), "stim_novel": bool(seg and seg["novel"]),
                "stim_unique": n_win == 1, "n_in_window": n_win,
                "n_distinct_content": va["n_distinct"],
                "match_ready": bool(seg and n_win == 1 and va["n_distinct"] >= 4),
                "stim_s": round(seg["s"], 2) if seg else "", "stim_e": round(seg["e"], 2) if seg else "",
                "dur_diff": round(abs(fdur - sdur), 2),
                "stim_source": "recut",
                "stim_text": seg["desc"] if seg else "",
            })

    novel = [r for r in rows if r["stim_aligned"] and r["stim_novel"]]
    print(json.dumps({"rows": len(rows), "skipped": dict(skipped),
                      "with_novel_aligned": len(novel),
                      "match_ready": sum(r["match_ready"] for r in rows),
                      "by_domain": dict(Counter(r["domain"] for r in rows))}, indent=2))

    # keep only usable (novel aligned) rows, matching pool.csv's fine-pass input contract
    fields = ["candidate_id", "video_id", "domain", "creator", "start_s", "end_s", "n_raters",
              "reaction_mean", "reaction_std", "stim_aligned", "stim_novel", "stim_unique",
              "n_in_window", "n_distinct_content", "match_ready", "stim_s", "stim_e", "dur_diff",
              "stim_source", "stim_text"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        w.writerows(r for r in novel)
    print(f"\nwrote {args.out} ({len(novel)} novel-aligned recut rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
