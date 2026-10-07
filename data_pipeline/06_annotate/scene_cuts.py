#!/usr/bin/env python3
"""Count scene cuts in each clip's stimulus window and store on gold_rationale_tasks.scene_cuts.

A splice/compilation stimulus shows up as several scene cuts inside the window; the review tool
surfaces this as a triage flag (it is a hint, not a filter — edited single scenes also cut).

  python data_pipeline/06_annotate/scene_cuts.py --set_csv $RVB_WORK_DIR/stimulus_ranking/backfill_set.csv
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
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_pilot_frame import default_data_root  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = cfg.work("stimulus_ranking")
    p.add_argument("--set_csv", type=Path, default=d / "rationale_set.csv")
    p.add_argument("--out", type=Path, default=d / "scene_cuts.jsonl")
    p.add_argument("--data_root", type=Path, default=default_data_root())
    p.add_argument("--recut_dir", type=Path, default=cfg.RECUT_ROOT / "stimuli")
    p.add_argument("--threshold", type=float, default=0.4)
    return p.parse_args()


def cuts(src: Path, w0: float, w1: float, thr: float) -> int:
    r = subprocess.run(["ffmpeg", "-hide_banner", "-ss", f"{max(0, w0):.2f}", "-to", f"{w1:.2f}", "-i", str(src),
        "-filter:v", f"select='gt(scene,{thr})',metadata=print", "-an", "-f", "null", "-"],
        capture_output=True, text=True)
    return sum(1 for ln in r.stderr.splitlines() if "pts_time" in ln)


def main():
    args = parse_args()
    stim, recut = args.data_root / "stimuli", args.recut_dir
    res = {}
    with open(args.out, "w") as f:
        rows = list(csv.DictReader(open(args.set_csv)))
        for i, r in enumerate(rows):
            src = (recut if r["stim_source"] == "recut" else stim) / f'{r["video_id"]}.mp4'
            ss, se = float(r["stim_s"]), float(r["stim_e"]); st, en = float(r["start_s"]), float(r["end_s"])
            w0 = max(0.0, min(ss, st) - 1.5); w1 = max(se, en) + 1.5
            n = cuts(src, w0, w1, args.threshold) if src.exists() else -1
            res[r["candidate_id"]] = n
            f.write(json.dumps({"candidate_id": r["candidate_id"], "scene_cuts": n}) + "\n"); f.flush()
            if (i + 1) % 25 == 0:
                print(f"{i + 1}/{len(rows)}", flush=True)
    import psycopg
    with psycopg.connect(os.environ["SUPABASE_CONNECTION_STRING"], autocommit=True) as c, c.cursor() as cur:
        cur.executemany("update public.gold_rationale_tasks set scene_cuts=%s where candidate_id=%s",
                        [(n, cid) for cid, n in res.items()])
        cur.execute("notify pgrst, 'reload schema'")
    print(f"[scene] updated {len(res)}; dist {dict(sorted(Counter(res.values()).items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
