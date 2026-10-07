#!/usr/bin/env python3
"""Rough Recall@2 for each variant, from the sweep pass.

The recall denominator is every human-verified >=2 moment: the merged candidates the
reviewer scored >=2, plus the sweep marks >=2 that no candidate covers (moments every
variant missed). A variant's recall is the fraction of those moments its own nominations
overlap.

Deliberately rough. The reviewer noted the sweep marks are late -- the key was pressed a
few seconds into a reaction -- so a point mark is matched against a window that reaches
BACK from the mark (``--lag``) more than forward (``--pad``): the true moment is before
the keypress, not after. Span marks (start..end) are used directly. Results are reported
at a few lag values so the sensitivity to this guess is visible.

Output: $RVB_WORK_DIR/recall.json (+ table on stdout)
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "common"))
import rvb_config as cfg  # noqa: E402  (loads .env, puts every stage dir on sys.path)

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

Interval = Tuple[float, float]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--review_dir", type=Path, default=cfg.work("review"))
    p.add_argument("--nominations_dir", type=Path, default=cfg.work("nominations"))
    p.add_argument("--reviewer", default=os.environ.get("USER", "reviewer1"))
    p.add_argument("--out", type=Path, default=cfg.work("recall.json"))
    p.add_argument("--lag", type=float, default=4.0, help="seconds a point mark reaches back (reaction+press lag)")
    p.add_argument("--pad", type=float, default=1.0, help="seconds a point mark reaches forward")
    p.add_argument("--exclude_variants", nargs="*", default=["V2_flash_low_rerun"])
    return p.parse_args()


def overlaps(a: Interval, b: Interval) -> bool:
    return max(a[0], b[0]) < min(a[1], b[1])


def load_jsonl_last(path: Path, key: str) -> Dict[str, dict]:
    latest: Dict[str, dict] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            latest[rec[key]] = rec
    return latest


def moment_intervals(mark: dict, lag: float, pad: float) -> Interval:
    """A sweep mark as a search interval, biased backward for the reviewer's press lag."""
    if mark.get("end_s") is not None:
        return (mark["t_s"] - lag, mark["end_s"])
    return (mark["t_s"] - lag, mark["t_s"] + pad)


def compute(args: argparse.Namespace, lag: float) -> dict:
    scores = load_jsonl_last(args.review_dir / f"scores_{args.reviewer}.jsonl", "candidate_id")
    sweeps = load_jsonl_last(args.review_dir / f"sweep_{args.reviewer}.jsonl", "clip_id")
    pool = {c["candidate_id"]: c for c in json.loads((args.review_dir / "review_pool.json").read_text())}

    skipped = {c for c, r in sweeps.items() if r.get("skip_reason")}

    # Per-variant raw nominations.
    variant_noms: Dict[str, Dict[str, List[Interval]]] = {}
    for path in sorted(args.nominations_dir.glob("*.jsonl")):
        if path.stem in args.exclude_variants:
            continue
        by_clip: Dict[str, List[Interval]] = defaultdict(list)
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("error"):
                continue
            for n in rec["nominations"]:
                if n["start_s"] < n["end_s"]:
                    by_clip[rec["id"]].append((n["start_s"], n["end_s"]))
        variant_noms[path.stem] = by_clip

    # Ground-truth >=2 moments per clip (excluding skipped clips).
    gt: Dict[str, List[Interval]] = defaultdict(list)
    n_scored = n_missed = 0
    for cid, cand in pool.items():
        clip = cand["clip_id"]
        if clip in skipped:
            continue
        rec = scores.get(cid)
        if rec and rec["score"] >= 2:
            gt[clip].append((cand["start_s"], cand["end_s"]))
            n_scored += 1
    for clip, rec in sweeps.items():
        if clip in skipped:
            continue
        for m in rec.get("marks", []):
            if m["score"] < 2:
                continue
            win = moment_intervals(m, lag, args.pad)
            if not any(overlaps(win, g) for g in gt[clip]):
                gt[clip].append(win)
                n_missed += 1

    total = sum(len(v) for v in gt.values())
    rows = []
    for variant, by_clip in variant_noms.items():
        covered = 0
        for clip, moments in gt.items():
            noms = by_clip.get(clip, [])
            for g in moments:
                if any(overlaps(nm, g) for nm in noms):
                    covered += 1
        rows.append({"variant": variant, "recall": round(covered / total, 3), "covered": covered, "total": total})
    rows.sort(key=lambda r: -r["recall"])
    return {"lag": lag, "n_ground_truth": total, "n_from_scored": n_scored,
            "n_from_sweep_misses": n_missed, "n_clips_excluded": len(skipped), "rows": rows}


def main() -> int:
    args = parse_args()
    main_res = compute(args, args.lag)

    print(f"[recall] {main_res['n_ground_truth']} ground-truth >=2 moments "
          f"({main_res['n_from_scored']} scored candidates + {main_res['n_from_sweep_misses']} sweep-only misses); "
          f"{main_res['n_clips_excluded']} not-realtime clips excluded\n")
    print(f"Recall@2 (lag={args.lag}s):")
    for r in main_res["rows"]:
        print(f"  {r['variant']:22} {r['recall']:.3f}  ({r['covered']}/{r['total']})")

    # Sensitivity to the lag guess.
    sens = {lag: {r["variant"]: r["recall"] for r in compute(args, lag)["rows"]} for lag in (2.0, 4.0, 6.0)}
    print("\nSensitivity to the press-lag assumption:")
    print(f"  {'variant':22} lag2   lag4   lag6")
    for r in main_res["rows"]:
        v = r["variant"]
        print(f"  {v:22} {sens[2.0][v]:.3f}  {sens[4.0][v]:.3f}  {sens[6.0][v]:.3f}")

    args.out.write_text(json.dumps({"main": main_res, "sensitivity": sens}, indent=2))
    print(f"\n[recall] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
