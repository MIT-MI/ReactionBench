#!/usr/bin/env python3
"""Compare pilot variants against each other, before any human judgement exists.

Nothing here says which variant is *right* -- only the blind review can. What it does
say is how much the variants actually differ, which decides whether the blind review is
worth running at all and which arms are redundant.

Reports per variant: empty-list rate, nominations per clip, anchored-score distribution,
and mean pairwise agreement with every other variant (temporal IoU at a matching
threshold). Two variants that agree at ~1.0 are the same variant.

Output: $RVB_WORK_DIR/variant_comparison.json (+ a table on stdout)
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "common"))
import rvb_config as cfg  # noqa: E402  (loads .env, puts every stage dir on sys.path)

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nominations_dir", type=Path, default=cfg.work("nominations"))
    p.add_argument("--sample_csv", type=Path, default=cfg.work("pilot_sample.csv"))
    p.add_argument("--out", type=Path, default=cfg.work("variant_comparison.json"))
    p.add_argument("--iou_threshold", type=float, default=0.5)
    return p.parse_args()


def load(nominations_dir: Path) -> Dict[str, Dict[str, List[dict]]]:
    variants: Dict[str, Dict[str, List[dict]]] = {}
    for path in sorted(nominations_dir.glob("*.jsonl")):
        by_clip: Dict[str, List[dict]] = {}
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("error"):
                continue
            by_clip[rec["id"]] = rec["nominations"]
        variants[path.stem] = by_clip
    return variants


def iou(a: dict, b: dict) -> float:
    lo = max(a["start_s"], b["start_s"])
    hi = min(a["end_s"], b["end_s"])
    inter = max(0.0, hi - lo)
    union = (a["end_s"] - a["start_s"]) + (b["end_s"] - b["start_s"]) - inter
    return inter / union if union > 0 else 0.0


def agreement(x: List[dict], y: List[dict], threshold: float) -> float:
    """Symmetric fraction of nominations that find a partner above the IoU threshold.

    Two empty lists count as full agreement: both variants said "nothing here", which
    is a real and deliberate answer, not a missing one.
    """
    if not x and not y:
        return 1.0
    if not x or not y:
        return 0.0
    matched_x = sum(1 for a in x if any(iou(a, b) >= threshold for b in y))
    matched_y = sum(1 for b in y if any(iou(a, b) >= threshold for a in x))
    return (matched_x + matched_y) / (len(x) + len(y))


def main() -> int:
    args = parse_args()
    variants = load(args.nominations_dir)
    if not variants:
        raise SystemExit(f"no nomination files in {args.nominations_dir}")
    names = sorted(variants)
    clips = sorted(set().union(*(set(v) for v in variants.values())))
    print(f"[compare] {len(names)} variants over {len(clips)} clips\n")

    # The runner appends as it goes, so a file read mid-run looks like a variant that
    # simply nominated less. Refuse to compare those silently.
    partial = {n: len(variants[n]) for n in names if len(variants[n]) < len(clips)}
    if partial:
        print(f"[compare] WARNING: incomplete variants (clips < {len(clips)}): {partial}\n", flush=True)

    per_variant = {}
    for name in names:
        by_clip = variants[name]
        counts = [len(by_clip.get(c, [])) for c in clips if c in by_clip]
        # Only the LLM variants score on the anchored 0-3 scale; V0's "score" is a
        # continuous AU total and must not be histogrammed alongside them.
        raw = [n["score"] for c in by_clip for n in by_clip[c] if "score" in n]
        anchored = [int(s) for s in raw if float(s).is_integer() and 0 <= s <= 3]
        scores = Counter(anchored) if len(anchored) == len(raw) else Counter()
        per_variant[name] = {
            "clips": len(by_clip),
            "empty_rate": float(np.mean([c == 0 for c in counts])) if counts else None,
            "noms_per_clip": float(np.mean(counts)) if counts else None,
            "anchored_score_counts": {str(k): v for k, v in sorted(scores.items())},
            "mean_raw_score": float(np.mean(raw)) if raw else None,
            "flagged": sum(1 for c in by_clip for n in by_clip[c] if n.get("flags")),
        }

    pairwise: Dict[str, float] = {}
    for a, b in itertools.combinations(names, 2):
        shared = [c for c in clips if c in variants[a] and c in variants[b]]
        if not shared:
            continue
        pairwise[f"{a}|{b}"] = float(np.mean([
            agreement(variants[a][c], variants[b][c], args.iou_threshold) for c in shared
        ]))

    table = pd.DataFrame([
        {"variant": n,
         "empty%": round(100 * per_variant[n]["empty_rate"], 0),
         "noms/clip": round(per_variant[n]["noms_per_clip"], 2),
         **{f"score{k}": v for k, v in per_variant[n]["anchored_score_counts"].items()}}
        for n in names
    ]).fillna(0)
    print(table.to_string(index=False))

    print(f"\n[compare] pairwise agreement (IoU>={args.iou_threshold}, empty==empty counts as agreement)")
    matrix = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        matrix.loc[a, a] = 1.0
    for key, value in pairwise.items():
        a, b = key.split("|")
        matrix.loc[a, b] = matrix.loc[b, a] = value
    print(matrix.round(2).to_string())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "iou_threshold": args.iou_threshold,
        "n_clips": len(clips),
        "per_variant": per_variant,
        "pairwise_agreement": pairwise,
    }, indent=2))
    print(f"\n[compare] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
