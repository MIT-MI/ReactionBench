#!/usr/bin/env python3
"""Hand the human scores back to the variants that proposed each candidate.

Precision@2 is computable from the scoring pass alone. Recall is not: it needs the
sweep pass, which supplies the moments every variant missed. If no sweep file is
present this reports precision only and says so, rather than quietly reporting a
recall figure whose denominator is just the candidates the variants happened to find.

Reports three cuts, because they answer different questions:
  * all clips -- the headline
  * excluding clips the reviewer flagged as not genuine reaction footage
  * restricted to the creators who dominate the pool, since the sample is
    creator-balanced by design and the full pool is not

Output: $RVB_WORK_DIR/variant_scores.json (+ tables on stdout)
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "common"))
import rvb_config as cfg  # noqa: E402  (loads .env, puts every stage dir on sys.path)

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

HIGHLIGHT_THRESHOLD = 2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--review_dir", type=Path, default=cfg.work("review"))
    p.add_argument("--reviewer", default=os.environ.get("USER", "reviewer1"))
    p.add_argument("--sample_csv", type=Path, default=cfg.work("pilot_sample.csv"))
    p.add_argument("--out", type=Path, default=cfg.work("variant_scores.json"))
    p.add_argument("--top_creators", type=int, default=7,
                   help="how many of the largest creators define the 'dominant creators' cut")
    p.add_argument("--exclude_clips", nargs="*", default=None,
                   help="clip ids to drop; defaults to the clips the reviewer flagged in notes")
    return p.parse_args()


def load_scores(path: Path) -> Dict[str, dict]:
    """Last record per candidate wins -- revisions are appended, not overwritten."""
    latest: Dict[str, dict] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            latest[rec["candidate_id"]] = rec
    return latest


def variant_table(
    attribution: Dict[str, dict],
    scores: Dict[str, dict],
    keep_clips: Optional[set] = None,
) -> pd.DataFrame:
    rows = []
    per_variant: Dict[str, List[int]] = defaultdict(list)
    for cid, attr in attribution.items():
        rec = scores.get(cid)
        if rec is None:
            continue
        if keep_clips is not None and rec["clip_id"] not in keep_clips:
            continue
        for variant in attr["variants"]:
            per_variant[variant].append(rec["score"])

    for variant, vals in sorted(per_variant.items()):
        arr = np.array(vals)
        rows.append({
            "variant": variant,
            "n_nom": len(arr),
            "mean_score": round(float(arr.mean()), 2),
            "P@>=2": round(float((arr >= HIGHLIGHT_THRESHOLD).mean()), 3),
            "P@3": round(float((arr == 3).mean()), 3),
            "n_zero": int((arr == 0).sum()),
        })
    return pd.DataFrame(rows).sort_values("P@>=2", ascending=False)


def main() -> int:
    args = parse_args()
    attribution = json.loads((args.review_dir / "attribution.json").read_text())
    scores = load_scores(args.review_dir / f"scores_{args.reviewer}.jsonl")
    sample = pd.read_csv(args.sample_csv, dtype={"id": str})

    unscored = sorted(set(attribution) - set(scores))
    print(f"[scores] {len(scores)}/{len(attribution)} candidates scored"
          + (f" | unscored: {unscored}" if unscored else ""))

    flagged = sorted({rec["clip_id"] for rec in scores.values() if rec.get("note")})
    excluded = set(args.exclude_clips) if args.exclude_clips is not None else set(flagged)
    print(f"[scores] clips flagged in reviewer notes: {flagged}")

    all_clips = set(sample["id"])
    big = set(sample["creator"].value_counts().head(args.top_creators).index)
    big_clips = set(sample.loc[sample["creator"].isin(big), "id"])

    cuts = {
        "all_clips": all_clips,
        "excluding_flagged": all_clips - excluded,
        f"top{args.top_creators}_creators_only": big_clips,
    }

    tables = {}
    for name, keep in cuts.items():
        table = variant_table(attribution, scores, keep)
        tables[name] = table.to_dict("records")
        print(f"\n=== {name} ({len(keep)} clips) ===")
        print(table.to_string(index=False))

    dist = Counter(rec["score"] for rec in scores.values())
    per_clip_max = defaultdict(int)
    for rec in scores.values():
        per_clip_max[rec["clip_id"]] = max(per_clip_max[rec["clip_id"]], rec["score"])
    with_highlight = sum(1 for v in per_clip_max.values() if v >= HIGHLIGHT_THRESHOLD)

    print(f"\n[scores] human score distribution: {dict(sorted(dist.items()))}")
    print(f"[scores] clips with at least one candidate rated >=2: {with_highlight}/{len(per_clip_max)}")

    sweep_path = args.review_dir / f"sweep_{args.reviewer}.jsonl"
    if sweep_path.exists():
        print(f"[scores] sweep file present: {sweep_path}")
    else:
        print("[scores] NO sweep file -- recall is not computable. The numbers above are "
              "precision only: they say how much of what a variant proposed was real, not "
              "how much of what was real the variant found.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "reviewer": args.reviewer,
        "n_scored": len(scores),
        "n_candidates": len(attribution),
        "unscored": unscored,
        "flagged_clips": flagged,
        "excluded_clips": sorted(excluded),
        "score_distribution": {str(k): v for k, v in sorted(dist.items())},
        "clips_with_highlight": with_highlight,
        "recall_computable": sweep_path.exists(),
        "tables": tables,
    }, indent=2))
    print(f"\n[scores] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
