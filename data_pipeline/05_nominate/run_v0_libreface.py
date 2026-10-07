#!/usr/bin/env python3
"""V0 baseline: nominate the strongest reaction moments from LibreFace AU intensity.

Works off the per-frame curve in ``face_intensity_series/{id}_au_intensity_series.csv``
rather than the segment table, and emits fixed-width windows around the top peaks.

That choice matters for fairness. The segment table's spans are AU-signature runs that
can be tens of seconds long -- on 10 of the 50 pilot clips a segment-based top-2 covered
more than half the clip, which would have made precision meaningless and handed V0 an
unearned recall advantage. Every variant, V0 included, now nominates the same total
amount of video, so Precision@2 / Recall@2 compare like with like.

Emits the shared nomination JSONL schema. This is a pre-screening heuristic, not an
intensity label, and it sees no audio at all -- exactly the asymmetry the
pilot exists to quantify.

Output: $RVB_WORK_DIR/nominations/V0_libreface.jsonl
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "common"))
import rvb_config as cfg  # noqa: E402  (loads .env, puts every stage dir on sys.path)

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from build_pilot_frame import default_data_root


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", type=Path, default=default_data_root())
    p.add_argument("--ids_file", type=Path, default=cfg.work("pilot_ids.txt"))
    p.add_argument("--out", type=Path, default=cfg.work("nominations/V0_libreface.jsonl"))
    p.add_argument("--sample_csv", type=Path, default=cfg.work("pilot_sample.csv"))
    p.add_argument("--top_k", type=int, default=2, help="nominations per clip")
    p.add_argument("--window_s", type=float, default=3.0, help="nomination width, shared by every variant")
    p.add_argument("--smooth_s", type=float, default=0.5, help="centred rolling mean applied before peak-picking")
    return p.parse_args()


def intensity_curve(csv_path: Path, smooth_s: float) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    au_cols = [c for c in df.columns if c.startswith("au_") and c.endswith("_intensity")]
    if df.empty or not au_cols or "time_s" not in df.columns:
        return None

    df = df.sort_values("time_s").reset_index(drop=True)
    total = df[au_cols].sum(axis=1)
    step = float(df["time_s"].diff().median() or 0.125)
    span = max(1, int(round(smooth_s / step)))
    return pd.DataFrame({"time_s": df["time_s"], "score": total.rolling(span, center=True, min_periods=1).mean()})


def nominate(curve: pd.DataFrame, top_k: int, window_s: float) -> List[dict]:
    """Greedy peak-picking with non-overlapping fixed-width windows."""
    clip_end = float(curve["time_s"].max())
    half = window_s / 2
    scores = curve["score"].to_numpy(copy=True)
    times = curve["time_s"].to_numpy()

    picked: List[dict] = []
    while len(picked) < top_k and np.isfinite(scores).any():
        idx = int(np.nanargmax(scores))
        peak_t, peak_v = float(times[idx]), float(scores[idx])
        start = min(max(0.0, peak_t - half), max(0.0, clip_end - window_s))
        end = min(start + window_s, clip_end)
        picked.append({
            "rank": len(picked) + 1,
            "start_s": round(start, 3),
            "end_s": round(end, 3),
            "peak_s": round(peak_t, 3),
            "score": round(peak_v, 4),
            "evidence": f"LibreFace smoothed AU total peaks at {peak_v:.2f} around {peak_t:.1f}s (no audio input)",
        })
        scores[(times >= start - window_s) & (times <= end + window_s)] = np.nan
    return picked


def main() -> int:
    args = parse_args()
    started = datetime.now(timezone.utc)
    ids = [line.strip() for line in args.ids_file.read_text().splitlines() if line.strip()]
    series_dir = args.data_root / "face_intensity_series"
    durations = {}
    if args.sample_csv.exists():
        s = pd.read_csv(args.sample_csv, dtype={"id": str})
        durations = dict(zip(s["id"], s["duration_s"]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".jsonl.tmp")
    n_ok, skipped, coverage = 0, [], []
    with tmp.open("w") as fh:
        for clip_id in ids:
            curve = intensity_curve(series_dir / f"{clip_id}_au_intensity_series.csv", args.smooth_s)
            if curve is None:
                skipped.append(clip_id)
                fh.write(json.dumps({"variant": "V0_libreface", "id": clip_id,
                                     "nominations": [], "error": "no libreface series"}) + "\n")
                continue
            noms = nominate(curve, args.top_k, args.window_s)
            # Share of the *clip*, not of the LibreFace span, so a short series shows up
            # as low coverage rather than as a suspiciously complete-looking baseline.
            clip_s = float(durations.get(clip_id) or curve["time_s"].max() or 1.0)
            coverage.append(sum(n["end_s"] - n["start_s"] for n in noms) / clip_s)
            fh.write(json.dumps({
                "variant": "V0_libreface",
                "id": clip_id,
                "nominations": noms,
                "n_frames": int(len(curve)),
                "error": None,
            }) + "\n")
            n_ok += 1
    tmp.replace(args.out)

    print(f"[V0] {n_ok}/{len(ids)} clips nominated, {len(skipped)} skipped: {skipped}")
    if coverage:
        print(f"[V0] nominated share of clip: median {np.median(coverage):.0%}, max {max(coverage):.0%}")
    print(f"[V0] wrote {args.out} in {(datetime.now(timezone.utc) - started).total_seconds():.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
