#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage:
  python build_clean_id_list.py \
      --audit $RVB_WORK_DIR/qc/reaction_quality_audit_merged.csv \
      --length $RVB_WORK_DIR/qc/stimuli_face_length_report.csv \
      --out $RVB_WORK_DIR/clean_ids/clean_id_list.txt

What it does:
- reads reaction_quality_audit_merged.csv (columns: id,is_playing,is_face_available,percent_frames_without_face,percent_freeze_duration)
- applies a different filter rule to each id range (see below)
- intersects the result with the diff_seconds threshold from stimuli_face_length_report.csv
- writes a txt with one id per line (five digits, zero-padded)
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "common"))
import rvb_config as cfg  # noqa: E402  (loads .env, puts every stage dir on sys.path)

import argparse
import math
import re
from pathlib import Path
import pandas as pd
import numpy as np


def parse_bool_series(s: pd.Series) -> pd.Series:
    """
    Normalize a boolean-like column to True/False/NaN.
    Accepted values: True/False, 1/0, 'true'/'false' (case-insensitive), 'yes'/'no'.
    """
    if s is None:
        return pd.Series([np.nan] * 0)
    def _to_bool(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, bool):
            return x
        if isinstance(x, (int, np.integer)):
            return bool(x)
        xs = str(x).strip().lower()
        if xs in {"true", "t", "1", "yes", "y"}:
            return True
        if xs in {"false", "f", "0", "no", "n"}:
            return False
        return np.nan
    return s.map(_to_bool)


def parse_percent_like(series: pd.Series) -> pd.Series:
    """
    Normalize a percentage field to percentage points: both '2.1%' and 2.1 parse to 2.1.
    Values are not rescaled to a 0-1 fraction.
    """
    def _to_pct(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, float, np.floating)) and not isinstance(x, bool):
            return float(x)
        xs = str(x).strip()
        if xs.endswith('%'):
            try:
                return float(xs[:-1])
            except:
                return np.nan
        try:
            return float(xs)
        except:
            return np.nan
    return series.map(_to_pct)


def parse_seconds(series: pd.Series) -> pd.Series:
    """
    Normalize a seconds field to float (seconds); accepts "0.08" or "0.08s".
    """
    def _to_sec(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, float, np.floating)):
            return float(x)
        xs = str(x).strip().lower()
        if xs.endswith('s'):
            xs = xs[:-1]
        try:
            return float(xs)
        except:
            return np.nan
    return series.map(_to_sec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", required=True, help="Path to reaction_quality_audit_merged.csv")
    ap.add_argument("--length", required=True, help="Path to stimuli_face_length_report.csv")
    ap.add_argument("--out", required=True, help="Output txt path (one id per line)")
    ap.add_argument("--skip-length-filter", action="store_true",
                    help="If set, do not intersect with length report; useful for debugging segment filters.")
    args = ap.parse_args()

    audit = pd.read_csv(args.audit)
    # Normalize column names (guards against case/whitespace differences)
    audit.columns = [c.strip() for c in audit.columns]

    # Required columns
    needed_cols = {"id", "is_playing", "is_face_available", "percent_frames_without_face", "percent_freeze_duration"}
    missing = [c for c in needed_cols if c not in set(audit.columns)]
    if missing:
        raise SystemExit(f"[ERROR] Missing columns in audit csv: {missing}")

    # Parse id as int (tolerates leading zeros)
    def coerce_id(x):
        if pd.isna(x): return np.nan
        try:
            return int(str(x).strip().lstrip('0') or "0")
        except:
            return np.nan

    audit["id"] = audit["id"].map(coerce_id)
    audit = audit.dropna(subset=["id"]).copy()
    audit["id"] = audit["id"].astype(int)

    # Parse booleans and percentages
    is_playing = parse_bool_series(audit["is_playing"])
    is_face_available = parse_bool_series(audit["is_face_available"])
    pfwf = parse_percent_like(audit["percent_frames_without_face"])
    pfd = parse_percent_like(audit["percent_freeze_duration"])

    # Shared thresholds (percentage points)
    pfwf_ok = pfwf < 5.0
    pfd_ok  = pfd  < 10.0
    flags_present = (~is_playing.isna()) & (~is_face_available.isna())
    both_true = (is_playing == True) & (is_face_available == True)

    ids = audit["id"]

    # Id ranges (bounds inclusive)
    # Segment A: 00000-30637; 37973-38475; >44579 -- with flags: both flags True and pfwf_ok and pfd_ok; without flags: pfwf_ok and pfd_ok
    A_mask_range = ((ids >= 0) & (ids <= 30637)) | ((ids >= 37973) & (ids <= 38475)) | (ids > 44579)
    A_pass = np.where(flags_present, (both_true & pfwf_ok & pfd_ok), (pfwf_ok & pfd_ok))
    A_ids = audit.loc[A_mask_range & A_pass, "id"]

    # Segment B: 30638-37972; 38476-42841 -- face and freeze thresholds only (freeze threshold relaxed)
    # Ranges are disjoint: 30637 belongs to A (B starts at 30638)
    B_mask_range = ((ids > 30637) & (ids <= 37972)) | ((ids >= 38476) & (ids <= 42841))
    B_pass = (pfwf < 10.0) & (pfd < 40.0)
    B_ids = audit.loc[B_mask_range & B_pass, "id"]

    # Segment C: 42842-44579 -- face threshold only
    C_mask_range = (ids >= 42842) & (ids <= 44579)
    C_pass = pfwf < 10.0
    C_ids = audit.loc[C_mask_range & C_pass, "id"]

    # Diagnostics: total and passing counts per segment
    A_all_cnt = int(((ids >= 0) & (ids <= 30637)).sum() + ((ids >= 37973) & (ids <= 38475)).sum() + (ids > 44579).sum())
    B_all_cnt = int((((ids > 30637) & (ids <= 37972)) | ((ids >= 38476) & (ids <= 42841))).sum())
    C_all_cnt = int(((ids >= 42842) & (ids <= 44579)).sum())
    print(f"[DBG] segment A total: {A_all_cnt}, A passed: {len(A_ids)}")
    print(f"[DBG] segment B total: {B_all_cnt}, B passed: {len(B_ids)}  (rule: pfwf<10 & pfd<40)")
    print(f"[DBG] segment C total: {C_all_cnt}, C passed: {len(C_ids)}  (rule: pfwf<10)")

    pass_ids_from_audit = pd.Index(A_ids).union(B_ids).union(C_ids)

    # Read the length report and filter on diff_seconds
    length_df = pd.read_csv(args.length)
    length_df.columns = [c.strip() for c in length_df.columns]
    if "id" not in length_df.columns or "diff_seconds" not in length_df.columns:
        raise SystemExit("[ERROR] length csv must contain columns: id, diff_seconds")

    length_df["id"] = length_df["id"].map(coerce_id)
    length_df = length_df.dropna(subset=["id"]).copy()
    length_df["id"] = length_df["id"].astype(int)

    diff = parse_seconds(length_df["diff_seconds"])
    length_ok_ids_global = set(length_df.loc[diff < 0.1, "id"].tolist())
    length_ok_ids_b = set(length_df.loc[diff < 5.0, "id"].tolist())

    # Segment B ids use the looser length threshold (5s); other segments keep 0.1s
    length_ok_ids = set()
    length_ok_ids.update(set(A_ids.tolist()) & length_ok_ids_global)
    length_ok_ids.update(set(B_ids.tolist()) & length_ok_ids_b)
    length_ok_ids.update(set(C_ids.tolist()) & length_ok_ids_global)

    # Diagnostics: effect of intersecting with the length report
    A_len = len(set(A_ids.tolist()) & length_ok_ids)
    B_len = len(set(B_ids.tolist()) & length_ok_ids)
    C_len = len(set(C_ids.tolist()) & length_ok_ids)
    print(f"[DBG] after length intersection: A: {A_len}, B: {B_len}, C: {C_len}  (A/C threshold: <0.1s, B threshold: <5s)")

    # Intersection (the length filter can be skipped for debugging)
    if args.skip_length_filter:
        final_ids = sorted(pass_ids_from_audit.tolist())
        print("[WARN] length filter skipped (--skip-length-filter).")
    else:
        final_ids = sorted([i for i in pass_ids_from_audit.tolist() if i in length_ok_ids])

    # Write output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for i in final_ids:
            f.write(f"{int(i):05d}\n")

    # Also write the two partition files
    out_path_part1 = out_path.with_name(out_path.stem + "_part1.txt")
    out_path_part2 = out_path.with_name(out_path.stem + "_part2.txt")

    part1_ids = [i for i in final_ids if (30638 <= i <= 37972) or (38476 <= i <= 42841)]
    part2_ids = [i for i in final_ids if i not in part1_ids]

    with out_path_part1.open("w", encoding="utf-8") as f1:
        for i in part1_ids:
            f1.write(f"{int(i):05d}\n")

    with out_path_part2.open("w", encoding="utf-8") as f2:
        for i in part2_ids:
            f2.write(f"{int(i):05d}\n")

    # Short summary
    print(f"[OK] passed audit: {len(pass_ids_from_audit)}")
    print(f"[OK] passed length: {len(length_ok_ids)}")
    print(f"[OK] final intersection: {len(final_ids)}")
    print(f"[OK] wrote: {out_path}")

    print(f"[OK] part1 count: {len(part1_ids)} -> {out_path_part1}")
    print(f"[OK] part2 count: {len(part2_ids)} -> {out_path_part2}")

    # Print segment-B ids that failed the length report (first 10)
    print("\n[DBG] segment B: examples failing the length report (first 10, diff_seconds above threshold):")
    b_pass_ids_set = set(B_ids.tolist())  # ids that passed the segment-B audit
    b_failed_len_ids = sorted(b_pass_ids_set - length_ok_ids)  # segment-B ids that failed the length report
    bad_length_b = length_df[length_df["id"].isin(b_failed_len_ids)][
        ["id", "diff_seconds", "stimuli_length", "face_origin_length"]
    ]
    print(f"  segment B passed audit: {len(b_pass_ids_set)}, of which failed length: {len(b_failed_len_ids)}")
    if len(bad_length_b) > 0:
        print(bad_length_b.head(10).to_string(index=False))
    else:
        print("  all segment-B ids satisfy the length threshold.")


if __name__ == "__main__":
    main()