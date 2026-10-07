#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "common"))
import rvb_config as cfg  # noqa: E402  (loads .env, puts every stage dir on sys.path)

import argparse
import os
import re
import sys
import pandas as pd
from collections import deque

def parse_time_to_seconds(s: str) -> int:
    """
    Supported formats:
        - HH:MM:SS  e.g. 1:23:45 or 01:23:45
        - MM:SS     e.g. 18:11
        - plain number, read as seconds, e.g. 1800
        - with units, e.g. 30m, 1.5h, 90s, 45min, 2h30m (mixed units are supported)
    """
    s = str(s).strip().lower()
    # HH:MM:SS or MM:SS
    if ":" in s:
        parts = [int(float(p)) for p in s.split(":")]
        if len(parts) == 3:
            h, m, sec = parts
            return h*3600 + m*60 + sec
        elif len(parts) == 2:
            m, sec = parts
            return m*60 + sec
        else:
            raise ValueError(f"Unrecognized time with colons: {s}")

    # Mixed units like "2h30m15s"
    unit_re = re.compile(r'(?P<val>\d+(\.\d+)?)(?P<unit>h|hr|hrs|hour|hours|m|min|mins|minute|minutes|s|sec|secs|second|seconds)')
    if unit_re.search(s):
        total = 0.0
        for m in unit_re.finditer(s):
            val = float(m.group('val'))
            u = m.group('unit')
            if u in ('h','hr','hrs','hour','hours'):
                total += val * 3600
            elif u in ('m','min','mins','minute','minutes'):
                total += val * 60
            elif u in ('s','sec','secs','second','seconds'):
                total += val
        return int(round(total))

    # trailing 'm' or 'h' like "30m", "1.5h"
    if s.endswith('m'):
        return int(round(float(s[:-1]) * 60))
    if s.endswith('h'):
        return int(round(float(s[:-1]) * 3600))
    if s.endswith('s'):
        return int(round(float(s[:-1])))

    # plain number => seconds
    if re.fullmatch(r'\d+(\.\d+)?', s):
        # A small integer (<= 24) may well have been meant as hours,
        # but we do not guess: plain numbers are always seconds. Use a unit for minutes or hours.
        return int(round(float(s)))

    raise ValueError(f"Unrecognized time string: {s}")

def parse_duration_col_to_seconds(x) -> int:
    """Convert a CSV duration (usually HH:MM:SS) to seconds; empty or invalid values count as 0."""
    if pd.isna(x):
        return 0
    try:
        return parse_time_to_seconds(str(x))
    except Exception:
        return 0

def main():
    ap = argparse.ArgumentParser(description="Select videos by total duration within a label and mark selected=True.")
    ap.add_argument("--csv", required=True, help="Input CSV file path.")
    ap.add_argument("--label", required=True, help="Label value to filter on (exact match).")
    ap.add_argument("--target", required=True,
                    help="Target total duration (e.g., 30m, 01:30:00, 1800, 1.5h, 90s).")
    ap.add_argument("--shuffle", action="store_true",
                    help="Randomize candidate order before selecting (default: keep original order).")
    ap.add_argument("--out", default=None,
                    help="Optional output path. If omitted, append _selected.csv to input filename.")
    ap.add_argument("--no-balance", dest="balance", action="store_false",
                    help="Disable round-robin balancing across youtubers and playlists (default: balanced).")
    ap.set_defaults(balance=True)
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        print(f"ERROR: CSV not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.csv)

    # Basic sanity checks
    required_cols = {"label", "duration"}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"ERROR: Missing required column(s): {missing}", file=sys.stderr)
        sys.exit(1)

    # Create selected column if not present
    if "selected" not in df.columns:
        df["selected"] = False

    # Filter rows by target label
    mask_label = df["label"] == args.label
    df_label = df[mask_label].copy()

    if df_label.empty:
        print(f"WARNING: No rows found for label == {args.label}. Nothing to select.")
        out_path = args.out or _default_out(args.csv)
        df.to_csv(out_path, index=False)
        print(f"Wrote (unchanged) to: {out_path}")
        return

    # Within label, only consider rows that are explicitly False (not True or Skipped)
    # Only "False" rows can be updated to "True"; "Skipped" rows should remain unchanged
    def is_false_and_selectable(val):
        if pd.isna(val):
            return True  # Treat NaN as selectable (equivalent to False)
        return str(val).strip().lower() == 'false'
    
    candidates = df_label[df_label["selected"].apply(is_false_and_selectable)].copy()
    if candidates.empty:
        print(f"INFO: All rows under label '{args.label}' are already selected. Nothing to change.")
        out_path = args.out or _default_out(args.csv)
        df.to_csv(out_path, index=False)
        print(f"Wrote (unchanged) to: {out_path}")
        return

    # Parse durations
    candidates["duration_seconds"] = candidates["duration"].apply(parse_duration_col_to_seconds)

    # Optionally shuffle
    if args.shuffle:
        candidates = candidates.sample(frac=1.0, random_state=42)

    target_seconds = parse_time_to_seconds(args.target)

    total = 0
    picked_idx = []

    if args.balance:
        # Hierarchical round-robin: first balance across youtubers, then across their playlists
        # Build queues of indices per (youtuber -> playlist)
        # Use empty strings for missing values to keep grouping stable
        cand = candidates.copy()
        # Preserve the current order in `cand` (already shuffled if --shuffle)
        cand["_y"] = cand.get("youtuber", "").fillna("").astype(str)
        cand["_p"] = cand.get("playlist", "").fillna("").astype(str)

        # Group indices
        by_y_p = {}
        for idx, row in cand.iterrows():
            y = row["_y"]
            p = row["_p"]
            by_y_p.setdefault(y, {})
            by_y_p[y].setdefault(p, [])
            by_y_p[y][p].append(idx)

        # Order of youtubers (preserve first appearance)
        y_order = [y for y in cand["_y"].tolist() if y in by_y_p]
        # De-duplicate while preserving order
        seen = set()
        y_order = [y for y in y_order if not (y in seen or seen.add(y))]

        # For each youtuber, maintain a rotating order of playlists
        pl_orders = {}
        for y in y_order:
            pls = [p for p in cand[cand["_y"] == y]["_p"].tolist() if p in by_y_p[y]]
            # de-duplicate while preserving order
            seen_pl = set()
            pls = [p for p in pls if not (p in seen_pl or seen_pl.add(p))]
            pl_orders[y] = deque(pls)

        # Round-robin selection
        progress = True
        while total < target_seconds and progress:
            progress = False
            for y in list(y_order):
                # Skip youtubers with no remaining items
                if y not in by_y_p:
                    continue
                # Ensure there exists any remaining item under this youtuber
                has_any = any(len(lst) > 0 for lst in by_y_p[y].values())
                if not has_any:
                    continue

                # Rotate through this youtuber's playlists until we find one with remaining items
                if not pl_orders[y]:
                    continue
                rotations = 0
                picked_here = False
                while rotations < len(pl_orders[y]):
                    p = pl_orders[y][0]
                    if by_y_p[y].get(p) and len(by_y_p[y][p]) > 0:
                        idx_pick = by_y_p[y][p].pop(0)
                        dur = int(cand.loc[idx_pick, "duration_seconds"]) if not pd.isna(cand.loc[idx_pick, "duration_seconds"]) else 0
                        picked_idx.append(idx_pick)
                        total += dur
                        progress = True
                        picked_here = True
                        # Move to next playlist for next round
                        pl_orders[y].rotate(-1)
                        break
                    else:
                        pl_orders[y].rotate(-1)
                        rotations += 1
                # If this youtuber had no playable playlists left, continue to next y
                if not picked_here:
                    continue
                if total >= target_seconds:
                    break
        # end while
    else:
        # Original sequential accumulation by current candidate order
        for idx, row in candidates.iterrows():
            dur = int(row["duration_seconds"]) if not pd.isna(row["duration_seconds"]) else 0
            total += dur
            picked_idx.append(idx)
            if total >= target_seconds:
                break

    if not picked_idx:
        print("WARNING: Could not pick any videos (durations may be zero?). No changes made.")
    else:
        print(f"Picked {len(picked_idx)} videos totaling ~{total} seconds (target {target_seconds} seconds).")
        df.loc[picked_idx, "selected"] = True

    out_path = args.out or _default_out(args.csv)
    df.to_csv(out_path, index=False)
    print(f"Wrote updated CSV to: {out_path}")

def _default_out(in_path: str) -> str:
    base, ext = os.path.splitext(in_path)
    return f"{base}_selected.csv"

if __name__ == "__main__":
    main()