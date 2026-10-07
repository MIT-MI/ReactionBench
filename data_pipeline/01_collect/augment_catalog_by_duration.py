#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "common"))
import rvb_config as cfg  # noqa: E402  (loads .env, puts every stage dir on sys.path)

import argparse
import os
import sys
import logging
import pandas as pd
from typing import List, Tuple

CATALOG_DEFAULT = os.environ.get("RVB_CATALOG", "youtube_catalog-new.csv")

def setup_logging(verbosity: int):
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(levelname)s | %(message)s"
    )

def parse_hms_to_seconds(s) -> int:
    """
    Accepts:
    - plain seconds (e.g. '123', 123)
    - 'MM:SS'
    - 'HH:MM:SS'
    - fractional seconds (rounded)
    """
    if pd.isna(s):
        return 0
    if isinstance(s, (int, float)):
        return int(round(float(s)))
    s = str(s).strip()
    # Plain number
    if s.replace('.', '', 1).isdigit():
        return int(round(float(s)))
    parts = s.split(":")
    try:
        if len(parts) == 2:
            m, sec = parts
            return int(m) * 60 + int(round(float(sec)))
        elif len(parts) == 3:
            h, m, sec = parts
            return int(h) * 3600 + int(m) * 60 + int(round(float(sec)))
    except Exception:
        pass
    # Unparseable -> 0
    return 0

def seconds_to_hhmmss(total: int) -> str:
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def ensure_selected_column(df: pd.DataFrame) -> pd.DataFrame:
    if "selected" not in df.columns:
        df["selected"] = False
    return df

def pick_rows_to_meet_duration(candidates: pd.DataFrame, target_seconds: int) -> Tuple[pd.DataFrame, int]:
    """
    Greedy: add rows from shortest to longest until the total is >= target_seconds.
    Returns the chosen rows and their total seconds.
    """
    c = candidates.copy()
    c["__duration_s__"] = c["duration"].apply(parse_hms_to_seconds)
    c = c.sort_values("__duration_s__", ascending=True)
    chosen_rows = []
    acc = 0
    for _, row in c.iterrows():
        d = row["__duration_s__"]
        if d <= 0:
            continue
        chosen_rows.append(row)
        acc += d
        if acc >= target_seconds:
            break
    if not chosen_rows:
        return c.iloc[0:0], 0
    chosen_df = pd.DataFrame(chosen_rows).drop(columns=["__duration_s__"])
    return chosen_df, acc

def build_outpath(input_csv: str) -> str:
    base, ext = os.path.splitext(input_csv)
    return input_csv

def main():
    parser = argparse.ArgumentParser(
        description="Top up a CSV with more videos from youtube_catalog with the same label until the requested duration is reached (new rows get selected=True). The input CSV is updated in place."
    )
    parser.add_argument("csv", help="Target CSV path (must contain a single label value)")
    parser.add_argument("hours", type=float, help="Total duration to add (hours)")
    parser.add_argument("--catalog", default=CATALOG_DEFAULT, help=f"Catalog CSV (default: {CATALOG_DEFAULT})")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase log verbosity (-v or -vv)")
    parser.add_argument("--dry-run", action="store_true", help="Only print how many rows and how much duration would be added; write nothing")
    args = parser.parse_args()

    setup_logging(args.verbose)

    # Read the target CSV
    if not os.path.exists(args.csv):
        logging.error(f"Target CSV does not exist: {args.csv}")
        sys.exit(1)
    try:
        df_target = pd.read_csv(args.csv)
    except Exception as e:
        logging.error(f"Failed to read target CSV: {e}")
        sys.exit(1)

    required_cols = ["youtuber", "playlist", "video_link", "video_title", "label", "duration", "num_reactors"]
    for col in required_cols:
        if col not in df_target.columns:
            logging.error(f"Target CSV is missing a required column: {col}")
            sys.exit(1)

    # The label must be unique
    unique_labels = df_target["label"].dropna().unique().tolist()
    if len(unique_labels) != 1:
        logging.error(f"Target CSV must contain exactly one label value, found: {unique_labels}")
        sys.exit(1)
    label_value = unique_labels[0]
    logging.info(f"Using label = {label_value!r}")

    # Read the catalog
    if not os.path.exists(args.catalog):
        logging.error(f"Catalog CSV does not exist: {args.catalog}")
        sys.exit(1)
    try:
        df_catalog = pd.read_csv(args.catalog)
    except Exception as e:
        logging.error(f"Failed to read catalog CSV: {e}")
        sys.exit(1)

    for col in required_cols:
        if col not in df_catalog.columns:
            logging.error(f"Catalog CSV is missing a required column: {col}")
            sys.exit(1)

    # video_links already in the target CSV
    existing_links = set(str(x).strip() for x in df_target["video_link"].dropna().tolist())

    # Catalog filter: same label & num_reactors==1 & video_link not already in the target
    df_candidates = df_catalog[
        (df_catalog["label"] == label_value) &
        (df_catalog["num_reactors"] == 1) &
        (~df_catalog["video_link"].astype(str).str.strip().isin(existing_links))
    ].copy()

    if df_candidates.empty:
        print("No candidate rows to add (same label, num_reactors==1, video_link not already present).")
        sys.exit(0)

    # Pick rows to reach the target duration
    target_seconds = int(round(args.hours * 3600))
    picked_df, picked_total = pick_rows_to_meet_duration(df_candidates, target_seconds)

    if picked_df.empty or picked_total <= 0:
        print("No candidate rows with a valid duration could be selected.")
        sys.exit(0)

    # Mark the new rows selected=True; make sure the target has a selected column
    df_target = ensure_selected_column(df_target)
    picked_df = ensure_selected_column(picked_df)
    picked_df["selected"] = True

    # Merge and write
    out_path = build_outpath(args.csv)
    # Align columns: missing columns become NaN, extra columns are dropped, order follows the target
    picked_df_aligned = picked_df.reindex(columns=df_target.columns)
    merged = pd.concat([df_target, picked_df_aligned], ignore_index=True)

    msg = (
        f"Adding {len(picked_df)} rows; added duration {seconds_to_hhmmss(picked_total)} "
        f"(target {seconds_to_hhmmss(target_seconds)}, difference {seconds_to_hhmmss(abs(picked_total - target_seconds))})."
    )
    if args.dry_run:
        print("[DRY RUN] " + msg)
        print("[DRY RUN] Nothing written. Output path would be:", out_path)
        sys.exit(0)

    try:
        merged.to_csv(out_path, index=False)
    except Exception as e:
        logging.error(f"Failed to write output: {e}")
        sys.exit(1)

    print(msg)
    print(f"Wrote: {out_path}")
    print(f"Target CSV rows before: {len(df_target)}; rows added: {len(picked_df)}; rows after merge: {len(merged)}.")

if __name__ == "__main__":
    main()