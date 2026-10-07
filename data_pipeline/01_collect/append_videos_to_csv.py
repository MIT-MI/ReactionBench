#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Append given YouTube videos to a catalog CSV.
- Columns: youtuber, playlist, video_link, video_title, label, duration
- Fills: playlist="" (empty), others via yt_dlp metadata where possible
- Deduplicates by canonical video_link (youtube.com/watch?v=VIDEO_ID)
- Creates CSV if not exists
Usage:
  pip install yt-dlp pandas
  python append_videos_to_csv.py --csv /path/to/youtube_catalog.csv --label "" 
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "common"))
import rvb_config as cfg  # noqa: E402  (loads .env, puts every stage dir on sys.path)
import argparse
import csv
import logging
import os
import re
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

import pandas as pd
from yt_dlp import YoutubeDL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger("append_videos")

CSV_COLUMNS = ["youtuber", "playlist", "video_link", "video_title", "label", "duration"]

# ====== Paste URLs here, or pass --urls_file ======
RAW_INPUT = r'''
'''.strip()  # one URL per line, e.g. https://www.youtube.com/watch?v=<VIDEO_ID>
# ========================================================

YDL_OPTS = {
    "quiet": True,
    "skip_download": True,
    "nocheckcertificate": True,
    "ignoreerrors": True,
}


@dataclass
class Row:
    youtuber: str
    playlist: str
    video_link: str
    video_title: str
    label: str
    duration: str


def parse_raw_urls(raw: str) -> List[str]:
    """Extract URLs from a messy, commented, quoted, comma-tailed block."""
    urls = []
    # Split on whitespace and commas, then filter valid-looking URLs
    tokens = re.split(r"[\s,]+", raw)
    for tok in tokens:
        if not tok or tok.startswith("#"):
            continue
        tok = tok.strip().strip('"').strip("'").strip()
        if tok.startswith("#") or not tok:
            continue
        if tok.startswith("http://") or tok.startswith("https://"):
            urls.append(tok)
    return urls


def canonicalize_youtube_url(url: str) -> Tuple[str, Optional[str]]:
    """
    Return (canonical_url, video_id).
    - youtu.be/ID?*  -> https://www.youtube.com/watch?v=ID
    - youtube.com/watch?v=ID&... -> https://www.youtube.com/watch?v=ID
    Drops other params like &t=
    """
    # short form
    m = re.match(r"https?://(?:www\.)?youtu\.be/([A-Za-z0-9_-]+)", url)
    if m:
        vid = m.group(1)
        return f"https://www.youtube.com/watch?v={vid}", vid

    # long form
    m = re.match(r"https?://(?:www\.)?youtube\.com/watch\?([^#]+)", url)
    if m:
        qs = m.group(1)
        # extract v=
        m2 = re.search(r"(?:^|&)v=([A-Za-z0-9_-]+)", qs)
        if m2:
            vid = m2.group(1)
            return f"https://www.youtube.com/watch?v={vid}", vid

    # If not recognized, return as-is
    return url, None


def sec_to_hhmmss(s: Optional[int]) -> str:
    if s is None:
        return ""
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"


def fetch_metadata(url: str) -> dict:
    with YoutubeDL(YDL_OPTS) as ydl:
        return ydl.extract_info(url, download=False)


def to_row(info: Optional[dict], canonical_link: str, default_label: str) -> Row:
    if not info:
        return Row(
            youtuber="",
            playlist="",
            video_link=canonical_link,
            video_title="",
            label=default_label,
            duration="",
        )
    # Prefer channel_url / uploader_url, fallback to uploader text
    youtuber = (
        info.get("channel_url")
        or info.get("uploader_url")
        or info.get("uploader")
        or ""
    )
    title = info.get("title") or ""
    duration = sec_to_hhmmss(info.get("duration"))
    return Row(
        youtuber=youtuber,
        playlist="",
        video_link=canonical_link,
        video_title=title,
        label=default_label,
        duration=duration,
    )


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in CSV_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    # Reorder
    return df[CSV_COLUMNS]


def load_or_create_csv(path: str) -> pd.DataFrame:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        df = pd.read_csv(path, dtype=str).fillna("")
        # Normalize existing links for dedupe
        df["__canonical"] = df["video_link"].map(lambda u: canonicalize_youtube_url(u)[0] if isinstance(u, str) else u)
        df = ensure_columns(df)
        return df
    else:
        return pd.DataFrame(columns=CSV_COLUMNS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to your catalog CSV")
    ap.add_argument("--label", default="", help="Default label to fill (optional)")
    ap.add_argument("--backup", action="store_true", help="Write a .bak alongside before saving")
    ap.add_argument("--urls_file", default=None, help="text file of video URLs (overrides RAW_INPUT)")
    args = ap.parse_args()

    df = load_or_create_csv(args.csv)
    existing = set(df.get("__canonical", pd.Series(dtype=str)))

    urls = parse_raw_urls(open(args.urls_file, encoding="utf-8").read() if args.urls_file else RAW_INPUT)
    log.info("Parsed %d URLs from input.", len(urls))

    new_rows: List[Row] = []
    for url in urls:
        canonical, vid = canonicalize_youtube_url(url)
        if canonical in existing:
            log.info("Skip duplicate: %s", canonical)
            continue

        info = None
        try:
            info = fetch_metadata(canonical)
        except Exception as e:
            log.warning("Metadata fetch failed for %s: %s", canonical, e)

        row = to_row(info, canonical, args.label)
        new_rows.append(row)

    if not new_rows:
        log.info("No new rows to append.")
        # Clean helper column if present
        if "__canonical" in df.columns:
            df = df.drop(columns=["__canonical"])
        df.to_csv(args.csv, index=False, quoting=csv.QUOTE_MINIMAL)
        log.info("CSV saved: %s", args.csv)
        return

    add_df = pd.DataFrame([asdict(r) for r in new_rows])
    combined = pd.concat([df.drop(columns=["__canonical"], errors="ignore"), add_df], ignore_index=True)
    combined = ensure_columns(combined)

    if args.backup and os.path.exists(args.csv):
        bak = args.csv + ".bak"
        combined.to_csv(bak, index=False, quoting=csv.QUOTE_MINIMAL)
        log.info("Backup written: %s", bak)

    combined.to_csv(args.csv, index=False, quoting=csv.QUOTE_MINIMAL)
    log.info("Appended %d rows. CSV saved: %s", len(new_rows), args.csv)


if __name__ == "__main__":
    main()