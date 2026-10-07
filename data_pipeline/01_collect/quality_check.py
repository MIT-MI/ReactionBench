#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset quality + summary pipeline for the Human Reaction dataset.

Features
- Reads 1+ manifest CSVs (column names may vary slightly)
- Rewrites "remote" absolute paths to a local base using a robust heuristic
- Verifies raw video(s) (exists, decodable by ffprobe, has audio stream)
- Enforces min duration threshold
- Optionally validates clips/stimuli/masked/face entries via *_df_path CSVs
- Aggregates summaries: overall, per category/label, per youtuber, per playlist
- Writes multiple CSV reports + a JSON snapshot for programmatic reuse
- Progress bars via tqdm

Example
python data_pipeline/01_collect/quality_check.py \
  --manifests path/to/m1.csv path/to/m2.csv \
  --local-base $RVB_RAW_ROOT \
  --min-seconds 8 \
  --check-clips \
  --out-dir ./reports
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "common"))
import rvb_config as cfg  # noqa: E402  (loads .env, puts every stage dir on sys.path)

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

__all__ = ["run_quality"]


# ---------------------------
# ffprobe helpers
# ---------------------------

def run_ffprobe_json(path: str) -> Optional[Dict[str, Any]]:
    """Return ffprobe JSON for media file or None if probe fails."""
    try:
        # Basic format + stream probe; -show_error for better diagnostics
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration:stream=index,codec_type,codec_name,width,height,channels",
            "-show_error",
            "-of", "json",
            path
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return json.loads(out.decode("utf-8", errors="ignore"))
    except subprocess.CalledProcessError:
        return None
    except FileNotFoundError:
        # ffprobe not installed
        return None


def probe_media(path: str) -> Dict[str, Any]:
    """Probe media and return a compact dict with quality info."""
    info = {
        "exists": os.path.exists(path),
        "probed": False,
        "duration_sec": np.nan,
        "has_video": False,
        "has_audio": False,
        "width": np.nan,
        "height": np.nan,
        "audio_channels": np.nan,
        "video_codec": None,
        "audio_codec": None,
    }
    if not info["exists"]:
        return info

    j = run_ffprobe_json(path)
    if not j:
        return info

    info["probed"] = True
    # duration
    try:
        dur = float(j.get("format", {}).get("duration", "nan"))
    except Exception:
        dur = math.nan
    info["duration_sec"] = dur

    # streams
    for s in j.get("streams", []):
        ctype = s.get("codec_type")
        if ctype == "video":
            info["has_video"] = True
            info["width"] = s.get("width", np.nan)
            info["height"] = s.get("height", np.nan)
            info["video_codec"] = s.get("codec_name")
        elif ctype == "audio":
            info["has_audio"] = True
            info["audio_codec"] = s.get("codec_name")
            info["audio_channels"] = s.get("channels", np.nan)

    return info


# ---------------------------
# Path normalization
# ---------------------------

HUMAN_REACTION_ANCHOR = "/Data/human_reaction_data"

def normalize_path(p: str, local_base: str) -> str:
    """
    Heuristic:
    - If p already starts with local_base, return as-is
    - Else, if it contains '/Data/human_reaction_data', replace prefix so that
      the suffix under that anchor is appended to local_base
    - Else, if it already looks relative to dataset root, try to join with local_base
    - Otherwise return original (still checked for existence)
    """
    if not isinstance(p, str) or p.strip() == "":
        return p

    p = p.strip()
    local_base = local_base.rstrip("/")

    if p.startswith(local_base):
        return p

    # Find the anchor in any absolute path (remote or local)
    anchor_idx = p.find(HUMAN_REACTION_ANCHOR)
    if anchor_idx != -1:
        suffix = p[anchor_idx + len(HUMAN_REACTION_ANCHOR):]
        # Ensure single leading slash in suffix join
        suffix = suffix if suffix.startswith("/") else f"/{suffix}"
        return local_base + suffix

    # If it already looks like a relative path under dataset structure, join it
    parts = [
        "raw", "clips", "audio_clean_clips", "face", "masked",
        "stimuli", "transcript"
    ]
    if any(p.startswith(f"{x}/") or f"/{x}/" in p for x in parts):
        return f"{local_base.rstrip('/')}/{p.lstrip('/')}"

    # Otherwise return as-is; we'll still check existence
    return p


# ---------------------------
# Duration parsing helpers
# ---------------------------

DUR_RE = re.compile(r"^(?:(\d{1,2}):)?(\d{1,2}):(\d{2})(?:\.\d+)?$")

def hms_to_seconds(s: Any) -> float:
    """Parse 'HH:MM:SS' or 'MM:SS' to seconds; returns NaN on failure."""
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return math.nan
    s = str(s).strip()
    m = DUR_RE.match(s)
    if not m:
        # Try pure number
        try:
            return float(s)
        except Exception:
            return math.nan
    hh, mm, ss = m.groups()
    h = int(hh) if hh is not None else 0
    return h * 3600 + int(mm) * 60 + int(ss)


# ---------------------------
# Manifest ingestion
# ---------------------------

LIKELY_PATH_COLS = [
    "local_path", "local_path_low", "subtitle_path",
    "comedy_internet_culture_df_path",  # example; will also accept any *_df_path
    "comedy_internet_culture_out_dir",
    "comedy_internet_culture_clean_out_dir",
]

LIKELY_DURATION_COLS = ["duration_seconds", "duration"]

def load_manifests(files: List[str]) -> pd.DataFrame:
    frames = []
    for fp in files:
        # robust CSV reader
        df = pd.read_csv(fp, dtype=str, keep_default_na=False)
        df["__manifest_file"] = fp
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)

    # Standardize column names a bit
    df.columns = [c.strip() for c in df.columns]

    # Normalize some expected columns if present
    # Flexible aliasing
    rename_map = {}
    if "label" in df.columns:
        rename_map["label"] = "category"
    if "youtuber" in df.columns:
        rename_map["youtuber"] = "youtuber"
    if "playlist" in df.columns:
        rename_map["playlist"] = "playlist"
    if "video_link" in df.columns:
        rename_map["video_link"] = "video_url"
    if "video_title" in df.columns:
        rename_map["video_title"] = "video_title"

    df = df.rename(columns=rename_map)

    # Ensure presence
    for col in ["category", "youtuber", "playlist"]:
        if col not in df.columns:
            df[col] = ""

    # Collect any *_df_path columns generically
    df_path_cols = [c for c in df.columns if c.endswith("_df_path")]
    # keep a column listing them to use later
    df["__df_path_cols"] = [df_path_cols] * len(df)

    return df


# ---------------------------
# Scene DF ingestion (optional)
# ---------------------------

def read_scene_df(scene_csv: str) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(scene_csv, dtype=str, keep_default_na=False)
    except Exception:
        return None


def summarize_scene_df(df_scene: pd.DataFrame) -> Dict[str, Any]:
    """
    Expects columns:
    scene_number,start_time,end_time,is_react,is_playing,clip_path,stimuli_out_dir,masked_out_dir,face_out_dir
    (Relaxed: we check presence)
    """
    out = {
        "scenes": len(df_scene),
        "react_scenes": 0,
        "playing_scenes": 0,
        "clips_present": 0,
        "clips_missing": 0,
        "stimuli_dirs": 0,
        "masked_dirs": 0,
        "face_dirs": 0,
    }
    cols = set(df_scene.columns)

    def exists_p(p): return isinstance(p, str) and os.path.exists(p)

    if "is_react" in cols:
        out["react_scenes"] = int((df_scene["is_react"].str.lower() == "true").sum())
    if "is_playing" in cols:
        out["playing_scenes"] = int((df_scene["is_playing"].str.lower() == "true").sum())

    if "clip_path" in cols:
        clip_exists = df_scene["clip_path"].apply(lambda p: os.path.exists(p))
        out["clips_present"] = int(clip_exists.sum())
        out["clips_missing"] = int((~clip_exists).sum())

    if "stimuli_out_dir" in cols:
        out["stimuli_dirs"] = int(df_scene["stimuli_out_dir"].apply(lambda d: os.path.isdir(d)).sum())
    if "masked_out_dir" in cols:
        out["masked_dirs"] = int(df_scene["masked_out_dir"].apply(lambda d: os.path.isdir(d)).sum())
    if "face_out_dir" in cols:
        out["face_dirs"] = int(df_scene["face_out_dir"].apply(lambda d: os.path.isdir(d)).sum())

    return out


# ---------------------------
# Main quality + summary pass
# ---------------------------


def run_quality(
    manifests,
    local_base,
    out_dir="./reports",
    min_seconds=10.0,
    check_clips=False,
    limit=0
):
    """
    Run the Human Reaction dataset quality & summary process.
    Returns:
        2 if videos_missing > 0, 1 if videos_too_short > 0, else 0.
    """
    os.makedirs(out_dir, exist_ok=True)

    df = load_manifests(manifests)
    if limit and limit > 0:
        df = df.iloc[:limit].copy()

    # Identify candidate path columns dynamically
    path_cols = [c for c in df.columns if any(k in c for k in ["path", "out_dir"])]

    # Normalize paths
    for c in path_cols:
        df[c] = df[c].apply(lambda p: normalize_path(p, local_base))

    # Choose raw video path columns if present
    raw_cols = [c for c in ["local_path", "local_path_low"] if c in df.columns]
    if not raw_cols:
        # Try to infer raw path columns from known folder tokens
        raw_cols = [c for c in path_cols if "/raw/" in df[c].astype(str).head(10).str.cat(sep=" ")]

    # Prepare result accumulators
    quality_rows: List[Dict[str, Any]] = []
    scene_rows: List[Dict[str, Any]] = []

    # Determine manifest provided durations if present
    dur_manifest = None
    if "duration_seconds" in df.columns:
        dur_manifest = pd.to_numeric(df["duration_seconds"], errors="coerce")
    elif "duration" in df.columns:
        dur_manifest = df["duration"].apply(hms_to_seconds)
    else:
        dur_manifest = pd.Series([np.nan] * len(df))

    # Iterate videos
    print("Probing raw videos…")
    for idx in tqdm(range(len(df))):
        row = df.iloc[idx].to_dict()
        youtuber = row.get("youtuber", "")
        playlist = row.get("playlist", "")
        category = row.get("category", "")
        video_url = row.get("video_url", "")
        title = row.get("video_title", "")
        manifest_file = row.get("__manifest_file", "")

        # Probe primary (prefer high quality if present)
        primary_path = None
        alt_path = None
        if "local_path" in row and row["local_path"]:
            primary_path = row["local_path"]
        if "local_path_low" in row and row["local_path_low"]:
            alt_path = row["local_path_low"]
            if primary_path is None:
                primary_path = alt_path

        pr_info = probe_media(primary_path) if primary_path else {
            "exists": False, "probed": False, "duration_sec": np.nan,
            "has_video": False, "has_audio": False, "width": np.nan, "height": np.nan,
            "audio_channels": np.nan, "video_codec": None, "audio_codec": None
        }
        lo_info = probe_media(alt_path) if (alt_path and alt_path != primary_path) else None

        # Duration checks
        dur_probe = pr_info["duration_sec"]
        dur_man = dur_manifest.iloc[idx] if idx < len(dur_manifest) else np.nan
        dur_match = False
        dur_delta = np.nan
        if not math.isnan(dur_probe) and not math.isnan(dur_man):
            dur_delta = abs(dur_probe - float(dur_man))
            dur_match = dur_delta <= 2.0  # 2s tolerance

        issues = []
        if not pr_info["exists"]:
            issues.append("raw_missing")
        elif not pr_info["probed"]:
            issues.append("raw_unreadable")
        else:
            if not pr_info["has_video"]:
                issues.append("no_video_stream")
            if not pr_info["has_audio"]:
                issues.append("no_audio_stream")
            if not math.isnan(dur_probe) and dur_probe < min_seconds:
                issues.append("too_short")

        # If primary had problems but low exists+ok, downgrade issue severity
        if issues and lo_info and lo_info["exists"] and lo_info["probed"]:
            # If low is acceptable, note fallback
            if lo_info["has_video"] and lo_info["has_audio"] and (not math.isnan(lo_info["duration_sec"])) and lo_info["duration_sec"] >= min_seconds:
                issues.append("fallback_ok_with_low")

        quality_rows.append({
            "idx": idx,
            "manifest_file": manifest_file,
            "youtuber": youtuber,
            "playlist": playlist,
            "category": category,
            "video_url": video_url,
            "video_title": title,
            "primary_path": primary_path,
            "alt_path": alt_path,
            "exists": pr_info["exists"],
            "probed": pr_info["probed"],
            "has_video": pr_info["has_video"],
            "has_audio": pr_info["has_audio"],
            "width": pr_info["width"],
            "height": pr_info["height"],
            "duration_probe_sec": dur_probe,
            "duration_manifest_sec": float(dur_man) if not math.isnan(dur_man) else np.nan,
            "duration_delta_sec": dur_delta,
            "duration_match_within_2s": dur_match,
            "low_exists": (lo_info or {}).get("exists", False),
            "low_probed": (lo_info or {}).get("probed", False),
            "low_has_video": (lo_info or {}).get("has_video", False),
            "low_has_audio": (lo_info or {}).get("has_audio", False),
            "low_duration_sec": (lo_info or {}).get("duration_sec", np.nan),
            "issues": ";".join(issues)
        })

        # Optional: per-scene / per-clip checks
        if check_clips:
            df_path_cols = row.get("__df_path_cols", [])
            for col in df_path_cols:
                if not row.get(col):
                    continue
                scene_csv = row[col]
                # Normalize scene_csv too (in case it also needs rewriting)
                scene_csv = normalize_path(scene_csv, local_base)
                df_scene = read_scene_df(scene_csv)
                if df_scene is None:
                    scene_rows.append({
                        "idx": idx, "youtuber": youtuber, "playlist": playlist, "category": category,
                        "scene_csv": scene_csv, "ok": False, "reason": "scene_csv_unreadable", **{k: np.nan for k in [
                            "scenes", "react_scenes", "playing_scenes", "clips_present", "clips_missing",
                            "stimuli_dirs", "masked_dirs", "face_dirs"
                        ]}
                    })
                    continue
                # Normalize potential path columns inside scene CSV
                for inner_col in ["clip_path", "stimuli_out_dir", "masked_out_dir", "face_out_dir"]:
                    if inner_col in df_scene.columns:
                        df_scene[inner_col] = df_scene[inner_col].apply(lambda p: normalize_path(p, local_base))

                ssum = summarize_scene_df(df_scene)
                scene_rows.append({
                    "idx": idx, "youtuber": youtuber, "playlist": playlist, "category": category,
                    "scene_csv": scene_csv, "ok": True, **ssum
                })

    # Build quality dataframe
    qdf = pd.DataFrame(quality_rows)

    # Overall duration stats (use probed durations when available, else manifest)
    dur_series = qdf["duration_probe_sec"].copy()
    fallback = qdf["duration_manifest_sec"]
    dur_series = np.where(np.isnan(dur_series), fallback, dur_series)
    qdf["duration_final_sec"] = dur_series

    # Summaries
    def summarize_series(x: pd.Series) -> Dict[str, Any]:
        x = pd.to_numeric(x, errors="coerce")
        x = x[~np.isnan(x)]
        if len(x) == 0:
            return {"count": 0, "mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan, "sum": np.nan}
        return {
            "count": int(x.count()),
            "mean": float(x.mean()),
            "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
            "min": float(x.min()),
            "max": float(x.max()),
            "sum": float(x.sum())
        }

    overall = {
        "videos_total": int(len(qdf)),
        "videos_existing": int(qdf["exists"].sum()),
        "videos_probed_ok": int(qdf["probed"].sum()),
        "videos_with_audio": int(qdf["has_audio"].sum()),
        "videos_with_video": int(qdf["has_video"].sum()),
        "videos_too_short": int(qdf["issues"].str.contains(r"\btoo_short\b", na=False).sum()),
        "videos_missing": int(qdf["issues"].str.contains(r"\braw_missing\b", na=False).sum()),
        "youtuber_count": int(df["youtuber"].nunique()),
        "playlist_count": int(df["playlist"].nunique()),
        "category_count": int(df["category"].nunique()),
        **{f"duration_{k}": v for k, v in summarize_series(qdf["duration_final_sec"]).items()}
    }

    # Per-group summaries
    def group_summary(by: str) -> pd.DataFrame:
        # `qdf` already carries manifest metadata (category/playlist/youtuber).
        # Only join back if the column is missing to avoid duplicate column errors.
        g = qdf if by in qdf.columns else qdf.join(df[[by]], how="left")
        agg = g.groupby(by).agg(
            videos_total=("idx", "count"),
            videos_existing=("exists", "sum"),
            videos_probed_ok=("probed", "sum"),
            videos_with_audio=("has_audio", "sum"),
            videos_with_video=("has_video", "sum"),
            videos_too_short=("issues", lambda s: s.str.contains(r"\btoo_short\b", na=False).sum()),
            duration_count=("duration_final_sec", lambda s: pd.to_numeric(s, errors="coerce").notna().sum()),
            duration_mean=("duration_final_sec", "mean"),
            duration_std=("duration_final_sec", "std"),
            duration_min=("duration_final_sec", "min"),
            duration_max=("duration_final_sec", "max"),
            duration_sum=("duration_final_sec", "sum"),
        ).reset_index()
        # Make numeric nice
        num_cols = [c for c in agg.columns if c != by]
        agg[num_cols] = agg[num_cols].apply(pd.to_numeric, errors="coerce")
        return agg.sort_values("videos_total", ascending=False)

    per_category = group_summary("category")
    per_youtuber = group_summary("youtuber")
    per_playlist = group_summary("playlist")

    # Optional scene report
    if scene_rows:
        scene_df = pd.DataFrame(scene_rows)
    else:
        scene_df = pd.DataFrame(columns=[
            "idx","youtuber","playlist","category","scene_csv","ok","scenes",
            "react_scenes","playing_scenes","clips_present","clips_missing",
            "stimuli_dirs","masked_dirs","face_dirs","reason"
        ])

    # Write reports
    base = Path(out_dir)
    qdf_path = base / "quality_details.csv"
    qdf.to_csv(qdf_path, index=False)

    per_category_path = base / "summary_per_category.csv"
    per_category.to_csv(per_category_path, index=False)

    per_youtuber_path = base / "summary_per_youtuber.csv"
    per_youtuber.to_csv(per_youtuber_path, index=False)

    per_playlist_path = base / "summary_per_playlist.csv"
    per_playlist.to_csv(per_playlist_path, index=False)

    issues_df = qdf[qdf["issues"].astype(str).str.len() > 0].copy()
    issues_path = base / "quality_issues.csv"
    issues_df.to_csv(issues_path, index=False)

    scene_path = base / "scene_clip_summary.csv"
    scene_df.to_csv(scene_path, index=False)

    # JSON snapshot (top-level + quick counts per group)
    snapshot = {
        "overall": overall,
        "by_category": per_category.to_dict(orient="records"),
        "by_youtuber": per_youtuber.to_dict(orient="records"),
        "by_playlist": per_playlist.to_dict(orient="records"),
        "generated_from_manifests": manifests,
        "local_base": local_base,
        "min_seconds": min_seconds,
        "check_clips": check_clips,
    }
    with open(base / "summary_snapshot.json", "w") as f:
        json.dump(snapshot, f, indent=2)

    # Human-readable printout
    print("\n=== OVERALL SUMMARY ===")
    for k, v in overall.items():
        if isinstance(v, float):
            if "duration_" in k:
                print(f"{k:>24}: {v:,.2f}")
            else:
                print(f"{k:>24}: {v:.3f}")
        else:
            print(f"{k:>24}: {v}")

    print("\nWrote:")
    for p in [qdf_path, per_category_path, per_youtuber_path, per_playlist_path, issues_path, scene_path, base / "summary_snapshot.json"]:
        print(" -", p)

    # Exit code hint if there are blocking issues
    videos_missing = int(overall["videos_missing"])
    videos_too_short = int(overall["videos_too_short"])
    if videos_missing > 0:
        return 2
    elif videos_too_short > 0:
        return 1
    else:
        return 0


def main():
    ap = argparse.ArgumentParser(description="Human Reaction dataset quality & summary")
    ap.add_argument("--manifests", nargs="+", required=True, help="Path(s) to manifest CSV(s)")
    ap.add_argument("--local-base", type=str, required=True,
                    help="Local base directory for /Data/human_reaction_data")
    ap.add_argument("--out-dir", type=str, default="./reports", help="Directory to write reports")
    ap.add_argument("--min-seconds", type=float, default=10.0, help="Minimum acceptable duration (sec)")
    ap.add_argument("--check-clips", action="store_true", help="Also parse *_df_path scene CSVs for clip/stimuli/masked/face checks")
    ap.add_argument("--limit", type=int, default=0, help="Optional: limit number of rows for a quick pass (0 = all)")
    args = ap.parse_args()

    exit_code = run_quality(
        manifests=args.manifests,
        local_base=args.local_base,
        out_dir=args.out_dir,
        min_seconds=args.min_seconds,
        check_clips=args.check_clips,
        limit=args.limit
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
