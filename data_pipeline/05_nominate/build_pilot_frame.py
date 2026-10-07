#!/usr/bin/env python3
"""Build the Phase 0 pilot sampling frame.

One row per clip in ``face_origin/`` with the covariates the pilot sampler needs:
domain, creator key, duration, and LibreFace expressiveness.

Deliberately reads expressiveness from ``face_intensity_avg/{id}_au_intensity_avg.csv``
rather than ``_meta/avg_intense_videos.csv``: the latter is already a LibreFace-selected
subset, and drawing the sampling frame from it would bias the pilot toward expressive clips.

Outputs (atomic writes):
  $RVB_WORK_DIR/pilot_frame.csv
  $RVB_LOG_DIR/build_pilot_frame_<timestamp>.json
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "common"))
import rvb_config as cfg  # noqa: E402  (loads .env, puts every stage dir on sys.path)

import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

def default_data_root() -> Path:
    """The consolidated clip pool: $RVB_DATA_ROOT (see common/rvb_config.py)."""
    return cfg.DATA_ROOT


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", type=Path, default=default_data_root())
    p.add_argument("--meta_dir", type=Path, default=None, help="default: <data_root>/_meta")
    p.add_argument("--out", type=Path, default=cfg.work("pilot_frame.csv"))
    p.add_argument("--channel_map", type=Path, default=cfg.work("video_url_channel_map.json"),
                   help="output of resolve_video_url_creators.py; collapses per-video keys to channels")
    p.add_argument("--log_dir", type=Path, default=cfg.LOG_DIR)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=None, help="debug: only process the first N ids")
    return p.parse_args()


def domain_from_path(path: str) -> Optional[str]:
    """Recover the source category from the pre-rename path in mapping.csv."""
    m = re.search(r"human_reaction_data\d*/(?:face|face_with_audio|stimuli)/([^/]+)/", path)
    if m:
        return m.group(1)
    if "test_patreon" in path:
        return "Patreon"
    return None


def load_domains(meta_dir: Path) -> Dict[str, str]:
    mp = pd.read_csv(meta_dir / "mapping.csv", dtype=str)
    mp["domain"] = mp["original_face_path"].fillna("").map(domain_from_path)
    return dict(zip(mp["new_id"].str.zfill(5), mp["domain"]))


def load_creators(meta_dir: Path) -> Dict[str, str]:
    raw = json.loads((meta_dir / "youtuber_id_map.json").read_text())
    return {str(vid).zfill(5): creator for creator, ids in raw.items() for vid in ids}


def load_channel_map(path: Path) -> Dict[str, str]:
    """watch?v= creator key -> real channel, from resolve_video_url_creators.py.

    Without this, the per-video keys in youtuber_id_map.json count videos as creators:
    117 of them turn out to be just 4 channels, so any creator cap applied to the raw
    keys would be enforcing nothing.
    """
    if not path.exists():
        return {}
    resolved = json.loads(path.read_text())
    return {
        url: (info.get("channel") or info["channel_id"])
        for url, info in resolved.items()
        if info.get("channel_id")
    }


def load_durations(meta_dir: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    path = meta_dir / "gemini_annotated.jsonl"
    if not path.exists():
        return out
    with path.open() as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            dur = rec.get("duration")
            if dur is not None:
                out[str(rec["video_id"]).zfill(5)] = float(dur)
    return out


def ffprobe_media(path: Path) -> Dict[str, Any]:
    """Duration plus video/audio codec, in a single probe.

    Audio presence is a hard requirement for this pilot, not a nicety: the whole
    question is whether a multimodal model beats a face-only heuristic, and it cannot
    if there is no audio track to hear. Video codec matters because MPEG-4 Part 2
    clips -- 9% of the pool -- cannot be decoded by any browser, so the review UI
    needs to know which clips require transcoding.
    """
    try:
        res = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "format=duration:stream=codec_type,codec_name", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=60,
        )
        info = json.loads(res.stdout)
    except (subprocess.SubprocessError, json.JSONDecodeError):
        return {"duration_s": None, "vcodec": None, "acodec": None}

    streams = info.get("streams", [])
    video = [s.get("codec_name") for s in streams if s.get("codec_type") == "video"]
    audio = [s.get("codec_name") for s in streams if s.get("codec_type") == "audio"]
    try:
        duration = float(info["format"]["duration"])
    except (KeyError, TypeError, ValueError):
        duration = None
    return {
        "duration_s": duration,
        "vcodec": video[0] if video else None,
        "acodec": audio[0] if audio else None,
    }


def series_end_time(csv_path: Path) -> Optional[float]:
    """Last timestamp LibreFace produced for a clip.

    V0 can only nominate inside this span, so a series that stops early is a blind spot
    for the baseline, not just missing metadata. One pilot clip had a 109s video with a
    0.11s series; the sampler uses this to exclude such clips from the frame.
    """
    try:
        df = pd.read_csv(csv_path, usecols=["time_s"])
    except Exception:
        return None
    return float(df["time_s"].max()) if not df.empty else None


def clip_expressiveness(csv_path: Path) -> Optional[Dict[str, float]]:
    """Peak / weighted-mean total AU intensity across a clip's LibreFace segments."""
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if df.empty or "au_intensity_summary" not in df.columns:
        return None

    totals, weights = [], []
    for summary, dur in zip(df["au_intensity_summary"], df.get("duration_s", pd.Series([1.0] * len(df)))):
        try:
            aus = json.loads(summary)
        except (TypeError, json.JSONDecodeError):
            continue
        totals.append(float(sum(aus.values())))
        weights.append(float(dur) if pd.notna(dur) else 1.0)
    if not totals:
        return None

    weight_sum = sum(weights) or 1.0
    return {
        "au_peak": max(totals),
        "au_mean": sum(t * w for t, w in zip(totals, weights)) / weight_sum,
        "n_segments": float(len(totals)),
    }


def write_atomic(df: pd.DataFrame, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(dest)


def main() -> int:
    args = parse_args()
    started = datetime.now(timezone.utc)
    data_root: Path = args.data_root
    meta_dir: Path = args.meta_dir or (data_root / "_meta")

    if not data_root.is_dir():
        sys.exit(f"data_root not found: {data_root}")
    if not meta_dir.is_dir():
        sys.exit(f"meta_dir not found: {meta_dir} (it holds mapping.csv / youtuber_id_map.json from 03_curate)")

    ids = sorted(p.stem for p in (data_root / "face_origin").glob("*.mp4"))
    if args.limit:
        ids = ids[: args.limit]
    print(f"[frame] {len(ids)} clips in face_origin", flush=True)

    id2domain = load_domains(meta_dir)
    id2creator = load_creators(meta_dir)
    id2duration = load_durations(meta_dir)
    channel_map = load_channel_map(args.channel_map)
    print(f"[frame] domains={len(id2domain)} creators={len(id2creator)} durations={len(id2duration)} "
          f"resolved_channels={len(set(channel_map.values()))} from {len(channel_map)} video keys", flush=True)

    intensity_dir = data_root / "face_intensity_avg"
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        expressiveness = list(pool.map(
            lambda i: clip_expressiveness(intensity_dir / f"{i}_au_intensity_avg.csv"), ids
        ))
    print(f"[frame] libreface expressiveness for {sum(e is not None for e in expressiveness)} clips", flush=True)

    series_dir = data_root / "face_intensity_series"
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        series_end = list(pool.map(
            lambda i: series_end_time(series_dir / f"{i}_au_intensity_series.csv"), ids
        ))

    # ffprobe is authoritative. The durations in gemini_annotated.jsonl disagree with the
    # actual media on ~12% of clips, by as much as 45s, which would put clips in the wrong
    # duration stratum and tell the model a clip length that does not exist.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        probed = list(pool.map(lambda i: ffprobe_media(data_root / "face_origin" / f"{i}.mp4"), ids))
    annotated_duration = id2duration
    id2duration = {i: m["duration_s"] for i, m in zip(ids, probed) if m["duration_s"] is not None}
    id2vcodec = {i: m["vcodec"] for i, m in zip(ids, probed)}
    id2acodec = {i: m["acodec"] for i, m in zip(ids, probed)}
    mismatched = [
        {"id": i, "annotated_s": annotated_duration[i], "probed_s": id2duration[i]}
        for i in ids
        if i in annotated_duration and i in id2duration
        and abs(annotated_duration[i] - id2duration[i]) > 0.5
    ]
    print(f"[frame] ffprobe resolved {len(id2duration)}/{len(ids)} durations; "
          f"{len(mismatched)} disagree with gemini_annotated.jsonl by >0.5s", flush=True)

    frame = pd.DataFrame({
        "id": ids,
        "domain": [id2domain.get(i) for i in ids],
        "creator_key": [id2creator.get(i) for i in ids],
        "creator": [channel_map.get(id2creator.get(i), id2creator.get(i)) for i in ids],
        "duration_s": [id2duration.get(i) for i in ids],
        "au_peak": [e["au_peak"] if e else None for e in expressiveness],
        "au_mean": [e["au_mean"] if e else None for e in expressiveness],
        "n_segments": [e["n_segments"] if e else None for e in expressiveness],
        "au_series_end_s": series_end,
        "vcodec": [id2vcodec.get(i) for i in ids],
        "acodec": [id2acodec.get(i) for i in ids],
    })
    frame["has_audio"] = frame["acodec"].notna()
    frame["browser_playable"] = frame["vcodec"].isin(["h264", "vp8", "vp9", "av1"])
    frame["has_libreface"] = frame["au_peak"].notna()
    frame["libreface_coverage"] = (frame["au_series_end_s"] / frame["duration_s"]).clip(upper=1.0)
    write_atomic(frame, args.out)

    log = {
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root),
        "meta_dir": str(meta_dir),
        "output": str(args.out),
        "n_clips": len(frame),
        "n_domain_known": int(frame["domain"].notna().sum()),
        "n_creator_known": int(frame["creator"].notna().sum()),
        "n_duration_known": int(frame["duration_s"].notna().sum()),
        "n_libreface": int(frame["has_libreface"].sum()),
        "n_libreface_coverage_below_0.8": int((frame["libreface_coverage"] < 0.8).sum()),
        "n_no_audio": int((~frame["has_audio"]).sum()),
        "n_not_browser_playable": int((~frame["browser_playable"]).sum()),
        "vcodec_counts": frame["vcodec"].value_counts(dropna=False).to_dict(),
        "acodec_counts": frame["acodec"].value_counts(dropna=False).to_dict(),
        "no_audio_by_domain": frame[~frame["has_audio"]]["domain"].value_counts().to_dict(),
        "ids_missing_domain": frame.loc[frame["domain"].isna(), "id"].tolist()[:200],
        "ids_missing_duration": frame.loc[frame["duration_s"].isna(), "id"].tolist()[:200],
        "duration_source": "ffprobe (authoritative)",
        "n_duration_mismatch_vs_annotated": len(mismatched),
        "duration_mismatches": sorted(mismatched, key=lambda m: -abs(m["annotated_s"] - m["probed_s"]))[:200],
        "domain_counts": frame["domain"].value_counts(dropna=False).to_dict(),
        "creators_per_domain": frame.groupby("domain")["creator"].nunique().to_dict(),
        "raw_creator_keys": int(frame["creator_key"].nunique()),
        "resolved_creators": int(frame["creator"].nunique()),
        "clips_per_creator": frame["creator"].value_counts().to_dict(),
    }
    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / f"build_pilot_frame_{started.strftime('%Y%m%dT%H%M%SZ')}.json"
    log_path.write_text(json.dumps(log, indent=2, default=str))
    print(f"[frame] wrote {args.out} and {log_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
