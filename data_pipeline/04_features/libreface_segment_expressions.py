#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch segment facial expressions (or AU sets) from videos using LibreFace.

Usage:
  python libreface_segment_expressions.py /path/to/videos /path/to/output \
      --device cuda:0 --mode emotion --min-duration 0.5

Output:
  - {video_stem}_emo_segments.csv  (emotion mode)
  - {video_stem}_au_segments.csv   (au mode)
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "common"))
import rvb_config as cfg  # noqa: E402  (loads .env, puts every stage dir on sys.path)

import argparse
import multiprocessing as mp
import os
import shutil
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _suppress_progress_bars() -> None:
    """
    Disable tqdm-style progress bars when running in a non-interactive context.

    LibreFace internally uses tqdm to render alignment progress such as
    "Aligning face for video frames...". When stdout/stderr are redirected to a
    log file this creates substantial I/O noise.  We suppress those bars here
    before LibreFace (and therefore tqdm) is imported so that any downstream
    `from tqdm.auto import tqdm` statements see the patched version.
    """

    # Allow callers to opt out explicitly.
    if os.environ.get("HALLO3_DISABLE_PROGRESS", "").lower() in {"0", "false", "no"}:
        return

    should_disable = os.environ.get("HALLO3_DISABLE_PROGRESS")
    if not should_disable:
        # Fall back to TTY detection to decide automatically.
        should_disable = "1" if not sys.stderr.isatty() else ""

    if not should_disable:
        return

    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    try:
        import tqdm  # type: ignore
    except Exception:
        return

    try:
        import tqdm.auto as tqdm_auto  # type: ignore
    except Exception:
        tqdm_auto = None  # type: ignore[assignment]

    try:
        import tqdm.std as tqdm_std  # type: ignore
    except Exception:
        tqdm_std = None  # type: ignore[assignment]

    original_tqdm = tqdm.tqdm
    original_trange = getattr(tqdm, "trange", None)

    class _SilentTqdm(original_tqdm):  # type: ignore[misc]
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("disable", True)
            super().__init__(*args, **kwargs)

    def _silent_tqdm(*args, **kwargs):  # type: ignore[no-redef]
        kwargs.setdefault("disable", True)
        return original_tqdm(*args, **kwargs)

    if isinstance(original_tqdm, type):
        patched = _SilentTqdm
    else:
        patched = _silent_tqdm

    def _silent_trange(*args, **kwargs):
        kwargs.setdefault("disable", True)
        if callable(original_trange):
            return original_trange(*args, **kwargs)
        return patched(range(*args, **kwargs))

    tqdm.tqdm = patched  # type: ignore[assignment]
    tqdm.trange = _silent_trange  # type: ignore[assignment]

    if tqdm_auto is not None:
        tqdm_auto.tqdm = patched  # type: ignore[attr-defined,assignment]
        tqdm_auto.trange = _silent_trange  # type: ignore[attr-defined,assignment]
    if tqdm_std is not None:
        tqdm_std.tqdm = patched  # type: ignore[attr-defined,assignment]
        tqdm_std.trange = _silent_trange  # type: ignore[attr-defined,assignment]


_suppress_progress_bars()

# Importable once installed
import libreface  # PyPI: libreface

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpg", ".mpeg", ".m4v"}

# Expression-code mapping (the CSV keeps both the expression name and its code)
EMO_ORDER = ["Neutral", "Happiness", "Sadness", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]
EMO_CODE = {name: i for i, name in enumerate(EMO_ORDER)}


def _enable_line_buffering() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except Exception:
            pass


_enable_line_buffering()

# ---- Logging helpers ----
def ts() -> str:
    """Timestamp for logs (HH:MM:SS)."""
    return datetime.now().strftime("%H:%M:%S")


def load_include_stems(path: Path):
    """Read a list of video stems (optionally with extensions) from a text file."""
    stems = []
    seen = set()
    with path.open("r") as fh:
        for line in fh:
            token = line.strip()
            if not token or token.startswith("#"):
                continue
            token = token.split()[0]
            stem = Path(token).stem
            if stem and stem not in seen:
                stems.append(stem)
                seen.add(stem)
    return stems


class Heartbeat:
    """Periodic log to show liveness while a long op runs."""
    def __init__(self, prefix: str, interval_sec: float = 15.0):
        self.prefix = prefix
        self.interval = max(0.1, float(interval_sec))
        self._stop = threading.Event()
        self._thr = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        ticks = 0
        while not self._stop.is_set():
            if ticks > 0:
                print(f"[HB {ts()}] {self.prefix} ... still working ({int(ticks*self.interval)}s)", flush=True)
            ticks += 1
            self._stop.wait(self.interval)

    def start(self):
        self._thr.start()

    def stop(self):
        self._stop.set()
        self._thr.join(timeout=1.0)


def list_videos(input_dir: Path):
    for p in sorted(input_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            yield p


def robust_get_df(video_path: Path, device: str, batch_size: int, num_workers: int,
                  temp_dir: Path = None, weights_download_dir: Path = None) -> pd.DataFrame:
    kwargs = dict(device=device, batch_size=batch_size, num_workers=num_workers)
    if temp_dir:
        kwargs["temp_dir"] = str(temp_dir)
    if weights_download_dir:
        kwargs["weights_download_dir"] = str(weights_download_dir)
    df = libreface.get_facial_attributes(str(video_path), **kwargs)
    # Compatibility: convert a non-DataFrame result to a DataFrame
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    return df


def sanitize_time_series(df: pd.DataFrame) -> pd.DataFrame:
    # Detect the time unit (ms vs s) and fix possible time wrap-around
    if "frame_time_in_ms" not in df.columns:
        raise ValueError("LibreFace output is missing the 'frame_time_in_ms' column")

    out = df.copy()
    raw = pd.to_numeric(out["frame_time_in_ms"], errors="coerce")

    # Time unit: a large span (>100) is most likely milliseconds, otherwise seconds
    span = float(raw.max() - raw.min()) if len(raw) else 0.0
    if span > 100.0:
        time_s = raw / 1000.0
    else:
        time_s = raw

    # Fix wrap-around caused by batching (later batches restart from 0)
    t = time_s.to_numpy(dtype=float, copy=True)
    if t.size >= 2:
        offset = 0.0
        for i in range(1, t.size):
            if t[i] < t[i - 1]:
                # On wrap-around, add the previous timestamp as an offset to all later frames
                offset += t[i - 1]
            t[i] += offset

    out["time_s"] = t
    out = out.sort_values("time_s").reset_index(drop=True)
    return out


def downsample_time_series(df: pd.DataFrame, target_fps: Optional[float]) -> pd.DataFrame:
    """Reduce frame rows to the requested FPS while preserving ordering and endpoints."""
    if df.empty:
        return df
    if not target_fps or target_fps <= 0:
        return df

    times = df["time_s"].to_numpy()
    keep_indices = [0]
    interval = 1.0 / float(target_fps)
    last_time = times[0]
    eps = interval * 0.01  # small tolerance for floating point jitter

    for idx in range(1, len(times)):
        if times[idx] - last_time >= interval - eps:
            keep_indices.append(idx)
            last_time = times[idx]

    last_idx = len(df) - 1
    if keep_indices[-1] != last_idx:
        keep_indices.append(last_idx)

    return df.iloc[keep_indices].reset_index(drop=True)


def segment_by_label(times: np.ndarray, labels: np.ndarray, min_duration: float):
    """
    Segment boundaries are the midpoints between adjacent label changes; the last segment is padded by half a frame.
    Short segments are merged into the longer neighbor (not always the previous one), and adjacent same-label segments are merged after each pass.
    Returns a list: [{start, end, duration, label}]
    """
    segments = []
    n = len(times)
    if n == 0:
        return segments

    # Estimate the frame interval (for padding the end)
    diffs = np.diff(times)
    dt = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else 1.0 / 30.0

    # Cut initial segments at the midpoints of label changes
    seg_st = float(times[0])
    cur_lab = labels[0]
    for i in range(1, n):
        if labels[i] != cur_lab:
            boundary = (float(times[i - 1]) + float(times[i])) / 2.0
            segments.append([seg_st, boundary, cur_lab])
            seg_st = boundary
            cur_lab = labels[i]
    # Last segment: up to the last frame plus half a frame
    segments.append([seg_st, float(times[-1]) + 0.5 * dt, cur_lab])

    # Merge adjacent same-label segments
    merged = []
    for s, e, lab in segments:
        if not merged:
            merged.append([s, e, lab])
            continue
        if merged[-1][2] == lab:
            merged[-1][1] = e
        else:
            merged.append([s, e, lab])

    # Repeatedly absorb short segments into the longer neighbor
    while True:
        if len(merged) <= 1:
            break
        changed = False
        for i in range(len(merged)):
            s, e, lab = merged[i]
            dur = e - s
            if dur >= min_duration:
                continue
            # Merge into the longer neighbor
            if i == 0:
                # Merge right
                merged[i + 1][0] = s
                changed = True
                del merged[i]
                break
            elif i == len(merged) - 1:
                # Merge left
                merged[i - 1][1] = e
                changed = True
                del merged[i]
                break
            else:
                prev_dur = merged[i - 1][1] - merged[i - 1][0]
                next_dur = merged[i + 1][1] - merged[i + 1][0]
                if next_dur >= prev_dur:
                    # Merge right
                    merged[i + 1][0] = s
                    changed = True
                    del merged[i]
                    break
                else:
                    # Merge left
                    merged[i - 1][1] = e
                    changed = True
                    del merged[i]
                    break
        # If anything changed this pass, merge adjacent same-label segments
        if changed:
            compact = []
            for s, e, lab in merged:
                if compact and compact[-1][2] == lab:
                    compact[-1][1] = e
                else:
                    compact.append([s, e, lab])
            merged = compact
        else:
            break

    return [
        {
            "start_time_s": float(s),
            "end_time_s": float(e),
            "duration_s": float(e - s),
            "label": lab,
        }
        for s, e, lab in merged
        if e > s
    ]


def emotion_segments(df: pd.DataFrame, min_duration: float) -> pd.DataFrame:
    if "facial_expression" not in df.columns:
        raise ValueError("LibreFace output is missing the 'facial_expression' column (discrete emotion labels)")

    # Keep time & expression columns; treat values outside the 8 classes as missing, then forward/backward fill for a continuous timeline
    work = df[["time_s", "facial_expression"]].copy()
    valid = set(EMO_CODE.keys())
    work.loc[~work["facial_expression"].isin(valid), "facial_expression"] = np.nan
    work["facial_expression"] = work["facial_expression"].ffill().bfill()

    sub = work.dropna()
    sub = sub[sub["facial_expression"].isin(valid)]
    times = sub["time_s"].to_numpy()
    labels = sub["facial_expression"].to_numpy()

    segs = segment_by_label(times, labels, min_duration)
    seg_df = pd.DataFrame(segs)
    if seg_df.empty:
        return seg_df
    seg_df["expression_code"] = seg_df["label"].map(EMO_CODE).astype(int)
    return seg_df


def detect_au_columns(df: pd.DataFrame):
    # e.g. au_1, au_2, ... (presence) and au_1_intensity, ... (intensity)
    au_presence = sorted([c for c in df.columns if c.startswith("au_") and not c.endswith("_intensity")])
    au_intensity = sorted([c for c in df.columns if c.startswith("au_") and c.endswith("_intensity")])
    return au_presence, au_intensity


def au_signature_row(row, au_presence_cols, au_intensity_cols, intensity_thresh):
    active = []
    if au_presence_cols:
        for c in au_presence_cols:
            try:
                v = int(row[c])
            except Exception:
                v = 0
            if v == 1:
                # Column name au_12 -> AU12
                active.append("AU" + c.split("_", 1)[1])
    else:
        # No explicit presence columns: threshold the intensities
        for c in au_intensity_cols:
            try:
                v = float(row[c])
            except Exception:
                v = 0.0
            if v >= intensity_thresh:
                idx = c.split("_", 1)[1].replace("_intensity", "")
                active.append("AU" + idx)
    if not active:
        return "NONE"
    return "+".join(sorted(active))


def au_segments(df: pd.DataFrame, min_duration: float, intensity_thresh: float) -> pd.DataFrame:
    au_presence_cols, au_intensity_cols = detect_au_columns(df)
    if not au_presence_cols and not au_intensity_cols:
        raise ValueError("No AU columns found (neither au_* nor au_*_intensity)")
    work = df.copy()
    work["au_sig"] = work.apply(
        lambda r: au_signature_row(r, au_presence_cols, au_intensity_cols, intensity_thresh), axis=1
    )
    times = work["time_s"].to_numpy()
    labels = work["au_sig"].to_numpy()
    segs = segment_by_label(times, labels, min_duration)
    seg_df = pd.DataFrame(segs).rename(columns={"label": "au_signature"})
    return seg_df


def process_video(video_path: Path, out_dir: Path, mode: str, device: str,
                  batch_size: int, num_workers: int,
                  min_duration: float, au_intensity_thresh: float,
                  target_fps: Optional[float] = None,
                  weights_dir: Path = None, temp_dir: Path = None,
                  verbose: bool = False, heartbeat_seconds: float = 15.0):
    print(f"[INFO] Processing: {video_path.name}")
    print(f"[INFO {ts()}] Starting inference with LibreFace | device={device} | batch_size={batch_size} | workers={num_workers}")
    hb = None
    if heartbeat_seconds and heartbeat_seconds > 0:
        hb = Heartbeat(prefix=f"{video_path.name}", interval_sec=float(heartbeat_seconds))
        hb.start()
    t0 = time.time()
    df = robust_get_df(video_path, device, batch_size, num_workers, temp_dir, weights_dir)
    if hb is not None:
        hb.stop()
    t1 = time.time()
    print(f"[INFO {ts()}] Inference finished in {t1 - t0:.2f}s")
    df = sanitize_time_series(df)
    raw_frames = len(df)
    df = downsample_time_series(df, target_fps)
    n_frames = len(df)
    total_dur = (df["time_s"].iloc[-1] - df["time_s"].iloc[0]) if n_frames > 1 else 0.0
    if target_fps and target_fps > 0:
        approx_fps = (n_frames - 1) / total_dur if total_dur > 0 and n_frames > 1 else 0.0
        print(
            f"[INFO {ts()}] Downsampled frames: {raw_frames} -> {n_frames} (target {target_fps:.2f} fps, actual ~{approx_fps:.2f})"
        )
    print(f"[INFO {ts()}] Frames: {n_frames} | Approx duration: {total_dur:.2f}s")
    if verbose and "facial_expression" in df.columns:
        vc = df["facial_expression"].value_counts(dropna=True).to_dict()
        print(f"[VERB {ts()}] Facial expression counts: {vc}")

    stem = video_path.stem
    results = []

    if mode in ("emotion", "both"):
        emo_df = emotion_segments(df, min_duration)
        emo_df.insert(0, "video_stem", stem)
        emo_df.insert(1, "video_path", str(video_path))
        emo_out = out_dir / f"{stem}_emo_segments.csv"
        emo_df.to_csv(emo_out, index=False)
        results.append(("emotion", emo_out, len(emo_df)))
        if mode in ("emotion", "both") and verbose:
            if not emo_df.empty:
                top3 = emo_df.sort_values("duration_s", ascending=False).head(3)[["label", "duration_s", "start_time_s", "end_time_s"]]
                print(f"[VERB {ts()}] Top-3 longest emotion segments (label, dur, start, end):\n{top3.to_string(index=False)}")

    if mode in ("au", "both"):
        au_df = au_segments(df, min_duration, au_intensity_thresh)
        au_df.insert(0, "video_stem", stem)
        au_df.insert(1, "video_path", str(video_path))
        au_out = out_dir / f"{stem}_au_segments.csv"
        au_df.to_csv(au_out, index=False)
        results.append(("au", au_out, len(au_df)))
        if mode in ("au", "both") and verbose:
            if not au_df.empty:
                top3_au = au_df.sort_values("duration_s", ascending=False).head(3)[["au_signature", "duration_s", "start_time_s", "end_time_s"]]
                print(f"[VERB {ts()}] Top-3 longest AU segments (sig, dur, start, end):\n{top3_au.to_string(index=False)}")

    print(f"[INFO {ts()}] Segmentation done for {video_path.name}")

    for kind, path, n in results:
        print(f"[OK] {kind} -> {path} ({n} segments)")


def auto_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


def expected_output_paths(stem: str, out_dir: Path, mode: str):
    """
    Return a list of expected output CSV Paths for the given video stem and mode.
    Used to decide whether to skip processing if outputs already exist.
    """
    paths = []
    if mode in ("emotion", "both"):
        paths.append(out_dir / f"{stem}_emo_segments.csv")
    if mode in ("au", "both"):
        paths.append(out_dir / f"{stem}_au_segments.csv")
    return paths


def main():
    parser = argparse.ArgumentParser(description="Segment facial expressions/AUs from videos using LibreFace.")
    parser.add_argument("input_dir", type=str, help="Folder containing videos.")
    parser.add_argument("output_dir", type=str, help="Folder to save segments CSVs.")
    parser.add_argument("--mode", choices=["emotion", "au", "both"], default="emotion",
                        help="Segment by emotion labels, AU signatures, or both.")
    parser.add_argument("--min-duration", type=float, default=0.5,
                        help="Minimum segment duration (seconds) to keep (shorter will be absorbed).")
    parser.add_argument("--au-intensity-thresh", type=float, default=3.0,
                        help="AU intensity threshold (0..5) when presence cols are absent.")
    parser.add_argument("--target-fps", type=float, default=4.0,
                        help="Downsample LibreFace outputs to this FPS before segmenting (<=0 to disable).")
    parser.add_argument("--device", type=str, default=auto_device(),
                        help='Compute device, e.g., "cpu", "cuda:0". Default auto.')
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for video frames.")
    parser.add_argument("--num-workers", type=int, default=2, help="Dataloader workers.")
    parser.add_argument("--weights-dir", type=str, default=None, help="Parent folder to store/download weights.")
    parser.add_argument("--temp-dir", type=str, default=None, help="Temp dir used by LibreFace.")
    parser.add_argument("--verbose", action="store_true", help="Print extra summaries (counts, top segments)")
    parser.add_argument("--heartbeat-seconds", type=float, default=15.0, help="Print a liveness line every N seconds during inference (0 to disable)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip videos whose expected output CSV(s) already exist, are non-empty, and are not older than the source video.")
    parser.add_argument("--max-concurrent", type=int, default=1,
                        help="Number of videos to process in parallel (>=1). Each worker loads LibreFace separately.")
    parser.add_argument("--include-list", type=str, default=None,
                        help="Optional path to a text file containing one video ID (stem or filename) per line. Only matching videos are processed.")
    args = parser.parse_args()

    in_dir = Path(args.input_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = list(list_videos(in_dir))
    if args.include_list:
        include_path = Path(args.include_list).expanduser().resolve()
        if not include_path.exists():
            raise FileNotFoundError(f"Include list not found: {include_path}")
        include_stems = load_include_stems(include_path)
        if not include_stems:
            print(f"[WARN {ts()}] Include list is empty: {include_path}")
            sys.exit(0)
        include_set = set(include_stems)
        original_count = len(videos)
        videos = [v for v in videos if v.stem in include_set]
        if not videos:
            print(f"[WARN {ts()}] No videos matched include list ({include_path}).")
            sys.exit(0)
        missing = include_set - {v.stem for v in videos}
        if missing:
            missing_list = ", ".join(sorted(missing)[:5])
            if len(missing) > 5:
                missing_list += ", ..."
            print(f"[WARN {ts()}] {len(missing)} IDs from include list not found under {in_dir}: {missing_list}")
        print(f"[INFO {ts()}] Include list filter matched {len(videos)} of {len(include_set)} requested videos (from {original_count} available).")
    if not videos:
        print(f"[WARN] No videos found under: {in_dir}")
        sys.exit(0)

    # Environment/info header
    libreface_ver = getattr(libreface, "__version__", "unknown")
    try:
        import torch
        torch_info = f"torch {torch.__version__} | CUDA={torch.cuda.is_available()}"
        if torch.cuda.is_available():
            torch_info += f" | device_name={torch.cuda.get_device_name(0)}"
    except Exception:
        torch_info = "torch unavailable"

    print(f"[INFO {ts()}] Device: {args.device} | Mode: {args.mode} | Videos: {len(videos)}")
    print(f"[INFO {ts()}] Output dir: {out_dir}")
    print(f"[INFO {ts()}] libreface {libreface_ver} | {torch_info}")

    to_process = []
    for v in videos:
        if args.skip_existing:
            stem = v.stem
            expected = expected_output_paths(stem, out_dir, args.mode)
            try:
                src_mtime = v.stat().st_mtime
                all_ok = True
                for p in expected:
                    if not p.exists():
                        all_ok = False
                        break
                    if p.stat().st_size <= 0:
                        all_ok = False
                        break
                    if p.stat().st_mtime < src_mtime:
                        all_ok = False
                        break
                if all_ok:
                    print(f"[SKIP {ts()}] {v.name} -> outputs exist and are up-to-date")
                    continue
            except Exception:
                pass
        to_process.append(v)

    if not to_process:
        print(f"[INFO {ts()}] Nothing to process.")
        return

    max_concurrent = max(1, int(args.max_concurrent))
    weights_dir = Path(args.weights_dir).resolve() if args.weights_dir else None
    base_temp_dir = Path(args.temp_dir).resolve() if args.temp_dir else None
    if base_temp_dir:
        base_temp_dir.mkdir(parents=True, exist_ok=True)

    if max_concurrent == 1:
        for v in to_process:
            try:
                process_video(
                    v, out_dir, args.mode, args.device, args.batch_size, args.num_workers,
                    args.min_duration, args.au_intensity_thresh, args.target_fps,
                    weights_dir,
                    base_temp_dir,
                    verbose=args.verbose,
                    heartbeat_seconds=args.heartbeat_seconds,
                )
            except Exception as e:
                print(f"[ERROR] {v.name}: {e}")
                traceback.print_exc()
        return

    print(f"[INFO {ts()}] Parallel segmentation with {max_concurrent} workers.")

    def _submit_video(executor, video_path: Path):
        temp_dir = None
        if base_temp_dir:
            temp_dir = base_temp_dir / f"{video_path.stem}_{uuid.uuid4().hex}"
        return executor.submit(
            _process_video_worker,
            str(video_path),
            str(out_dir),
            args.mode,
            args.device,
            args.batch_size,
            args.num_workers,
            args.min_duration,
            args.au_intensity_thresh,
            args.target_fps,
            str(weights_dir) if weights_dir else None,
            str(temp_dir) if temp_dir else None,
            args.verbose,
            args.heartbeat_seconds,
        )

    spawn_ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_concurrent, mp_context=spawn_ctx) as executor:
        future_to_video = {}
        for v in to_process:
            future = _submit_video(executor, v)
            future_to_video[future] = v.name
        for future in as_completed(future_to_video):
            video_name = future_to_video[future]
            result = future.result()
            if result.get("error"):
                print(f"[ERROR] {video_name}: {result['error']}")
                if result.get("traceback"):
                    print(result["traceback"])


def _process_video_worker(
    video_path: str,
    out_dir: str,
    mode: str,
    device: str,
    batch_size: int,
    num_workers: int,
    min_duration: float,
    au_intensity_thresh: float,
    target_fps: Optional[float],
    weights_dir,
    temp_dir,
    verbose: bool,
    heartbeat_seconds: float,
):
    video = Path(video_path)
    out = Path(out_dir)
    weights = Path(weights_dir) if weights_dir else None
    temp = Path(temp_dir) if temp_dir else None
    try:
        if temp:
            temp.mkdir(parents=True, exist_ok=True)
        process_video(
            video,
            out,
            mode,
            device,
            batch_size,
            num_workers,
            min_duration,
            au_intensity_thresh,
            target_fps,
            weights,
            temp,
            verbose=verbose,
            heartbeat_seconds=heartbeat_seconds,
        )
        return {"video": video.name, "error": None}
    except Exception as exc:
        return {"video": video.name, "error": str(exc), "traceback": traceback.format_exc()}
    finally:
        if temp:
            shutil.rmtree(temp, ignore_errors=True)


if __name__ == "__main__":
    main()
