#!/usr/bin/env python3
"""
Batch LibreFace AU intensity extraction for ReactionBench dataset construction.

Outputs:
  - face_intensity_avg/{id}_au_intensity_avg.csv (segment-level, aligned with description_clean)
  - face_intensity_series/{id}_au_intensity_series.csv
  - logs/libreface_intensity/libreface_au_intensity_<timestamp>.json
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "common"))
import rvb_config as cfg  # noqa: E402  (loads .env, puts every stage dir on sys.path)

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

import pandas as pd

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore[assignment]


DEFAULT_DATA_ROOT = cfg.DATA_ROOT
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpg", ".mpeg", ".m4v"}


def _enable_line_buffering() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except Exception:
            pass


def _suppress_progress_bars() -> None:
    if os.environ.get("HALLO3_DISABLE_PROGRESS", "").lower() in {"0", "false", "no"}:
        return

    should_disable = os.environ.get("HALLO3_DISABLE_PROGRESS")
    if not should_disable:
        should_disable = "1" if not sys.stderr.isatty() else ""
    if not should_disable:
        return

    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    try:
        import tqdm  # type: ignore
    except Exception:
        return

    original_tqdm = tqdm.tqdm
    original_trange = getattr(tqdm, "trange", None)

    class _SilentTqdm(original_tqdm):  # type: ignore[misc]
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("disable", True)
            super().__init__(*args, **kwargs)

    def _silent_trange(*args, **kwargs):
        kwargs.setdefault("disable", True)
        if callable(original_trange):
            return original_trange(*args, **kwargs)
        return _SilentTqdm(range(*args, **kwargs))

    tqdm.tqdm = _SilentTqdm  # type: ignore[assignment]
    tqdm.trange = _silent_trange  # type: ignore[assignment]


_enable_line_buffering()
_suppress_progress_bars()

try:
    import libreface
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Failed to import libreface. Install it first, e.g. "
        "`pip install libreface` or `pip install -e <path/to/LibreFace>`."
    ) from exc


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def auto_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


def load_id_list(path: Path) -> List[str]:
    stems: List[str] = []
    seen: Set[str] = set()
    for line in path.read_text().splitlines():
        token = line.strip()
        if not token or token.startswith("#"):
            continue
        stem = Path(token.split()[0]).stem
        if stem and stem not in seen:
            stems.append(stem)
            seen.add(stem)
    return stems


def atomic_write_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        suffix=".tmp",
        prefix=f"{path.name}.",
        dir=path.parent,
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def atomic_write_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".tmp",
        prefix=f"{path.name}.",
        dir=path.parent,
        delete=False,
    ) as tmp:
        json.dump(data, tmp, ensure_ascii=True, indent=2)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    try:
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def sanitize_time_series(df: pd.DataFrame) -> pd.DataFrame:
    if "frame_time_in_ms" not in df.columns:
        raise ValueError("LibreFace output is missing required column 'frame_time_in_ms'.")

    out = df.copy()
    raw = pd.to_numeric(out["frame_time_in_ms"], errors="coerce")
    if raw.isna().all():
        raise ValueError("LibreFace output has no valid timestamps in 'frame_time_in_ms'.")

    span = float(raw.max() - raw.min()) if len(raw) else 0.0
    time_s = raw / 1000.0 if span > 100.0 else raw

    values = time_s.to_numpy(dtype=float, copy=True)
    if values.size >= 2:
        offset = 0.0
        for idx in range(1, values.size):
            if values[idx] < values[idx - 1]:
                offset += values[idx - 1]
            values[idx] += offset

    out["time_s"] = values
    return out


def sample_time_series(df: pd.DataFrame, interval_s: float) -> pd.DataFrame:
    if interval_s <= 0 or len(df) <= 1:
        return df.reset_index(drop=True)

    times = pd.to_numeric(df["time_s"], errors="coerce").to_numpy(dtype=float)
    keep_indices = [0]
    last_time = times[0]
    eps = max(1e-9, interval_s * 1e-6)
    for idx in range(1, len(times)):
        if times[idx] - last_time >= interval_s - eps:
            keep_indices.append(idx)
            last_time = times[idx]

    last_idx = len(df) - 1
    if keep_indices[-1] != last_idx:
        keep_indices.append(last_idx)
    return df.iloc[keep_indices].reset_index(drop=True)


def detect_intensity_columns(df: pd.DataFrame) -> List[str]:
    return sorted(
        col for col in df.columns if col.startswith("au_") and col.endswith("_intensity")
    )


def load_boundary_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Boundary file not found: {path}")
    df = pd.read_csv(path)
    required = {"start_time_s", "end_time_s", "duration_s"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Boundary file missing required columns {sorted(missing)}: {path}")
    if df.empty:
        raise ValueError(f"Boundary file is empty: {path}")
    out = df.copy().reset_index(drop=True)
    out["start_time_s"] = pd.to_numeric(out["start_time_s"], errors="raise")
    out["end_time_s"] = pd.to_numeric(out["end_time_s"], errors="raise")
    out["duration_s"] = pd.to_numeric(out["duration_s"], errors="raise")
    return out


def robust_get_df(
    video_path: Path,
    *,
    device: str,
    batch_size: int,
    num_workers: int,
    temp_dir: Optional[Path],
    weights_dir: Optional[Path],
) -> pd.DataFrame:
    kwargs: Dict[str, Any] = {
        "device": device,
        "batch_size": batch_size,
        "num_workers": num_workers,
    }
    if temp_dir is not None:
        kwargs["temp_dir"] = str(temp_dir)
    if weights_dir is not None:
        kwargs["weights_download_dir"] = str(weights_dir)

    df = libreface.get_facial_attributes(str(video_path), **kwargs)
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    if df.empty:
        raise RuntimeError("LibreFace returned an empty DataFrame.")
    return df


def build_segment_avg_dataframe(
    stem: str,
    video_path: Path,
    df: pd.DataFrame,
    intensity_cols: Sequence[str],
    boundaries: pd.DataFrame,
    boundary_path: Path,
    include_intensity_summary: bool,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    total_rows = len(boundaries)
    for idx, boundary in boundaries.iterrows():
        start_time = float(boundary["start_time_s"])
        end_time = float(boundary["end_time_s"])
        if idx == total_rows - 1:
            mask = (df["time_s"] >= start_time) & (df["time_s"] <= end_time)
        else:
            mask = (df["time_s"] >= start_time) & (df["time_s"] < end_time)
        segment_df = df.loc[mask]
        row: Dict[str, Any] = {
            "video": boundary.get("video", f"{stem}{video_path.suffix}"),
            "start_time_s": start_time,
            "end_time_s": end_time,
            "duration_s": float(boundary["duration_s"]),
            "au_signature": "",
            "emotion_description": "",
            "reaction_description": "",
            "video_stem": stem,
            "video_path": str(video_path),
            "boundary_path": str(boundary_path),
            "segment_index": int(idx),
            "frame_count": int(len(segment_df)),
        }
        for passthrough_col in ("video", "au_signature", "emotion_description", "reaction_description"):
            if passthrough_col in boundaries.columns:
                value = boundary.get(passthrough_col)
                row[passthrough_col] = "" if pd.isna(value) else value

        if include_intensity_summary:
            intensity_summary: Dict[str, float] = {}
            for col in intensity_cols:
                numeric = (
                    pd.to_numeric(segment_df[col], errors="coerce")
                    if not segment_df.empty
                    else pd.Series(dtype=float)
                )
                if numeric.empty or numeric.isna().all():
                    continue
                intensity_summary[col] = round(float(numeric.mean()), 6)
            row["au_intensity_summary"] = json.dumps(
                intensity_summary,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
        rows.append(row)
    ordered_columns = [
        "video",
        "start_time_s",
        "end_time_s",
        "duration_s",
        "au_signature",
        "emotion_description",
        "reaction_description",
        "video_stem",
        "video_path",
        "boundary_path",
        "segment_index",
        "frame_count",
    ]
    if include_intensity_summary:
        ordered_columns.append("au_intensity_summary")
    return pd.DataFrame(rows)[ordered_columns]


def build_series_dataframe(
    stem: str,
    video_path: Path,
    df: pd.DataFrame,
    intensity_cols: Sequence[str],
    sampling_interval_s: float,
) -> pd.DataFrame:
    sampled = sample_time_series(df, sampling_interval_s).copy()
    sampled.insert(0, "video_stem", stem)
    sampled.insert(1, "video_path", str(video_path))
    sampled.insert(2, "sample_index", range(len(sampled)))
    series_cols = ["video_stem", "video_path", "sample_index", "time_s", *intensity_cols]
    for col in intensity_cols:
        sampled[col] = pd.to_numeric(sampled[col], errors="coerce")
    return sampled[series_cols]


def output_paths(stem: str, avg_dir: Path, series_dir: Path) -> tuple[Path, Path]:
    avg_path = avg_dir / f"{stem}_au_intensity_avg.csv"
    series_path = series_dir / f"{stem}_au_intensity_series.csv"
    return avg_path, series_path


def is_up_to_date(sources: Sequence[Path], outputs: Sequence[Path]) -> bool:
    try:
        source_mtime = max(source.stat().st_mtime for source in sources)
    except OSError:
        return False
    for path in outputs:
        if not path.exists():
            return False
        try:
            if path.stat().st_size <= 0 or path.stat().st_mtime < source_mtime:
                return False
        except OSError:
            return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frame-level and description-segment-level LibreFace AU intensity outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Canonical dataset root directory.",
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=None,
        help="Input directory containing face videos. Defaults to <data_root>/face_origin.",
    )
    parser.add_argument(
        "--avg_dir",
        type=Path,
        default=None,
        help="Output directory for description-aligned segment CSVs. Defaults to <data_root>/face_intensity_avg.",
    )
    parser.add_argument(
        "--series_dir",
        type=Path,
        default=None,
        help="Output directory for frame or interval CSVs. Defaults to <data_root>/face_intensity_series.",
    )
    parser.add_argument(
        "--boundary_dir",
        type=Path,
        default=None,
        help="Directory containing description boundary CSVs. Defaults to <data_root>/description_clean.",
    )
    parser.add_argument(
        "--log_dir",
        type=Path,
        default=None,
        help="Directory for structured JSON run logs. Defaults to <data_root>/logs/libreface_intensity.",
    )
    parser.add_argument(
        "--id-list",
        type=Path,
        default=None,
        help="Optional text file containing one clip ID or filename per line.",
    )
    parser.add_argument(
        "--sampling-interval",
        type=float,
        default=0.0,
        help="Series sampling interval in seconds. Use 0 to keep every frame.",
    )
    parser.add_argument(
        "--include-intensity-summary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Include a compact JSON AU intensity summary column in the segment-level CSV. "
            "Disable with --no-include-intensity-summary to emit only description-aligned rows."
        ),
    )
    parser.add_argument("--device", type=str, default=auto_device(), help='Compute device, e.g. "cpu" or "cuda:0".')
    parser.add_argument("--batch-size", type=int, default=256, help="LibreFace video batch size.")
    parser.add_argument("--num-workers", type=int, default=2, help="LibreFace dataloader workers.")
    parser.add_argument("--weights-dir", type=Path, default=None, help="Optional directory for LibreFace weights.")
    parser.add_argument(
        "--temp-root",
        type=Path,
        default=None,
        help="Optional temp root for LibreFace frame extraction.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument("--dry-run", action="store_true", help="Discover work only; do not run inference.")
    parser.add_argument(
        "--tb-log-dir",
        type=Path,
        default=None,
        help="Optional TensorBoard log directory. Defaults to <data_root>/logs/libreface_intensity_tb when enabled.",
    )
    parser.add_argument(
        "--tb-run-name",
        type=str,
        default=None,
        help="Optional TensorBoard run subdirectory name. Defaults to a timestamped name.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of videos to process. Useful for small test runs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    data_root = args.data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise SystemExit(f"[FATAL] data_root not found: {data_root}")

    input_dir = (args.input_dir or (data_root / "face_origin")).expanduser().resolve()
    avg_dir = (args.avg_dir or (data_root / "face_intensity_avg")).expanduser().resolve()
    series_dir = (args.series_dir or (data_root / "face_intensity_series")).expanduser().resolve()
    boundary_dir = (args.boundary_dir or (data_root / "description_clean")).expanduser().resolve()
    log_dir = (args.log_dir or (data_root / "logs" / "libreface_intensity")).expanduser().resolve()
    tb_base_dir = (
        (args.tb_log_dir or (data_root / "logs" / "libreface_intensity_tb")).expanduser().resolve()
    )
    weights_dir = args.weights_dir.expanduser().resolve() if args.weights_dir else None
    temp_root = args.temp_root.expanduser().resolve() if args.temp_root else None

    if args.sampling_interval < 0:
        raise SystemExit("[FATAL] --sampling-interval must be >= 0.")
    if not input_dir.is_dir():
        raise SystemExit(f"[FATAL] input_dir not found: {input_dir}")
    if not boundary_dir.is_dir():
        raise SystemExit(f"[FATAL] boundary_dir not found: {boundary_dir}")

    log_dir.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        avg_dir.mkdir(parents=True, exist_ok=True)
        series_dir.mkdir(parents=True, exist_ok=True)
        if SummaryWriter is not None:
            tb_base_dir.mkdir(parents=True, exist_ok=True)
    if temp_root is not None and not args.dry_run:
        temp_root.mkdir(parents=True, exist_ok=True)

    all_videos = sorted(
        path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_EXTS
    )
    print(f"[INFO] Found {len(all_videos)} videos in {input_dir}")

    requested_ids: Optional[List[str]] = None
    allowed_ids: Optional[Set[str]] = None
    missing_requested_ids: List[str] = []
    if args.id_list is not None:
        id_path = args.id_list.expanduser().resolve()
        if not id_path.exists():
            raise SystemExit(f"[FATAL] ID list not found: {id_path}")
        requested_ids = load_id_list(id_path)
        allowed_ids = set(requested_ids)
        found_ids = {path.stem for path in all_videos}
        missing_requested_ids = [clip_id for clip_id in requested_ids if clip_id not in found_ids]
        print(
            f"[INFO] ID filter requested {len(requested_ids)} clips; "
            f"missing_from_input={len(missing_requested_ids)}"
        )

    work_items: List[Path] = []
    pre_entries: List[Dict[str, Any]] = []
    skipped_existing = 0
    skipped_filtered = 0
    for video_path in all_videos:
        clip_id = video_path.stem
        if allowed_ids is not None and clip_id not in allowed_ids:
            skipped_filtered += 1
            pre_entries.append(
                {
                    "video_stem": clip_id,
                    "status": "skipped_filtered",
                    "video_path": str(video_path),
                }
            )
            continue
        avg_path, series_path = output_paths(clip_id, avg_dir, series_dir)
        boundary_path = boundary_dir / f"{clip_id}.csv"
        if not args.overwrite and is_up_to_date((video_path, boundary_path), (avg_path, series_path)):
            skipped_existing += 1
            pre_entries.append(
                {
                    "video_stem": clip_id,
                    "status": "skipped_existing",
                    "video_path": str(video_path),
                    "boundary_path": str(boundary_path),
                    "avg_path": str(avg_path),
                    "series_path": str(series_path),
                }
            )
            continue
        work_items.append(video_path)

    if args.limit is not None:
        work_items = work_items[: max(0, args.limit)]

    print(
        f"[INFO] Work list: {len(work_items)} to process | "
        f"skipped_existing={skipped_existing} skipped_filtered={skipped_filtered}"
    )

    writer = None
    tb_run_dir: Optional[Path] = None
    if not args.dry_run and SummaryWriter is not None:
        tb_run_name = args.tb_run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        tb_run_dir = tb_base_dir / tb_run_name
        tb_run_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_run_dir))
        writer.add_text("config/data_root", str(data_root), 0)
        writer.add_text("config/input_dir", str(input_dir), 0)
        writer.add_text("config/device", str(args.device), 0)
        writer.add_scalar("run/work_items_total", len(work_items), 0)
        writer.add_scalar("run/skipped_existing_initial", skipped_existing, 0)
        writer.add_scalar("run/skipped_filtered_initial", skipped_filtered, 0)
        writer.add_scalar("run/missing_requested_initial", len(missing_requested_ids), 0)
        writer.flush()

    run_log: Dict[str, Any] = {
        "started_at": now_iso(),
        "data_root": str(data_root),
        "input_dir": str(input_dir),
        "avg_dir": str(avg_dir),
        "series_dir": str(series_dir),
        "boundary_dir": str(boundary_dir),
        "log_dir": str(log_dir),
        "device": args.device,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "sampling_interval_s": args.sampling_interval,
        "include_intensity_summary": bool(args.include_intensity_summary),
        "limit": args.limit,
        "tensorboard_log_dir": None if tb_run_dir is None else str(tb_run_dir),
        "overwrite": bool(args.overwrite),
        "dry_run": bool(args.dry_run),
        "requested_id_count": 0 if requested_ids is None else len(requested_ids),
        "missing_requested_ids": missing_requested_ids,
        "entries": pre_entries,
    }

    for clip_id in missing_requested_ids:
        run_log["entries"].append(
            {
                "video_stem": clip_id,
                "status": "missing_input",
                "reason": "requested_id_not_found_under_input_dir",
            }
        )

    if args.dry_run:
        for video_path in work_items[:20]:
            print(f"[DRY] {video_path.stem} -> {video_path}")
            run_log["entries"].append(
                {
                    "video_stem": video_path.stem,
                    "status": "dry_run_candidate",
                    "video_path": str(video_path),
                }
            )
        if len(work_items) > 20:
            print(f"[DRY] ... and {len(work_items) - 20} more")
        run_log["finished_at"] = now_iso()
        run_log["summary"] = {
            "processed": 0,
            "skipped_existing": skipped_existing,
            "skipped_filtered": skipped_filtered,
            "missing_input": len(missing_requested_ids),
            "errors": 0,
            "dry_run_candidates": len(work_items),
        }
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"libreface_au_intensity_{timestamp}.json"
        atomic_write_json(run_log, log_path)
        print(f"[INFO] Wrote run log to {log_path}")
        return 0

    processed = 0
    errors = 0
    total_elapsed_s = 0.0
    total_video_duration_s = 0.0
    for video_path in work_items:
        clip_id = video_path.stem
        avg_path, series_path = output_paths(clip_id, avg_dir, series_dir)
        boundary_path = boundary_dir / f"{clip_id}.csv"
        temp_dir = None
        started_at = now_iso()
        t0 = time.perf_counter()
        try:
            if not boundary_path.exists():
                raise FileNotFoundError(f"Boundary file not found: {boundary_path}")
            if temp_root is not None:
                temp_dir = temp_root / f"{clip_id}.tmp"
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                temp_dir.mkdir(parents=True, exist_ok=True)

            print(f"[INFO] Processing {clip_id}")
            df = robust_get_df(
                video_path,
                device=args.device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                temp_dir=temp_dir,
                weights_dir=weights_dir,
            )
            df = sanitize_time_series(df)
            intensity_cols = detect_intensity_columns(df)
            if not intensity_cols:
                raise RuntimeError("LibreFace output contains no au_*_intensity columns.")
            boundaries = load_boundary_dataframe(boundary_path)

            avg_df = build_segment_avg_dataframe(
                clip_id,
                video_path,
                df,
                intensity_cols,
                boundaries,
                boundary_path,
                args.include_intensity_summary,
            )
            series_df = build_series_dataframe(
                clip_id,
                video_path,
                df,
                intensity_cols,
                args.sampling_interval,
            )

            atomic_write_dataframe(avg_df, avg_path)
            atomic_write_dataframe(series_df, series_path)
            processed += 1
            elapsed_s = time.perf_counter() - t0
            video_duration_s = float(df["time_s"].iloc[-1] - df["time_s"].iloc[0]) if len(df) > 1 else 0.0
            processing_fps = (float(len(df)) / elapsed_s) if elapsed_s > 0 else None
            wallclock_per_input_minute = (
                elapsed_s / (video_duration_s / 60.0) if video_duration_s > 0 else None
            )
            total_elapsed_s += elapsed_s
            total_video_duration_s += video_duration_s
            run_log["entries"].append(
                {
                    "video_stem": clip_id,
                    "status": "processed",
                    "started_at": started_at,
                    "finished_at": now_iso(),
                    "elapsed_s": elapsed_s,
                    "video_path": str(video_path),
                    "boundary_path": str(boundary_path),
                    "avg_path": str(avg_path),
                    "series_path": str(series_path),
                    "frame_count": int(len(df)),
                    "boundary_count": int(len(boundaries)),
                    "sample_count": int(len(series_df)),
                    "video_duration_s": video_duration_s,
                    "processing_fps": processing_fps,
                    "wallclock_per_input_minute": wallclock_per_input_minute,
                    "intensity_columns": intensity_cols,
                }
            )
            if writer is not None:
                writer.add_scalar("video/elapsed_s", elapsed_s, processed)
                writer.add_scalar("video/video_duration_s", video_duration_s, processed)
                writer.add_scalar("video/frame_count", int(len(df)), processed)
                writer.add_scalar("video/sample_count", int(len(series_df)), processed)
                if processing_fps is not None:
                    writer.add_scalar("video/processing_fps", processing_fps, processed)
                if wallclock_per_input_minute is not None:
                    writer.add_scalar(
                        "video/wallclock_per_input_minute",
                        wallclock_per_input_minute,
                        processed,
                    )
                writer.add_scalar("run/processed_count", processed, processed)
                writer.add_scalar("run/error_count", errors, processed)
                if total_elapsed_s > 0:
                    writer.add_scalar(
                        "run/avg_elapsed_s",
                        total_elapsed_s / processed,
                        processed,
                    )
                if total_video_duration_s > 0:
                    writer.add_scalar(
                        "run/avg_wallclock_per_input_minute",
                        total_elapsed_s / (total_video_duration_s / 60.0),
                        processed,
                    )
                writer.flush()
        except Exception as exc:
            errors += 1
            elapsed_s = time.perf_counter() - t0
            print(f"[ERROR] {clip_id}: {exc}", file=sys.stderr)
            run_log["entries"].append(
                {
                    "video_stem": clip_id,
                    "status": "error",
                    "started_at": started_at,
                    "finished_at": now_iso(),
                    "elapsed_s": elapsed_s,
                    "video_path": str(video_path),
                    "error": str(exc),
                }
            )
            if writer is not None:
                step = processed + errors
                writer.add_scalar("run/processed_count", processed, step)
                writer.add_scalar("run/error_count", errors, step)
                writer.add_scalar("video/error_elapsed_s", elapsed_s, step)
                writer.flush()
        finally:
            if temp_dir is not None and temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log["finished_at"] = now_iso()
    run_log["summary"] = {
        "processed": processed,
        "skipped_existing": skipped_existing,
        "skipped_filtered": skipped_filtered,
        "missing_input": len(missing_requested_ids),
        "errors": errors,
        "total_elapsed_s": total_elapsed_s,
        "total_video_duration_s": total_video_duration_s,
        "avg_elapsed_s": (total_elapsed_s / processed) if processed > 0 else None,
        "avg_wallclock_per_input_minute": (
            total_elapsed_s / (total_video_duration_s / 60.0) if total_video_duration_s > 0 else None
        ),
    }

    log_path = log_dir / f"libreface_au_intensity_{timestamp}.json"
    atomic_write_json(run_log, log_path)
    print(f"[INFO] Wrote run log to {log_path}")
    if writer is not None:
        writer.add_text("run/json_log_path", str(log_path), max(processed, errors))
        writer.close()
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
