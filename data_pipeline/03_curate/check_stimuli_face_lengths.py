#!/usr/bin/env python3
"""Compare paired stimuli and face_origin video durations and export results.

This script walks two directories that contain paired video files, measures the
duration of each video with ``ffprobe``, and writes a CSV report listing the
duration for every id present in both directories.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "common"))
import rvb_config as cfg  # noqa: E402  (loads .env, puts every stage dir on sys.path)

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple


def parse_args() -> argparse.Namespace:
    default_base = cfg.DATA_ROOT
    parser = argparse.ArgumentParser(
        description="Check that paired stimuli and face_origin videos share the same duration."
    )
    parser.add_argument(
        "--stimuli-dir",
        type=Path,
        default=default_base / "stimuli",
        help="Directory containing the stimuli videos (default: %(default)s)",
    )
    parser.add_argument(
        "--face-origin-dir",
        type=Path,
        default=default_base / "face_origin",
        help="Directory containing the face_origin videos (default: %(default)s)",
    )
    parser.add_argument(
        "--extension",
        default="mp4",
        help="Video file extension to consider (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("stimuli_face_length_report.csv"),
        help="Path to the CSV file to write (default: %(default)s)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="Tolerance (in seconds) when flagging mismatched durations (default: %(default)s)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, (os.cpu_count() or 2)),
        help="Number of parallel ffprobe workers to use (default: %(default)s)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of ids to process (useful for spot checks)",
    )
    return parser.parse_args()


def ffprobe_duration(path: Path) -> Optional[float]:
    """Return the duration of the given video file in seconds."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        str(path),
    ]
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe is not available on PATH") from exc
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(f"[WARN] ffprobe failed for {path}: {exc.stderr.strip()}\n")
        return None

    try:
        payload = json.loads(result.stdout)
        duration_str = payload["format"]["duration"]
        return float(duration_str)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"[WARN] Could not parse duration for {path}: {exc}\n")
        return None


def collect_durations(
    files: Dict[str, Path], workers: int
) -> Dict[str, Optional[float]]:
    """Measure durations for all provided files."""
    if not files:
        return {}

    durations: Dict[str, Optional[float]] = {}

    if workers <= 1:
        for file_id, path in files.items():
            durations[file_id] = ffprobe_duration(path)
        return durations

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_id = {
            executor.submit(ffprobe_duration, path): file_id
            for file_id, path in files.items()
        }
        for future in as_completed(future_to_id):
            file_id = future_to_id[future]
            try:
                durations[file_id] = future.result()
            except RuntimeError:
                # ffprobe missing; propagate to caller so the run clearly fails.
                for pending in future_to_id:
                    pending.cancel()
                raise
            except Exception as exc:  # noqa: BLE001 - capture unexpected failures
                sys.stderr.write(f"[WARN] Failed to measure {file_id}: {exc}\n")
                durations[file_id] = None

    return durations


def discover_files(directory: Path, extension: str) -> Dict[str, Path]:
    """Return a mapping of id -> file path for the given directory."""
    pattern = f"*.{extension.lstrip('.')}"
    return {file_path.stem: file_path for file_path in directory.glob(pattern)}


def format_duration(duration: Optional[float]) -> str:
    return "" if duration is None else f"{duration:.6f}"


def summarize_mismatches(
    ids: Iterable[str],
    stimuli_durations: Dict[str, Optional[float]],
    face_durations: Dict[str, Optional[float]],
    tolerance: float,
) -> Tuple[int, int]:
    missing = 0
    mismatched = 0
    for file_id in ids:
        stim = stimuli_durations.get(file_id)
        face = face_durations.get(file_id)
        if stim is None or face is None:
            missing += 1
            continue
        if abs(stim - face) > tolerance:
            mismatched += 1
    return missing, mismatched


def main() -> None:
    args = parse_args()

    stimuli_dir = args.stimuli_dir.expanduser().resolve()
    face_dir = args.face_origin_dir.expanduser().resolve()

    if not stimuli_dir.exists():
        sys.exit(f"Stimuli directory not found: {stimuli_dir}")
    if not face_dir.exists():
        sys.exit(f"face_origin directory not found: {face_dir}")

    stimuli_files = discover_files(stimuli_dir, args.extension)
    face_files = discover_files(face_dir, args.extension)

    common_ids = sorted(stimuli_files.keys() & face_files.keys())

    if not common_ids:
        sys.exit("No overlapping ids between stimuli and face_origin directories.")

    if args.limit is not None:
        common_ids = common_ids[: args.limit]

    stimuli_subset = {file_id: stimuli_files[file_id] for file_id in common_ids}
    face_subset = {file_id: face_files[file_id] for file_id in common_ids}

    stimuli_durations = collect_durations(stimuli_subset, args.workers)
    face_durations = collect_durations(face_subset, args.workers)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "stimuli_length", "face_origin_length"])
        for file_id in common_ids:
            writer.writerow(
                [
                    file_id,
                    format_duration(stimuli_durations.get(file_id)),
                    format_duration(face_durations.get(file_id)),
                ]
            )

    missing, mismatched = summarize_mismatches(
        common_ids, stimuli_durations, face_durations, args.tolerance
    )
    print(f"Wrote {len(common_ids)} rows to {args.output}")
    if missing:
        print(f"Warning: {missing} ids could not be measured (see stderr warnings).")
    if mismatched:
        print(
            f"Found {mismatched} ids where durations differ by more than {args.tolerance} seconds."
        )
    else:
        print("All measured durations are within tolerance.")


if __name__ == "__main__":
    main()
