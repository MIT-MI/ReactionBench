#!/usr/bin/env python3
"""
Split selected reaction videos into one-minute segments while preserving audio.

This script reads a list of clean IDs, splits the corresponding face and stimuli
videos into consecutive one-minute clips, assigns fresh IDs, and updates the
dataset metadata. It appends the new entries to the main mapping CSV, writes a
lineage CSV that records the relationship between the new clips and their source
video, and replaces the original clean ID list with the new identifiers.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "common"))
import rvb_config as cfg  # noqa: E402  (loads .env, puts every stage dir on sys.path)

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import subprocess


SEGMENT_SECONDS = 60
SEEK_EPSILON = 1.0  # tolerate this much tail without creating an additional segment


class SplitError(RuntimeError):
    """Raised when a required asset is missing or ffmpeg fails."""


@dataclass
class MappingRow:
    new_id: int
    original_face_path: str
    original_stimuli_path: str
    face_path: Path
    stimuli_path: Path


@dataclass
class SegmentRecord:
    parent_id: int
    new_id: int
    segment_index: int
    start_seconds: float
    duration_seconds: float
    face_path: Path
    stimuli_path: Path


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mapping",
        type=Path,
        default=cfg.meta_file("mapping.csv"),
        help="Path to the primary mapping CSV file.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=cfg.DATA_ROOT,
        help="Root directory containing face_origin/ and stimuli/ folders.",
    )
    parser.add_argument(
        "--id-list",
        type=Path,
        default=cfg.work("clean_ids", "clean_id_list_part1.txt"),
        help="Path to the text file listing IDs that need to be split.",
    )
    parser.add_argument(
        "--lineage-output",
        type=Path,
        default=cfg.DATA_ROOT / "mapping_lineage_part1.csv",
        help="Where to store the new lineage CSV describing segment ancestry.",
    )
    parser.add_argument(
        "--segment-seconds",
        type=int,
        default=SEGMENT_SECONDS,
        help="Target length (in seconds) for each segment.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=SEEK_EPSILON,
        help="If the remaining duration is less than this, do not create an extra clip.",
    )
    return parser.parse_args(argv)


def read_mapping(mapping_path: Path) -> Tuple[Dict[int, MappingRow], int]:
    rows: Dict[int, MappingRow] = {}
    with mapping_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SplitError(f"Mapping file {mapping_path} is missing a header row.")
        max_id = -1
        for row in reader:
            try:
                new_id = int(row["new_id"])
            except (KeyError, ValueError) as exc:
                raise SplitError(f"Invalid new_id in mapping row: {row}") from exc
            face_path = Path(row["new_face_origin_path"])
            stimuli_path = Path(row["new_stimuli_path"])
            rows[new_id] = MappingRow(
                new_id=new_id,
                original_face_path=row["original_face_path"],
                original_stimuli_path=row["original_stimuli_path"],
                face_path=face_path,
                stimuli_path=stimuli_path,
            )
            if new_id > max_id:
                max_id = new_id
    if max_id < 0:
        raise SplitError("Could not determine the maximum ID from the mapping file.")
    return rows, max_id


def read_id_list(id_list_path: Path) -> List[int]:
    ids: List[int] = []
    with id_list_path.open("r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                ids.append(int(line))
            except ValueError as exc:
                raise SplitError(f"Invalid ID '{line}' in {id_list_path}") from exc
    if not ids:
        raise SplitError(f"No IDs found in {id_list_path}")
    return ids


def probe_duration(video_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(
        cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        return float(result.stdout.strip())
    except ValueError as exc:
        raise SplitError(f"Unable to parse duration for {video_path}") from exc


def compute_segment_plan(duration: float, segment_seconds: int, tolerance: float) -> List[Tuple[float, float]]:
    """Return (start, duration) tuples for each segment."""
    if duration <= 0:
        raise SplitError(f"Encountered non-positive duration: {duration}")
    segment_count = int(duration // segment_seconds)
    remainder = duration - (segment_count * segment_seconds)
    if segment_count == 0 or remainder > tolerance:
        segment_count += 1
    plan: List[Tuple[float, float]] = []
    for index in range(segment_count):
        start = index * segment_seconds
        remaining = max(duration - start, 0.0)
        clip_duration = min(segment_seconds, remaining)
        if clip_duration <= 0:
            continue
        plan.append((start, clip_duration))
    return plan


def run_ffmpeg_split(source: Path, destination: Path, start: float, duration: float) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(source),
        "-t",
        f"{duration:.3f}",
        "-c",
        "copy",
        "-avoid_negative_ts",
        "1",
        str(destination),
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise SplitError(
            f"ffmpeg failed for {source} -> {destination} (start={start}, duration={duration}):\n{result.stderr}"
        )


def format_id(value: int) -> str:
    return f"{value:05d}"


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)

    mapping_rows, max_existing_id = read_mapping(args.mapping)
    clean_ids = read_id_list(args.id_list)

    face_dir = args.data_root / "face_origin"
    stimuli_dir = args.data_root / "stimuli"

    next_id = max_existing_id + 1
    new_mapping_rows: List[List[str]] = []
    lineage_rows: List[SegmentRecord] = []
    new_id_list: List[str] = []

    print(
        f"Loaded {len(mapping_rows)} mapping rows; max existing ID is {max_existing_id}.",
        file=sys.stderr,
    )
    print(f"Processing {len(clean_ids)} source IDs.", file=sys.stderr)

    for index, parent_id in enumerate(clean_ids, start=1):
        if parent_id not in mapping_rows:
            raise SplitError(f"Parent ID {parent_id} not found in mapping file.")
        mapping_row = mapping_rows[parent_id]
        face_path = mapping_row.face_path
        stimuli_path = mapping_row.stimuli_path
        if not face_path.exists():
            raise SplitError(f"Missing face video: {face_path}")
        if not stimuli_path.exists():
            raise SplitError(f"Missing stimuli video: {stimuli_path}")

        duration_stimuli = probe_duration(stimuli_path)
        duration_face = probe_duration(face_path)
        duration_gap = abs(duration_face - duration_stimuli)
        if duration_gap > 1.5:
            print(
                f"Warning: duration mismatch for ID {parent_id}: "
                f"face={duration_face:.3f}s, stimuli={duration_stimuli:.3f}s",
                file=sys.stderr,
            )
        segment_plan = compute_segment_plan(
            min(duration_stimuli, duration_face), args.segment_seconds, args.tolerance
        )

        for segment_index, (start, duration) in enumerate(segment_plan):
            new_id = next_id
            next_id += 1
            new_filename = f"{format_id(new_id)}.mp4"
            target_face = face_dir / new_filename
            target_stimuli = stimuli_dir / new_filename

            run_ffmpeg_split(face_path, target_face, start, duration)
            run_ffmpeg_split(stimuli_path, target_stimuli, start, duration)

            new_mapping_rows.append(
                [
                    str(new_id),
                    mapping_row.original_face_path,
                    mapping_row.original_stimuli_path,
                    str(target_face),
                    str(target_stimuli),
                ]
            )
            lineage_rows.append(
                SegmentRecord(
                    parent_id=parent_id,
                    new_id=new_id,
                    segment_index=segment_index,
                    start_seconds=round(start, 3),
                    duration_seconds=round(duration, 3),
                    face_path=target_face,
                    stimuli_path=target_stimuli,
                )
            )
            new_id_list.append(str(new_id))

        if index % 25 == 0 or index == len(clean_ids):
            print(
                f"[{index}/{len(clean_ids)}] Processed parent ID {parent_id}; "
                f"generated {len(segment_plan)} segments.",
                file=sys.stderr,
            )

    # Append to mapping.csv
    with args.mapping.open("a", newline="") as handle:
        writer = csv.writer(handle)
        for row in new_mapping_rows:
            writer.writerow(row)

    # Write lineage CSV
    with args.lineage_output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "parent_id",
                "new_id",
                "segment_index",
                "start_seconds",
                "duration_seconds",
                "new_face_origin_path",
                "new_stimuli_path",
            ]
        )
        for record in lineage_rows:
            writer.writerow(
                [
                    record.parent_id,
                    record.new_id,
                    record.segment_index,
                    f"{record.start_seconds:.3f}",
                    f"{record.duration_seconds:.3f}",
                    str(record.face_path),
                    str(record.stimuli_path),
                ]
            )

    # Replace ID list with newly generated IDs
    temp_id_list_path = args.id_list.with_suffix(".tmp")
    with temp_id_list_path.open("w") as handle:
        handle.write("\n".join(new_id_list))
        handle.write("\n")
    temp_id_list_path.replace(args.id_list)

    print(
        f"Created {len(new_id_list)} new clips across {len(clean_ids)} source videos. "
        f"New ID range {new_mapping_rows[0][0]} - {new_mapping_rows[-1][0]}.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except SplitError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        print(f"Subprocess failed: {exc}", file=sys.stderr)
        sys.exit(1)
