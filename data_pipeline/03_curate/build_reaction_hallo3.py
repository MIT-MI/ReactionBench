#!/usr/bin/env python3
"""Move high-quality face_origin and stimuli videos into reaction_benchmark_hallo3.
- Outputs only two folders: face_origin/ and stimuli/
- No commenter audio, no muxing, no face/ folder.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "common"))
import rvb_config as cfg  # noqa: E402  (loads .env, puts every stage dir on sys.path)

import argparse
import csv
import shutil
import sys
from pathlib import Path

BOOL_TRUE = {"true", "1", "yes", "y"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quality-csv",
        type=Path,
        default=cfg.work("qc", "per_clip_quality.csv"),
        help="Path to per_clip_quality.csv (must contain face_path and stimuli path columns)",
    )
    parser.add_argument(
        "--dest-root",
        type=Path,
        default=cfg.DATA_ROOT,
        help="Destination root for the curated dataset",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting index for naming (zero padded to 5 digits)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of entries to process",
    )
    parser.add_argument(
        "--allow-nonempty",
        action="store_true",
        help="Allow processing when destination directories already contain files",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume into an existing dataset: read existing mapping.csv to skip already-added sources "
            "and continue numbering after the highest existing new_id. Implies --allow-nonempty."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Perform a dry run: simulate file moves without modifying files. "
            "Only write a new dry_run_mapping.csv preview to the destination root."
        ),
    )
    parser.add_argument(
        "--face-col",
        type=str,
        default=None,
        help="Explicit column name to use for face path (overrides auto-detect)",
    )
    parser.add_argument(
        "--stimuli-col",
        type=str,
        default=None,
        help="Explicit column name to use for stimuli path (overrides auto-detect)",
    )
    parser.add_argument(
        "--stimuli-replace",
        action="append",
        default=None,
        metavar="SRC=DST",
        help=(
            "Optional. When no stimuli column is found, derive stimuli path from face path by applying one or more "
            "string replacements in order. Example: --stimuli-replace '/face_origin/=/stimuli/' --stimuli-replace '.mp4=_stim.mp4'"
        ),
    )
    return parser.parse_args()


def normalize_bool(value: str) -> bool:
    return (value or "").strip().lower() in BOOL_TRUE


def list_video_files(directory: Path | None) -> list[Path]:
    """Return a sorted list of video files within a directory (recursive)."""
    if directory is None or not directory.exists():
        return []
    if directory.is_file():
        return [directory] if directory.suffix.lower() in VIDEO_EXTENSIONS else []
    return sorted(
        p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


def _digits_only(text: str) -> str:
    return "".join(ch for ch in text if ch.isdigit())


def match_best_stimulus(
    face_path: Path,
    stim_dir: Path | None,
    stim_candidates: list[Path],
    default_stim: Path | None,
) -> Path | None:
    """Heuristically pick the best matching stimuli file for a given face file."""
    if default_stim and default_stim.is_file():
        return default_stim
    if stim_dir and stim_dir.is_dir():
        same_name = stim_dir / face_path.name
        if same_name.is_file():
            return same_name
        candidates = stim_candidates if stim_candidates else list_video_files(stim_dir)
        if not candidates:
            return None
        for candidate in candidates:
            if candidate.stem == face_path.stem:
                return candidate
        face_digits = _digits_only(face_path.stem)
        if face_digits:
            for candidate in candidates:
                if _digits_only(candidate.stem) == face_digits:
                    return candidate
        return candidates[0]
    return None


def derive_pairs_from_raw(raw_path: Path) -> list[tuple[Path, Path]]:
    """Derive (face_path, stimuli_path) pairs by enumerating face/stimuli directories."""
    parts = raw_path.parts
    if "raw" not in parts:
        return []
    raw_idx = parts.index("raw")
    if raw_idx < 0 or raw_idx >= len(parts) - 1:
        return []
    base_root = Path(*parts[:raw_idx])
    category = parts[raw_idx + 1]
    if not category:
        return []
    stem = raw_path.stem
    suffix = raw_path.suffix

    face_root = base_root / "face" / category
    stim_dir = base_root / "stimuli" / category / f"{stem}_stimuli"
    default_stim = stim_dir / f"{stem}{suffix}"
    stim_candidates = list_video_files(stim_dir)

    face_dirs = [
        face_root / f"{stem}_face",
        face_root / f"{stem}_masked_face",
    ]

    derived: list[tuple[Path, Path]] = []
    seen_faces: set[str] = set()
    for face_dir in face_dirs:
        for face_path in list_video_files(face_dir):
            face_key = str(face_path)
            if face_key in seen_faces:
                continue
            stim_path = match_best_stimulus(face_path, stim_dir, stim_candidates, default_stim)
            if stim_path and stim_path.is_file():
                derived.append((face_path, stim_path))
                seen_faces.add(face_key)

    if not derived:
        default_face = face_root / f"{stem}_face" / f"{stem}{suffix}"
        if default_face.is_file():
            stim_path = match_best_stimulus(default_face, stim_dir, stim_candidates, default_stim)
            if stim_path and stim_path.is_file():
                derived.append((default_face, stim_path))

    return derived


def load_valid_entries(csv_path: Path, args: argparse.Namespace) -> list[tuple[Path, Path]]:
    """Load (face_path, stimuli_path) pairs from a quality CSV with flexible column names.

    - Accepts multiple possible column names for face and stimuli.
    - Does not require special boolean flags; relies on file existence check.
    - Prints diagnostics if nothing is found to help debug column mismatches.
    """
    # If user provides explicit column names, prefer them
    face_force = (args.face_col or "").strip() if hasattr(args, "face_col") else ""
    stim_force = (args.stimuli_col or "").strip() if hasattr(args, "stimuli_col") else ""

    face_keys = [
        "face_path",
        "face_origin_path",
        "reactor_face_path",
        "face_local_path",
        "face_video_path",
        "face",
        "face_file",
        "primary_path",
        "alt_path",
    ]
    stim_keys = [
        "stimuli_path",
        "stimulus_path",
        "stimuli_video_path",
        "stimulus_video_path",
        "stimuli_local_path",
        "stimuli_clip_path",
        "stimuli",
        "stimulus",
    ]

    # Stats for diagnostics
    total_rows = 0
    face_missing = 0
    stim_missing = 0
    face_not_file = 0
    stim_not_file = 0

    dedup: dict[str, dict[str, str]] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        for row in reader:
            total_rows += 1
            # Pick raw path from primary_path (if exists==true), else alt_path (if low_exists==true), else fallback to face-like columns
            raw_val = ""
            primary = (row.get("primary_path") or "").strip()
            alt = (row.get("alt_path") or "").strip()
            exists_flag = (row.get("exists") or "").strip().lower() in BOOL_TRUE
            low_exists_flag = (row.get("low_exists") or "").strip().lower() in BOOL_TRUE
            if primary and exists_flag:
                raw_val = primary
            elif alt and low_exists_flag:
                raw_val = alt
            else:
                for k in face_keys:
                    if k in row and (row.get(k) or "").strip():
                        raw_val = (row.get(k) or "").strip()
                        break

            # We derive processed paths from the RAW path by convention:
            #   <base>/raw/<category>/<stem>.ext
            # → <base>/face/<category>/<stem>_face/<stem>.ext
            # → <base>/stimuli/<category>/<stem>_stimuli/<stem>.ext
            face_val = ""
            stim_val_from_raw = ""
            if raw_val:
                p = Path(raw_val)
                stem = p.stem
                suf = p.suffix
                parts = p.parts
                # find the 'raw' segment to split base and subpath
                if "raw" in parts:
                    raw_idx = parts.index("raw")
                    base_root = Path(*parts[:raw_idx])  # e.g., .../human_reaction_data
                    # category is the next component after 'raw' if present
                    category = parts[raw_idx + 1] if len(parts) > raw_idx + 1 else ""
                    # build processed paths following convention
                    if category and stem and suf:
                        face_val = str(base_root / "face" / category / f"{stem}_face" / f"{stem}{suf}")
                        stim_val_from_raw = str(base_root / "stimuli" / category / f"{stem}_stimuli" / f"{stem}{suf}")

            row_pairs: list[tuple[str, str]] = []
            if raw_val:
                for face_path, stim_path in derive_pairs_from_raw(Path(raw_val)):
                    row_pairs.append((str(face_path), str(stim_path)))
            if row_pairs:
                for face_str, stim_str in row_pairs:
                    if face_str not in dedup:
                        dedup[face_str] = {"face": face_str, "stim": stim_str}
                continue

            # Fallbacks when we could not derive from raw path
            if not face_val:
                # previous face selection fallback (try any face-like column)
                for k in face_keys:
                    if k in row and (row.get(k) or "").strip():
                        face_val = (row.get(k) or "").strip()
                        break
            # If no stimuli was derived from raw and no explicit col, try cols/rules
            stim_val = stim_val_from_raw
            if not stim_val:
                if stim_force and stim_force in row and (row.get(stim_force) or "").strip():
                    stim_val = (row.get(stim_force) or "").strip()
                else:
                    for k in stim_keys:
                        if k in row and (row.get(k) or "").strip():
                            stim_val = (row.get(k) or "").strip()
                            break
                if not stim_val:
                    # apply rules (user-provided or smart defaults)
                    rules = []
                    if args.stimuli_replace:
                        rules = [r for r in args.stimuli_replace if r and "=" in r]
                    else:
                        rules = ["/face_origin/=/stimuli/", "/face/=/stimuli/"]
                    if face_val and rules:
                        derived = face_val
                        for rule in rules:
                            src, dst = rule.split("=", 1)
                            derived = derived.replace(src, dst)
                        stim_val = derived

            if not face_val:
                face_missing += 1
                continue
            if not stim_val:
                stim_missing += 1
                continue

            # Dedup by processed face path string
            if face_val not in dedup:
                dedup[face_val] = {"face": face_val, "stim": stim_val}

    valid: list[tuple[Path, Path]] = []
    for face_str in sorted(dedup.keys()):
        face = Path(dedup[face_str]["face"])
        stim = Path(dedup[face_str]["stim"])
        if not face.is_file():
            face_not_file += 1
            continue
        if not stim.is_file():
            stim_not_file += 1
            continue
        valid.append((face, stim))

    if not valid:
        # Print helpful diagnostics
        print("[load_valid_entries] No valid entries constructed.")
        try:
            with csv_path.open(newline="", encoding="utf-8") as h2:
                r2 = csv.DictReader(h2)
                headers2 = r2.fieldnames or []
                print(f"[load_valid_entries] CSV headers: {headers2}")
        except Exception as _e:  # noqa: BLE001
            pass
        print(
            f"[load_valid_entries] rows={total_rows}, face_missing={face_missing}, "
            f"stim_missing={stim_missing}, face_not_file={face_not_file}, stim_not_file={stim_not_file}"
        )
        print("[load_valid_entries] Expecting face columns from:", face_keys)
        print("[load_valid_entries] Expecting stimuli columns from:", stim_keys)
        print(f"[load_valid_entries] face_col_forced={face_force!r} stimuli_col_forced={stim_force!r}")
        if args.stimuli_replace:
            print(f"[load_valid_entries] stimuli_replace rules: {args.stimuli_replace}")
        if not args.stimuli_replace:
            print("[load_valid_entries] using built-in default mapping rules: ['/face_origin/=/stimuli/', '/face/=/stimuli/']")
        print("[load_valid_entries] attempted raw→processed derivation: '/raw/<cat>/<stem>.ext' → 'face/<cat>/<stem>_face/<stem>.ext' & 'stimuli/<cat>/<stem>_stimuli/<stem>.ext'")

    return valid


def load_existing_mapping(dest_root: Path) -> tuple[set[str], int]:
    """Return (existing_original_face_paths, next_index) from mapping.csv if present.

    next_index is computed as 1 + max(existing new_id), or 0 if none.
    """
    mapping_path = dest_root / "mapping.csv"
    existing_face_sources: set[str] = set()
    next_index = 0
    if not mapping_path.exists():
        return existing_face_sources, next_index
    with mapping_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        max_id = -1
        for row in reader:
            # Keep original face_path as the unique key for deduplication
            src_face = (row.get("original_face_path") or "").strip()
            if src_face:
                existing_face_sources.add(src_face)
            # Track highest assigned id
            nid = (row.get("new_id") or "").strip()
            if nid.isdigit():
                try:
                    max_id = max(max_id, int(nid))
                except ValueError:
                    pass
        next_index = max_id + 1 if max_id >= 0 else 0
    return existing_face_sources, next_index



def ensure_clean_dir(path: Path, allow_nonempty: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if allow_nonempty:
        return
    if any(path.iterdir()):
        raise SystemExit(f"Destination directory {path} is not empty. Use --allow-nonempty to override.")


# Helper: compute next index from destination dirs for index collision avoidance
def next_index_from_dirs(face_origin_dir: Path, stimuli_dir: Path) -> int:
    """Return 1 + max five-digit numeric basename found in destination dirs, or 0 if none."""
    def max_idx_in(dirpath: Path) -> int:
        m = -1
        if dirpath.exists():
            for p in dirpath.iterdir():
                if not p.is_file():
                    continue
                name = p.stem  # e.g., '00012'
                if name.isdigit():
                    try:
                        m = max(m, int(name))
                    except ValueError:
                        pass
        return m
    return max(max_idx_in(face_origin_dir), max_idx_in(stimuli_dir)) + 1


def main() -> int:
    args = parse_args()
    entries = load_valid_entries(args.quality_csv, args)
    if args.limit is not None:
        entries = entries[: args.limit]
    if not entries:
        print("No entries found.")
        return 0

    dest_root = args.dest_root
    face_origin_dir = dest_root / "face_origin"
    stimuli_dir = dest_root / "stimuli"
    logs_dir = dest_root / "logs"

    # If resuming, read existing mapping and continue numbering; also allow non-empty dirs
    existing_face_sources: set[str] = set()
    next_index_from_map = 0
    if args.resume:
        existing_face_sources, next_index_from_map = load_existing_mapping(dest_root)
        # When resuming, allow non-empty directories implicitly
        args.allow_nonempty = True

    ensure_clean_dir(face_origin_dir, args.allow_nonempty)
    ensure_clean_dir(stimuli_dir, args.allow_nonempty)
    logs_dir.mkdir(parents=True, exist_ok=True)

    mapping_rows: list[list[str]] = []
    failures: list[tuple[str, str]] = []

    # Determine starting index: consider mapping.csv, destination folders, and user-provided start
    dir_next = next_index_from_dirs(face_origin_dir, stimuli_dir) if args.resume else 0
    current_index = max(args.start_index, next_index_from_map, dir_next) if args.resume else args.start_index

    for face_path, stimuli_path in entries:
        # Skip if this source has already been mapped (dedup by original face video path)
        if args.resume and str(face_path) in existing_face_sources:
            print(f"[SKIP] already in mapping: {face_path}")
            failures.append((str(face_path), "already_mapped"))
            continue

        # Probe forward for the next free index so we never collide with existing files
        probe_index = current_index
        while True:
            probe_str = f"{probe_index:05d}"
            probe_face = face_origin_dir / f"{probe_str}{face_path.suffix.lower()}"
            probe_stim = stimuli_dir / f"{probe_str}{Path(stimuli_path).suffix.lower()}"
            if not (probe_face.exists() or probe_stim.exists()):
                break
            probe_index += 1
        index_str = f"{probe_index:05d}"
        dest_face_origin = face_origin_dir / f"{index_str}{face_path.suffix.lower()}"
        dest_stimuli = stimuli_dir / f"{index_str}{Path(stimuli_path).suffix.lower()}"
        print(f"[{index_str}] processing {face_path}")

        if args.dry_run:
            # Record what would happen using the probed free index
            mapping_rows.append([
                index_str,
                str(face_path),
                str(stimuli_path),
                str(dest_face_origin),
                str(dest_stimuli),
            ])
            # Advance current_index to one past the probe, so the next item starts after this slot
            current_index = probe_index + 1
            continue

        try:
            dest_face_origin.parent.mkdir(parents=True, exist_ok=True)
            dest_stimuli.parent.mkdir(parents=True, exist_ok=True)

            shutil.move(str(face_path), str(dest_face_origin))
            shutil.move(str(stimuli_path), str(dest_stimuli))

        except Exception as exc:  # noqa: BLE001
            print(f"[{index_str}] ERROR: {exc}")
            failures.append((str(face_path), str(exc)))
            continue

        mapping_rows.append([
            index_str,
            str(face_path),
            str(stimuli_path),
            str(dest_face_origin),
            str(dest_stimuli),
        ])
        current_index = probe_index + 1

    mapping_path = dest_root / ("dry_run_mapping.csv" if args.dry_run else "mapping.csv")
    write_header = not mapping_path.exists() or not args.resume
    mode = "a" if args.resume and mapping_path.exists() and not args.dry_run else "w"
    with mapping_path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow([
                "new_id",
                "original_face_path",
                "original_stimuli_path",
                "new_face_origin_path",
                "new_stimuli_path",
            ])
        writer.writerows(mapping_rows)

    if args.dry_run:
        print(f"[DRY RUN] Simulated {len(mapping_rows)} new entries -> {mapping_path}")
    else:
        print(("Appended" if args.resume and mapping_rows else "Wrote") + f" mapping for {len(mapping_rows)} entries to {mapping_path}")
    if failures:
        failure_path = logs_dir / "failures.csv"
        fail_write_header = not failure_path.exists() or not args.resume
        fail_mode = "a" if args.resume and failure_path.exists() else "w"
        with failure_path.open(fail_mode, newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if fail_write_header:
                writer.writerow(["original_face_path", "error"])
            writer.writerows(failures)
        print(f"Encountered {len(failures)} failures; see {failure_path}")
    else:
        print("No failures encountered.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
