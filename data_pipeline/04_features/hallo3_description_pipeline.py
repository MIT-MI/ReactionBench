#!/usr/bin/env python3
"""
Generate AU-based facial expression descriptions for the Hallo3 benchmark.

Steps:
1. Run LibreFace (GPU if available, otherwise CPU) to segment videos by AU signatures.
2. Export short clips per AU segment and describe each via Vertex AI Gemini.
3. Save concise per-video mapping files under the description root.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "common"))
import rvb_config as cfg  # noqa: E402  (loads .env, puts every stage dir on sys.path)

import argparse
import json
import mimetypes
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import pandas as pd
from google import genai
from google.genai.types import GenerateContentConfig, Part


SCRIPT_DIR = Path(__file__).resolve().parent
SEGMENT_SCRIPT = SCRIPT_DIR / "libreface_segment_expressions.py"

DEFAULT_INPUT_ROOT = Path("/workspace/data/face_origin")
DEFAULT_SEGMENT_ROOT = Path("/workspace/data/face_segments")
DEFAULT_DESCRIPTION_ROOT = Path("/workspace/data/description")


def _enable_line_buffering() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except Exception:
            # Older Python/alternate streams do not support reconfigure; ignore.
            pass


_enable_line_buffering()


def _set_progress_env() -> None:
    """
    Ensure downstream subprocesses know to disable verbose progress bars when
    the pipeline runs non-interactively (stdout/stderr redirected to a log).
    """

    disable_flag = os.environ.get("HALLO3_DISABLE_PROGRESS")
    if disable_flag is None:
        if not sys.stderr.isatty():
            disable_flag = "1"
            os.environ["HALLO3_DISABLE_PROGRESS"] = disable_flag
    if disable_flag and disable_flag not in {"0", "false", "no", "False"}:
        os.environ.setdefault("TQDM_DISABLE", "1")


_set_progress_env()


@dataclass
class SegmentInfo:
    csv_path: Path
    row_index: int
    video_path: Path
    start_time: float
    end_time: float
    duration: float
    au_signature: str

    @property
    def stem(self) -> str:
        return self.video_path.stem


def run_cmd(cmd: List[str], *, extra_env: Optional[Dict[str, str]] = None) -> None:
    """Run a command with streaming output and fail loudly on non-zero exit."""
    print(f"[RUN] {' '.join(cmd)}", flush=True)
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    subprocess.run(cmd, check=True, env=env)


def ensure_positive(value: float, minimum: float) -> float:
    return max(minimum, float(value))


def normalise_sentence(text: str) -> str:
    cleaned = " ".join(str(text).strip().split())
    if not cleaned:
        return ""
    stop_idx = len(cleaned)
    for delimiter in [".", "!", "?"]:
        idx = cleaned.find(delimiter)
        if idx != -1 and (idx + 1) <= stop_idx:
            stop_idx = idx + 1
    sentence = cleaned[:stop_idx].strip() or cleaned
    if sentence and sentence[-1] not in ".!?":
        sentence += "."
    return sentence


def format_prompt(template: str, **kwargs) -> str:
    try:
        return template.format(**kwargs)
    except KeyError:
        return template


def load_video_id_list(path: Path) -> List[str]:
    """Load unique video stems from a text file (accepts bare IDs or filenames)."""
    stems: List[str] = []
    seen = set()
    for line in path.read_text().splitlines():
        token = line.strip()
        if not token or token.startswith("#"):
            continue
        token = token.split()[0]
        stem = Path(token).stem
        if stem and stem not in seen:
            stems.append(stem)
            seen.add(stem)
    return stems


def _derive_stem_from_csv(csv_path: Path) -> str:
    stem = csv_path.stem
    suffix = "_au_segments"
    if stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    return stem


def _part_from_local_file(path: Path, mime_fallback: str = "application/octet-stream") -> Part:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    mime, _ = mimetypes.guess_type(path.name)
    mime = mime or mime_fallback
    data = path.read_bytes()
    return Part.from_bytes(data=data, mime_type=mime)


def build_vertex_client(project: Optional[str], location: str) -> genai.Client:
    project_id = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        raise RuntimeError("Vertex project ID missing. Provide --project or set GOOGLE_CLOUD_PROJECT.")
    return genai.Client(vertexai=True, project=project_id, location=location)


def describe_clip(
    client: genai.Client,
    *,
    model: str,
    prompt: str,
    clip_path: Path,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    max_retries: int,
    retry_cooldown: float,
) -> str:
    parts: List[Part] = [
        Part.from_text(text=prompt),
        _part_from_local_file(clip_path, mime_fallback="video/mp4"),
    ]
    config = GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
        response_mime_type="application/json",
        response_schema={
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "One concise English sentence describing the visible facial expression.",
                }
            },
            "required": ["description"],
        },
    )

    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=parts,
                config=config,
            )
            desc = extract_description(response)
            desc = normalise_sentence(desc)
            if not desc:
                raise ValueError("Empty description returned by Gemini.")
            return desc
        except Exception as exc:  # pragma: no cover - external dependency
            last_error = exc
            if attempt < max_retries:
                time.sleep(max(0.0, retry_cooldown))
            else:
                raise RuntimeError(f"Gemini description failed for {clip_path.name}: {exc}") from exc
    assert last_error is not None
    raise RuntimeError(f"Gemini description failed for {clip_path.name}: {last_error}")


def extract_description(response) -> str:
    parsed = getattr(response, "parsed", None)
    if parsed:
        if isinstance(parsed, dict):
            text = parsed.get("description") or parsed.get("reaction")
            if isinstance(text, str) and text.strip():
                return text.strip()
        elif isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    text = item.get("description") or item.get("reaction")
                    if isinstance(text, str) and text.strip():
                        return text.strip()

    text = (getattr(response, "text", "") or "").strip()
    if text.startswith("{") or text.startswith("[" ):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            for key in ("description", "reaction"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            for value in data.values():
                if isinstance(value, str) and value.strip():
                    return value.strip()
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    for key in ("description", "reaction"):
                        value = item.get(key)
                        if isinstance(value, str) and value.strip():
                            return value.strip()
                elif isinstance(item, str) and item.strip():
                    return item.strip()
    if text:
        return text
    # As a last resort, inspect candidates
    candidates = getattr(response, "candidates", None)
    if candidates:
        for cand in candidates:
            content = getattr(cand, "content", None)
            if not content:
                continue
            parts = getattr(content, "parts", None)
            if not parts:
                continue
            for part in parts:
                if hasattr(part, "text") and part.text and part.text.strip():
                    return part.text.strip()
    return ""


def run_ffmpeg(
    *,
    ffmpeg_bin: str,
    segment: SegmentInfo,
    temp_dir: Path,
    reencode: bool,
    min_duration: float,
    x264_preset: Optional[str],
    use_hwaccel: bool,
    use_nvenc: bool,
    ffmpeg_threads: int,
    nvenc_preset: Optional[str],
    nvenc_cq: Optional[int],
) -> Path:
    clip_path = temp_dir / f"clip_{uuid.uuid4().hex}.mp4"
    start = max(0.0, float(segment.start_time))
    duration = ensure_positive(segment.duration, min_duration)

    def build_cmd(want_reencode: bool) -> List[str]:
        cmd: List[str] = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
        ]
        # GPU decode if requested
        if use_hwaccel:
            cmd += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        cmd += [
            "-ss",
            f"{start:.3f}",
            "-i",
            str(segment.video_path),
            "-t",
            f"{duration:.3f}",
        ]
        if want_reencode:
            # Choose encoder: NVENC vs libx264
            if use_nvenc:
                cmd += ["-c:v", "h264_nvenc"]
                # NVENC constant quality if provided, otherwise default 28
                if nvenc_preset:
                    cmd += ["-preset", nvenc_preset]
                if nvenc_cq is not None:
                    cmd += ["-cq", str(nvenc_cq)]
                else:
                    cmd += ["-cq", "28"]
            else:
                cmd += [
                    "-vf",
                    "scale=iw:ih",
                    "-c:v",
                    "libx264",
                ]
                if x264_preset:
                    cmd += ["-preset", x264_preset]
                cmd += ["-crf", "23"]
            # Audio
            cmd += [
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
            ]
            # Encoder threads (applies to encoders that honor it)
            if ffmpeg_threads and ffmpeg_threads > 0:
                cmd += ["-threads", str(ffmpeg_threads)]
        else:
            # Stream copy (fast, requires keyframe alignment)
            cmd += ["-c", "copy"]
        cmd.append(str(clip_path))
        return cmd

    try:
        subprocess.run(build_cmd(reencode), check=True)
    except subprocess.CalledProcessError:
        if reencode:
            if clip_path.exists():
                try:
                    clip_path.unlink()
                except FileNotFoundError:
                    pass
            print(f"[WARN] Re-encode failed for {segment.video_path.name}; retrying with stream copy", flush=True)
            subprocess.run(build_cmd(False), check=True)
        else:
            raise
    return clip_path


def load_segments(
    csv_path: Path,
    *,
    video_root: Optional[Path] = None,
    ignore_csv_video_path: bool = False,
) -> List[SegmentInfo]:
    df = pd.read_csv(csv_path)
    required_cols = {"start_time_s", "end_time_s", "duration_s"}
    if not ignore_csv_video_path:
        required_cols.add("video_path")
    if df.empty or not required_cols.issubset(df.columns):
        return []
    segments: List[SegmentInfo] = []
    missing_paths: Set[Path] = set()
    for idx, row in df.iterrows():
        try:
            if ignore_csv_video_path and video_root:
                raw_stem = row.get("video_stem") or _derive_stem_from_csv(csv_path)
                if not raw_stem:
                    raise ValueError("Missing video_stem while ignoring CSV video_path.")
                stem = Path(str(raw_stem)).stem
                candidates = [stem]
                if stem.isdigit():
                    padded = stem.zfill(5)
                    if padded not in candidates:
                        candidates.append(padded)
                video_path = (video_root / f"{candidates[0]}.mp4").resolve()
                for candidate in candidates:
                    candidate_path = (video_root / f"{candidate}.mp4").resolve()
                    if candidate_path.exists():
                        video_path = candidate_path
                        break
                    video_path = candidate_path
            else:
                video_path = Path(str(row["video_path"])).resolve()
            if not video_path.exists():
                if video_path not in missing_paths:
                    print(f"[WARN] Video path missing for segment: {video_path}")
                    missing_paths.add(video_path)
                continue
            segments.append(
                SegmentInfo(
                    csv_path=csv_path,
                    row_index=int(idx),
                    video_path=video_path,
                    start_time=float(row["start_time_s"]),
                    end_time=float(row["end_time_s"]),
                    duration=float(row["duration_s"]),
                    au_signature=str(row.get("au_signature", "")),
                )
            )
        except Exception as exc:
            print(f"[WARN] Failed to parse segment #{idx} in {csv_path}: {exc}")
    return segments


def annotate_segments(
    csv_path: Path,
    *,
    client: genai.Client,
    args: argparse.Namespace,
    per_video_prompt: str,
) -> List[Dict[str, object]]:
    segments = load_segments(
        csv_path,
        video_root=getattr(args, "video_root", None),
        ignore_csv_video_path=getattr(args, "ignore_csv_video_path", False),
    )
    if not segments:
        print(f"[WARN] No segments to annotate in {csv_path}")
        return []

    ffmpeg_bin = args.ffmpeg
    if shutil_which := shutil.which(ffmpeg_bin):
        ffmpeg_bin = shutil_which
    else:
        raise RuntimeError(f"ffmpeg binary not found: {ffmpeg_bin}")

    records: List[Dict[str, object]] = []
    total_segments = len(segments)
    use_context = getattr(args, "contextual_descriptions", False)

    def build_context_payload(existing: List[Dict[str, object]]) -> str:
        if not existing:
            return "No previous descriptions. Begin the running log with this clip."
        lines: List[str] = []
        for idx, item in enumerate(existing, start=1):
            lines.append(f"{idx}. {item['start_time_s']:.2f}-{item['end_time_s']:.2f}s: {item['description']}")
        return "\n".join(lines)

    with tempfile.TemporaryDirectory(dir=str(args.temp_root)) as tmp_dir:
        tmp_root = Path(tmp_dir)
        for seg in segments:
            clip_path = None
            try:
                clip_path = run_ffmpeg(
                    ffmpeg_bin=ffmpeg_bin,
                    segment=seg,
                    temp_dir=tmp_root,
                    reencode=not args.copy,
                    min_duration=args.min_clip_duration,
                    x264_preset=args.x264_preset,
                    use_hwaccel=args.use_gpu_decode,
                    use_nvenc=args.use_nvenc,
                    ffmpeg_threads=args.ffmpeg_threads,
                    nvenc_preset=args.nvenc_preset,
                    nvenc_cq=args.nvenc_cq,
                )
                segment_number = len(records) + 1
                context_payload = build_context_payload(records) if use_context else ""
                last_description = records[-1]["description"] if records else ""
                prompt = format_prompt(
                    per_video_prompt,
                    start_time=f"{seg.start_time:.2f}",
                    end_time=f"{seg.end_time:.2f}",
                    au_signature=seg.au_signature or "NONE",
                    context=context_payload,
                    previous_context=context_payload,
                    previous_descriptions=context_payload,
                    history=context_payload,
                    last_description=last_description,
                    segment_number=segment_number,
                    segment_index=segment_number - 1,
                    total_segments=total_segments,
                )
                desc = describe_clip(
                    client,
                    model=args.model,
                    prompt=prompt,
                    clip_path=clip_path,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_output_tokens=args.max_output_tokens,
                    max_retries=args.retry,
                    retry_cooldown=args.retry_cooldown,
                )
                records.append(
                    {
                        "video": seg.video_path.name,
                        "start_time_s": round(seg.start_time, 3),
                        "end_time_s": round(seg.end_time, 3),
                        "duration_s": round(seg.duration, 3),
                        "au_signature": seg.au_signature,
                        "description": desc,
                    }
                )
                if args.sleep_after_call > 0:
                    time.sleep(args.sleep_after_call)
            except Exception as exc:
                print(f"[ERROR] Failed to annotate segment {seg.video_path.name} ({seg.start_time:.2f}-{seg.end_time:.2f}s): {exc}")
            finally:
                if clip_path and clip_path.exists():
                    try:
                        clip_path.unlink()
                    except OSError:
                        pass
    return records


def export_mapping(records: List[Dict[str, object]], *, description_root: Path, stem: str) -> Path:
    if not records:
        raise ValueError("No records to export.")
    df = pd.DataFrame(records)
    df = df.sort_values("start_time_s").reset_index(drop=True)
    out_path = description_root / f"{stem}.csv"
    description_root.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AU segment descriptions for Hallo3 videos.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT, help="Directory containing source reaction videos.")
    parser.add_argument("--segment-root", type=Path, default=DEFAULT_SEGMENT_ROOT, help="Directory to store LibreFace segment CSVs.")
    parser.add_argument("--description-root", type=Path, default=DEFAULT_DESCRIPTION_ROOT, help="Directory for final mapping files.")
    parser.add_argument("--project", type=str, default=None, help="Vertex AI project ID (defaults to GOOGLE_CLOUD_PROJECT).")
    parser.add_argument("--location", type=str, default="us-central1", help="Vertex AI region.")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash", help="Gemini model.")
    parser.add_argument("--prompt", type=str, default=(
        "You are an attentive observer. Watch the short reaction clip carefully and respond with exactly one concise English sentence "
        "describing the visible facial expression or emotion. Focus only on clear visual evidence. "
        "Segment start: {start_time}s, end: {end_time}s, AU signature: {au_signature}."
    ), help="Prompt sent to Gemini. Supports {start_time}, {end_time}, {au_signature}.")
    parser.add_argument("--min-duration", type=float, default=1, help="Minimum AU segment duration in seconds for segmentation.")
    parser.add_argument("--au-threshold", type=float, default=3.0, help="AU intensity threshold for LibreFace segmentation.")
    parser.add_argument("--batch-size", type=int, default=64, help="LibreFace batch size (CPU).")
    parser.add_argument("--num-workers", type=int, default=max(1, min(8, os.cpu_count() or 8)), help="LibreFace dataloader workers.")
    parser.add_argument("--segment-temp-root", type=Path, default=None, help="Temporary directory for LibreFace frame extraction.")
    parser.add_argument(
        "--segment-device",
        type=str,
        default=None,
        help="Device passed to LibreFace segmentation (e.g. cuda:0). Defaults to LibreFace auto-detect.",
    )
    parser.add_argument("--segment-max-concurrent", type=int, default=1, help="Number of LibreFace processes to run in parallel.")
    parser.add_argument("--description-only", action="store_true", help="Skip segmentation and only generate descriptions from existing AU segment CSVs.")
    parser.add_argument("--max-output-tokens", type=int, default=256, help="Max tokens for Gemini response.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Gemini sampling temperature.")
    parser.add_argument("--top_p", type=float, default=1.0, help="Gemini top_p.")
    parser.add_argument("--retry", type=int, default=2, help="Retries for Gemini calls.")
    parser.add_argument("--retry-cooldown", type=float, default=5.0, help="Seconds to wait between retries.")
    parser.add_argument("--sleep-after-call", type=float, default=0.25, help="Throttle between successful Gemini calls.")
    parser.add_argument("--min-clip-duration", type=float, default=0.5, help="Minimum clip duration when exporting segments.")
    parser.add_argument("--ffmpeg", type=str, default="ffmpeg", help="Path to ffmpeg binary.")
    parser.add_argument("--x264-preset", type=str, default=None, help="Optional x264 preset for re-encoding.")
    parser.add_argument("--copy", action="store_true", help="Use stream copy instead of re-encoding clips.")
    parser.add_argument("--use-gpu-decode", action="store_true", help="Use GPU-accelerated decode via ffmpeg (-hwaccel cuda).")
    parser.add_argument("--use-nvenc", action="store_true", help="Use NVIDIA NVENC (h264_nvenc) for re-encoding instead of libx264.")
    parser.add_argument("--ffmpeg-threads", type=int, default=1, help="Number of encoder threads to pass to ffmpeg (-threads).")
    parser.add_argument("--nvenc-preset", type=str, default="p6", help="NVENC preset (e.g., p1..p7). Used when --use-nvenc is set.")
    parser.add_argument("--nvenc-cq", type=int, default=28, help="NVENC constant quality (lower=better). Used when --use-nvenc is set.")
    parser.add_argument("--temp-root", type=Path, default=None, help="Optional directory for temporary clips.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip segmentation if outputs up-to-date and skip existing description files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing description files.")
    parser.add_argument("--id-list", type=Path, default=None, help="Optional path to a text file with one video ID (stem or filename) per line to restrict processing.")
    args = parser.parse_args()

    filter_ids: Optional[List[str]] = None
    filter_set: Optional[Set[str]] = None
    if args.id_list is not None:
        args.id_list = args.id_list.expanduser().resolve()
        if not args.id_list.exists():
            raise FileNotFoundError(f"ID list not found: {args.id_list}")
        filter_ids = load_video_id_list(args.id_list)
        if not filter_ids:
            raise ValueError(f"ID list is empty: {args.id_list}")
        filter_set = set(filter_ids)
        print(f"[FILTER] Restricting pipeline to {len(filter_ids)} video IDs from {args.id_list}")

    if not args.description_only:
        if not SEGMENT_SCRIPT.exists():
            raise FileNotFoundError(f"Missing segmentation script: {SEGMENT_SCRIPT}")
    if not args.description_only and not args.input_root.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_root}")

    if not args.segment_root.exists():
        if args.description_only:
            raise FileNotFoundError(f"Segment directory not found: {args.segment_root}")
        args.segment_root.mkdir(parents=True, exist_ok=True)
    elif not args.segment_root.is_dir():
        raise NotADirectoryError(f"Segment root is not a directory: {args.segment_root}")

    if not args.description_only:
        if args.segment_temp_root is None:
            args.segment_temp_root = args.segment_root / ".tmp_frames"
        args.segment_temp_root.mkdir(parents=True, exist_ok=True)

    if args.temp_root is None:
        args.temp_root = Path(tempfile.gettempdir()) / "hallo3_vertex_clips"
    args.temp_root.mkdir(parents=True, exist_ok=True)

    # Ensure downstream description step infers video paths from the configured input root.
    args.video_root = args.input_root
    args.ignore_csv_video_path = True

    if not args.description_only:
        python_exe = sys.executable
        segment_cmd = [
            python_exe,
            str(SEGMENT_SCRIPT),
            str(args.input_root),
            str(args.segment_root),
            "--mode",
            "au",
            "--min-duration",
            str(args.min_duration),
            "--au-intensity-thresh",
            str(args.au_threshold),
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--heartbeat-seconds",
            "30",
        ]
        if args.segment_device:
            segment_cmd += [
                "--device",
                str(args.segment_device),
            ]
        segment_cmd += [
            "--temp-dir",
            str(args.segment_temp_root),
        ]
        if args.segment_max_concurrent and args.segment_max_concurrent > 1:
            segment_cmd += [
                "--max-concurrent",
                str(args.segment_max_concurrent),
            ]
        if filter_ids:
            segment_cmd += [
                "--include-list",
                str(args.id_list),
            ]
        if args.skip_existing:
            segment_cmd.append("--skip-existing")

        print("[STEP 1] LibreFace AU segmentation")
        progress_env: Dict[str, str] = {}
        disable_flag = os.environ.get("HALLO3_DISABLE_PROGRESS")
        if disable_flag and disable_flag.lower() not in {"0", "false", "no"}:
            progress_env["HALLO3_DISABLE_PROGRESS"] = disable_flag
            progress_env.setdefault("TQDM_DISABLE", os.environ.get("TQDM_DISABLE", "1"))
        run_cmd(segment_cmd, extra_env=progress_env or None)
    else:
        print("[MODE] Description-only: skipping LibreFace segmentation and using existing AU segment CSVs.")

    client = build_vertex_client(args.project, args.location)
    description_root = args.description_root
    description_root.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(
        p
        for p in args.segment_root.glob("*_au_segments.csv")
        if p.is_file() and not p.name.startswith(".")
    )
    if filter_set:
        csv_files = [p for p in csv_files if p.stem.replace("_au_segments", "") in filter_set]
        available = {p.stem.replace("_au_segments", "") for p in csv_files}
        missing = filter_set - available
        if missing:
            missing_preview = ", ".join(sorted(missing)[:5])
            if len(missing) > 5:
                missing_preview += ", ..."
            print(f"[WARN] No *_au_segments.csv found for {len(missing)} requested IDs: {missing_preview}")
    if not csv_files:
        if filter_set:
            raise RuntimeError(f"No *_au_segments.csv files found for the requested IDs under {args.segment_root}")
        raise RuntimeError(f"No *_au_segments.csv files found under {args.segment_root}")

    step_label = "[STEP 2]" if not args.description_only else "[STEP]"
    print(f"{step_label} Gemini descriptions via Vertex AI ({len(csv_files)} files)")

    processed = 0
    for csv_path in csv_files:
        stem = csv_path.stem.replace("_au_segments", "")
        out_path = description_root / f"{stem}.csv"
        if out_path.exists() and args.skip_existing and not args.overwrite:
            print(f"[SKIP] {out_path.name} already exists (use --overwrite to regenerate).")
            continue

        records = annotate_segments(
            csv_path,
            client=client,
            args=args,
            per_video_prompt=args.prompt,
        )
        if not records:
            print(f"[WARN] No descriptions generated for {csv_path.name}")
            continue
        out_path = export_mapping(records, description_root=description_root, stem=stem)
        print(f"[OK] {csv_path.name} -> {out_path.name} ({len(records)} segments)")
        processed += 1

    if processed == 0:
        raise RuntimeError("No description files were produced.")

    print(f"[DONE] Generated {processed} description files under {description_root}")


if __name__ == "__main__":
    main()
