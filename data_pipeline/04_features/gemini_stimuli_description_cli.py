#!/usr/bin/env python3
"""
Generate stimulus video descriptions from existing LibreFace segment outputs using Gemini (Vertex AI or API key).

This CLI mirrors `gemini_au_description_cli.py` but instead of focusing on the
participant's face footage it applies the same time segments to the paired
stimulus videos located under --root/stimuli/{ID}.mp4 and exports the responses
to --root/stimuli_description/{ID}.csv.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "common"))
import rvb_config as cfg  # noqa: E402  (loads .env, puts every stage dir on sys.path)

import argparse
import copy
import os
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, List, Optional, Sequence, Set

from google import genai  # type: ignore

from concurrent.futures import ThreadPoolExecutor, as_completed

THIS_DIR = Path(__file__).resolve().parent
# rvb_config put 04_features/ and common/ on sys.path.
import hallo3_description_pipeline as pipeline  # noqa: E402
from gemini_client_pool import GeminiClientPool, gather_api_keys  # noqa: E402


def _build_gemini_client(
    auth_mode: str,
    project: Optional[str],
    location: str,
    api_keys: Optional[Sequence[str]] = None,
    api_key_files: Optional[Sequence[Path]] = None,
) -> genai.Client:
    mode = (auth_mode or "vertex").lower()
    if mode == "vertex":
        return pipeline.build_vertex_client(project, location)
    if mode == "api-key":
        keys = gather_api_keys(api_keys, api_key_files)
        if not keys:
            raise RuntimeError(
                "Gemini API key not provided. Use --gemini-api-key/--gemini-api-key-file or set "
                "GEMINI_API_KEY/GOOGLE_API_KEY when --gemini-auth=api-key."
            )
        if len(keys) == 1:
            return genai.Client(api_key=keys[0])
        return GeminiClientPool(
            keys,
            lambda key: genai.Client(api_key=key),
        )
    raise ValueError(f"Unsupported --gemini-auth mode: {auth_mode}")


def _default_prompt() -> str:
    return (
        "You are building a running, hyper-detailed log of everything that happens in a stimulus video. "
        "Use the PREVIOUS CONTEXT to stay consistent, but only repeat details if they clearly evolve. "
        "Describe the current clip (segment {segment_number} of {total_segments}, {start_time}s–{end_time}s) "
        "with 2–3 vivid sentences that call out visible actions, objects, scene dynamics, camera movement, "
        "and any new or changing elements. Prioritize what is happening in this exact moment and make sure "
        "each sentence introduces fresh, precise information.\n\n"
        "PREVIOUS CONTEXT:\n"
        "{context}\n\n"
        "CURRENT CLIP INSTRUCTIONS:\n"
        "- Focus on newly observed or evolving details.\n"
        "- If nothing changes, state what continues and what remains static.\n"
        "- Avoid speculation or dialogue; describe only what can be seen.\n"
        "Return only the description text."
    )


def _iter_segment_files(segment_root: Path) -> Iterable[Path]:
    if not segment_root.exists():
        raise FileNotFoundError(f"Segment root not found: {segment_root}")
    for path in sorted(segment_root.glob("*_au_segments.csv")):
        if path.name.startswith("."):
            continue
        yield path


def _normalize_stem(path: Path) -> str:
    return path.stem.replace("_au_segments", "")


def _load_id_list(path: Path) -> List[str]:
    stems: List[str] = []
    seen: Set[str] = set()
    for line in path.read_text().splitlines():
        token = line.strip()
        if not token or token.startswith("#"):
            continue
        stem = Path(token.split()[0]).stem
        if not stem or stem in seen:
            continue
        stems.append(stem)
        seen.add(stem)
    return stems


def _collect_targets(
    segment_root: Path,
    explicit: List[Path] | None,
    limit: int | None,
    allowed_stems: Set[str] | None,
) -> List[Path]:
    def _accept(path: Path) -> bool:
        if allowed_stems is None:
            return True
        return _normalize_stem(path) in allowed_stems

    files: List[Path] = []
    if explicit:
        for item in explicit:
            path = item.resolve()
            if path.is_file():
                if _accept(path):
                    files.append(path)
            else:
                print(f"[WARN] Segment file missing: {path}")
    else:
        for csv_path in _iter_segment_files(segment_root.resolve()):
            if not _accept(csv_path):
                continue
            files.append(csv_path)
            if limit is not None and len(files) >= limit:
                break
    return files


def build_args_namespace(cli_args: argparse.Namespace, temp_root: Path) -> SimpleNamespace:
    video_root = cli_args.video_root.resolve() if cli_args.video_root else None
    return SimpleNamespace(
        ffmpeg=cli_args.ffmpeg,
        ffmpeg_threads=cli_args.ffmpeg_threads,
        copy=cli_args.copy,
        temp_root=temp_root,
        min_clip_duration=cli_args.min_clip_duration,
        x264_preset=cli_args.x264_preset,
        model=cli_args.model,
        temperature=cli_args.temperature,
        top_p=cli_args.top_p,
        max_output_tokens=cli_args.max_output_tokens,
        retry=cli_args.retry,
        retry_cooldown=cli_args.retry_cooldown,
        sleep_after_call=cli_args.sleep_after_call,
        video_root=video_root,
        use_gpu_ffmpeg=cli_args.use_gpu_ffmpeg,
        gpu_codec=cli_args.gpu_codec,
        hwaccel=cli_args.hwaccel,
        force_clip_only=True,
        use_gpu_decode=cli_args.use_gpu_decode,
        decode_hwaccel=cli_args.decode_hwaccel,
        decode_hwaccel_device=cli_args.decode_hwaccel_device,
        use_nvenc=cli_args.use_gpu_ffmpeg,
        nvenc_codec=cli_args.gpu_codec,
        nvenc_preset=cli_args.x264_preset,
        nvenc_cq=cli_args.nvenc_cq,
        contextual_descriptions=True,
    )


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate stimulus video descriptions for existing LibreFace segment outputs using Gemini (Vertex or API key)."
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Dataset root directory containing subfolders: face_segments/, stimuli/, stimuli_description/."
    )
    parser.add_argument(
        "--segments",
        type=Path,
        nargs="+",
        default=None,
        help="Optional explicit list of *_au_segments.csv files to annotate.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many segment files (useful for spot checks).",
    )
    parser.add_argument("--project", type=str, default=None, help="Vertex AI project ID (ignored when --gemini-auth=api-key).")
    parser.add_argument("--location", type=str, default="us-central1", help="Vertex AI region.")
    parser.add_argument("--gemini-auth", choices=("vertex", "api-key"), default="vertex", help="Authenticate via Vertex (default) or Gemini API key.")
    parser.add_argument("--gemini-api-key", dest="gemini_api_keys", action="append", default=None, help="Gemini API key(s) when --gemini-auth=api-key (repeat flag or comma-separate entries).")
    parser.add_argument("--gemini-api-key-file", dest="gemini_api_key_files", action="append", type=Path, default=None, help="Path to a file containing additional Gemini API keys (one per line or comma-separated).")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash", help="Gemini model name.")
    parser.add_argument("--prompt", type=str, default=_default_prompt(), help="Prompt template passed to Gemini.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Gemini sampling temperature.")
    parser.add_argument("--top_p", type=float, default=1.0, help="Gemini top_p.")
    parser.add_argument("--max-output-tokens", type=int, default=256, help="Maximum tokens for Gemini response.")
    parser.add_argument("--retry", type=int, default=2, help="Number of retries for Gemini calls.")
    parser.add_argument("--retry-cooldown", type=float, default=5.0, help="Seconds to sleep between retries.")
    parser.add_argument("--sleep-after-call", type=float, default=0.25, help="Delay between successful Gemini calls.")
    parser.add_argument("--min-clip-duration", type=float, default=0.5, help="Minimum clip duration when exporting.")
    parser.add_argument("--ffmpeg", type=str, default="ffmpeg", help="ffmpeg binary.")
    parser.add_argument("--x264-preset", type=str, default=None, help="Optional x264 preset for re-encoding.")
    parser.add_argument("--copy", action="store_true", help="Use stream copy instead of re-encoding clips.")
    parser.add_argument(
        "--temp-root",
        type=Path,
        default=None,
        help="Optional directory for temporary clips (default: system temp directory).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip videos whose description CSV already exists.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing description CSVs even if they exist.",
    )
    parser.add_argument(
        "--id-list",
        type=Path,
        default=None,
        help="Optional text file containing video IDs or filenames (one per line) to restrict processing.",
    )
    parser.add_argument("--max-parallel", type=int, default=1, help="Process up to this many *_au_segments.csv files concurrently.")
    parser.add_argument("--target-qps", type=float, default=None, help="Global target QPS to Gemini across all workers; used to auto-pace calls.")
    parser.add_argument("--use-gpu-ffmpeg", action="store_true", help="Use GPU-accelerated ffmpeg for clip export (e.g., NVENC).")
    parser.add_argument("--gpu-codec", type=str, default="h264_nvenc", help="GPU video codec to use when --use-gpu-ffmpeg is enabled.")
    parser.add_argument("--hwaccel", type=str, default="cuda", help="ffmpeg hwaccel to use when --use-gpu-ffmpeg is enabled (e.g., cuda).")
    parser.add_argument("--use-gpu-decode", action="store_true", help="Use GPU-accelerated decode (e.g., hwaccel for reading source video).")
    parser.add_argument("--decode-hwaccel", type=str, default="cuda", help="Decoder hwaccel to use when --use-gpu-decode is enabled (e.g., cuda, vaapi).")
    parser.add_argument("--decode-hwaccel-device", type=str, default=None, help="Optional hwaccel device index/path for decode (e.g., 0 or /dev/dri/renderD128).")
    parser.add_argument(
        "--ffmpeg-threads",
        type=int,
        default=None,
        help="Number of threads to pass to ffmpeg (-threads N). Use None/omit to let ffmpeg decide."
    )
    parser.add_argument(
        "--nvenc-cq",
        type=float,
        default=None,
        help="NVENC constant quality (CQ) value (e.g., 19–28). If not set, pipeline default is used."
    )
    return parser.parse_args()


def main() -> None:
    cli_args = parse_cli()

    data_root = cli_args.root.resolve()
    segment_root = data_root / "face_segments"
    description_root = data_root / "stimuli_description"
    video_root = data_root / "stimuli"

    description_root.mkdir(parents=True, exist_ok=True)
    if cli_args.temp_root:
        temp_root = cli_args.temp_root.resolve()
    else:
        temp_root = Path(tempfile.gettempdir()) / "gemini_vertex_stimuli_clips"
    temp_root.mkdir(parents=True, exist_ok=True)

    allowed_stems: Set[str] | None = None
    if cli_args.id_list is not None:
        id_path = cli_args.id_list.expanduser().resolve()
        if not id_path.exists():
            raise FileNotFoundError(f"ID list file not found: {id_path}")
        ids = _load_id_list(id_path)
        if not ids:
            raise ValueError(f"ID list is empty: {id_path}")
        allowed_stems = set(ids)

    setattr(cli_args, "video_root", video_root)

    targets = _collect_targets(segment_root, cli_args.segments, cli_args.limit, allowed_stems)
    if not targets:
        print("[INFO] No segment CSVs to process.")
        return
    if allowed_stems is not None and not cli_args.segments:
        found = {_normalize_stem(path) for path in targets}
        missing = allowed_stems - found
        if missing:
            preview = ", ".join(sorted(missing)[:5])
            if len(missing) > 5:
                preview += ", ..."
            print(
                f"[WARN] {len(missing)} IDs from --id-list not found under {segment_root}: {preview}",
                file=sys.stderr,
            )

    client: genai.Client = _build_gemini_client(
        cli_args.gemini_auth,
        cli_args.project,
        cli_args.location,
        cli_args.gemini_api_keys,
        cli_args.gemini_api_key_files,
    )
    shared_args = build_args_namespace(cli_args, temp_root)

    processed = 0
    skipped = 0

    if cli_args.target_qps and cli_args.max_parallel > 0:
        per_worker_sleep = max(shared_args.sleep_after_call, (cli_args.max_parallel / float(cli_args.target_qps)))
    else:
        per_worker_sleep = shared_args.sleep_after_call

    def worker(csv_path: Path):
        stem = csv_path.stem.replace("_au_segments", "")
        out_path = description_root / f"{stem}.csv"
        if out_path.exists():
            try:
                if cli_args.skip_existing and not cli_args.overwrite and out_path.stat().st_size > 0:
                    return ("skipped", csv_path.name, out_path.name, 0)
            except Exception:
                pass
        try:
            worker_args = copy.copy(shared_args)
            worker_args.sleep_after_call = per_worker_sleep
            worker_args.video_root = video_root
            worker_args.ignore_csv_video_path = True
            records = pipeline.annotate_segments(
                csv_path,
                client=client,
                args=worker_args,
                per_video_prompt=cli_args.prompt,
            )
            if not records:
                return ("warn", csv_path.name, out_path.name, 0)
            pipeline.export_mapping(records, description_root=description_root, stem=stem)
            return ("ok", csv_path.name, out_path.name, len(records))
        except Exception as exc:
            return ("error", csv_path.name, str(exc), 0)

    with ThreadPoolExecutor(max_workers=max(1, cli_args.max_parallel)) as ex:
        futures = {ex.submit(worker, csv): csv for csv in targets}
        for fut in as_completed(futures):
            status, src, dst_or_err, nseg = fut.result()
            if status == "ok":
                print(f"[OK] {src} -> {dst_or_err} ({nseg} segments)")
                processed += 1
            elif status == "skipped":
                skipped += 1
            elif status == "warn":
                print(f"[WARN] No descriptions produced for {src}")
            else:
                print(f"[ERROR] Failed to process {src}: {dst_or_err}")

    if skipped > 0:
        print(f"[INFO] Skipped {skipped} existing description files.")
    print(f"[DONE] Generated {processed} stimulus description files (skipped {skipped}).")


if __name__ == "__main__":
    main()
