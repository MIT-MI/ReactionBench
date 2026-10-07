#!/usr/bin/env python3
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "common"))
import rvb_config as cfg  # noqa: E402  (loads .env, puts every stage dir on sys.path)

"""
Batch runner for the talking-head (ghs_crop) extraction pipeline.

Given an input directory tree of videos, this script mirrors the structure in an
output directory and runs the ghs_crop tool to generate head-and-shoulders crops.

Example:
    python data_pipeline/02_align/talking_head_batch_crop.py \
        --input-root $RVB_RAW_ROOT/patreon/face \
        --output-root $RVB_RAW_ROOT/patreon/face_cropped
"""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import threading
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_MIN_OK_BYTES = 16 * 1024
DEFAULT_PROVIDERS = "TensorrtExecutionProvider,CUDAExecutionProvider"
DEFAULT_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
DEFAULT_TIMEOUT_SEC = 600
DEFAULT_WORKERS = max(1, min(4, os.cpu_count() or 1))


@dataclass
class RunnerConfig:
    talking_head_root: Path
    providers: str | None
    fast_mode: bool
    skip_existing: bool
    min_ok_bytes: int
    timeout: int
    dry_run: bool
    verbose: bool
    omp_threads: int
    trt_cache_root: Path | None
    max_workers: int


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run talking-head ghs_crop extraction on a directory tree."
    )
    parser.add_argument(
        "--input-root",
        required=True,
        help="Root directory containing source videos.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Directory where cropped videos will be written (structure mirrors input).",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=sorted(DEFAULT_VIDEO_EXTENSIONS),
        help="File extensions (case-insensitive) to process. Default: %(default)s",
    )
    parser.add_argument(
        "--providers",
        default=DEFAULT_PROVIDERS,
        help="ONNX Runtime execution providers string passed to ghs_crop.",
    )
    parser.add_argument(
        "--min-ok-bytes",
        type=int,
        default=DEFAULT_MIN_OK_BYTES,
        help="Minimum output size (bytes) treated as a successful crop.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SEC,
        help="Per-video timeout in seconds for ghs_crop.",
    )
    parser.add_argument(
        "--accurate",
        dest="fast_mode",
        action="store_false",
        help="Use slower, higher-recall parameters (default is fast mode).",
    )
    parser.set_defaults(fast_mode=True)
    parser.add_argument(
        "--no-skip",
        dest="skip_existing",
        action="store_false",
        help="Re-run extraction even if output already exists.",
    )
    parser.set_defaults(skip_existing=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print planned actions without running ghs_crop.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra logging (including ghs_crop stdout/stderr on success).",
    )
    parser.add_argument(
        "--omp-threads",
        type=int,
        default=4,
        help="Threads hint for OpenMP / MKL heavy dependencies.",
    )
    parser.add_argument(
        "--trt-cache-root",
        help="Optional directory to store TensorRT engine cache files.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Process at most this many files (0 means no limit).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Maximum parallel workers. Default: {DEFAULT_WORKERS}",
    )
    return parser.parse_args(argv)


def _normalize_extensions(exts: Iterable[str]) -> set[str]:
    normalized = set()
    for ext in exts:
        ext = ext.strip()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = f".{ext}"
        normalized.add(ext.lower())
    return normalized


def _providers_require_gpu(providers: str | None) -> bool:
    if not providers:
        return False
    for token in providers.split(","):
        t = token.strip().lower()
        if not t:
            continue
        if "cuda" in t or "tensorrt" in t:
            return True
    return False


def _is_ok(path: Path, min_bytes: int) -> bool:
    try:
        return path.is_file() and path.stat().st_size >= max(1, min_bytes)
    except OSError:
        return False


def _discover_videos(input_root: Path, extensions: set[str]) -> list[Path]:
    videos = []
    for node in sorted(input_root.rglob("*")):
        if node.is_file() and node.suffix.lower() in extensions:
            videos.append(node)
    return videos


def _run_single(
    src: Path,
    dst: Path,
    display_name: str,
    *,
    cfg: RunnerConfig,
) -> tuple[bool, str]:
    rel = dst.name
    if cfg.skip_existing and _is_ok(dst, cfg.min_ok_bytes):
        return True, f"skip-existing ({display_name})"

    if cfg.dry_run:
        return True, f"dry-run ({display_name})"

    dst.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", str(cfg.omp_threads))
    env.setdefault("MKL_NUM_THREADS", str(cfg.omp_threads))
    env.setdefault("CUDA_MODULE_LOADING", "LAZY")

    cache_root = cfg.trt_cache_root
    if cache_root is None:
        cache_root = dst.parent / ".trt_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    env["ORT_TENSORRT_ENGINE_CACHE_ENABLE"] = "1"
    env["ORT_TENSORRT_ENGINE_CACHE_PATH"] = str(cache_root)
    env["ORT_TENSORRT_FP16_ENABLE"] = "1"
    env["ORT_TENSORRT_MAX_WORKSPACE_SIZE"] = str(8 * 1024 * 1024 * 1024)
    if _providers_require_gpu(cfg.providers):
        env["ORT_DISABLE_CPU"] = "1"
        env.setdefault("CUDA_VISIBLE_DEVICES", env.get("CUDA_VISIBLE_DEVICES", "0"))
    else:
        env.setdefault("ORT_DISABLE_CPU", "0")

    def _providers_fallback_list(providers: str | None) -> list[str]:
        base = (providers or "").strip() or "TensorrtExecutionProvider,CUDAExecutionProvider"
        order = [base, "CUDAExecutionProvider", "CPUExecutionProvider"]
        seen: set[str] = set()
        out: list[str] = []
        for p in order:
            if p and p not in seen:
                seen.add(p)
                out.append(p)
        return out

    def _run_stage(
        prov: str,
        *,
        step: str,
        conf_thres: str,
        det_size: str,
        safety_scale: str,
        face_pct: str,
        timeout: int,
        tag: str,
    ) -> tuple[bool, str]:
        cmd = [
            sys.executable,
            "-m",
            "ghs_crop",
            "--input",
            str(src),
            "--output",
            str(dst),
            "--step",
            step,
            "--conf-thres",
            conf_thres,
            "--det-size",
            det_size,
            "--headroom-bias",
            "0.05",
            "--safety-scale",
            safety_scale,
            "--face-area-percentile",
            face_pct,
            "--providers",
            prov,
        ]
        env_local = env.copy()
        if "CPUExecutionProvider" in prov:
            env_local["ORT_DISABLE_CPU"] = "0"
        start = time.time()
        try:
            completed = subprocess.run(
                cmd,
                check=True,
                capture_output=not cfg.verbose,
                text=True,
                env=env_local,
                timeout=timeout,
            )
            elapsed = time.time() - start
            if cfg.verbose and completed.stdout:
                print(f"[{display_name}] {completed.stdout.strip()}")
            if cfg.verbose and completed.stderr:
                print(f"[{display_name}] {completed.stderr.strip()}")
            if not _is_ok(dst, cfg.min_ok_bytes):
                return False, f"{tag}: output-too-small"
            return True, f"{tag}: {elapsed:.1f}s via {prov}"
        except subprocess.TimeoutExpired:
            return False, f"{tag}: timeout via {prov}"
        except subprocess.CalledProcessError as exc:
            details = exc.stderr or exc.stdout or str(exc)
            detail_tail = details.strip().splitlines()[-1] if details else "unknown error"
            return False, f"{tag}: ghs_crop failed via {prov}: {detail_tail}"

    stages: list[tuple[str, dict[str, str]]] = []
    if cfg.fast_mode:
        stages.append(("fast", {"step": "10", "conf_thres": "0.6", "det_size": "384", "safety_scale": "1.5", "face_pct": "60"}))
    else:
        stages.append(("accurate", {"step": "3", "conf_thres": "0.7", "det_size": "640", "safety_scale": "1.3", "face_pct": "60"}))
    stages.append(("robust", {"step": "3", "conf_thres": "0.30", "det_size": "640", "safety_scale": "1.5", "face_pct": "55"}))
    stages.append(("ultra", {"step": "1", "conf_thres": "0.15", "det_size": "960", "safety_scale": "1.1", "face_pct": "40"}))

    orig_cwd = Path.cwd()
    try:
        os.chdir(cfg.talking_head_root)
        for stage_idx, (tag, params) in enumerate(stages, start=1):
            provider_attempts = _providers_fallback_list(cfg.providers)
            print(f"      [{display_name}] stage {stage_idx}/{len(stages)} {tag} params={params} providers={provider_attempts}")
            for prov in provider_attempts:
                ok, msg = _run_stage(
                    prov,
                    step=params["step"],
                    conf_thres=params["conf_thres"],
                    det_size=params["det_size"],
                    safety_scale=params["safety_scale"],
                    face_pct=params["face_pct"],
                    timeout=cfg.timeout,
                    tag=tag,
                )
                print(f"        [{display_name}] {msg}")
                if ok:
                    return True, msg
        return False, "all stages failed"
    finally:
        os.chdir(orig_cwd)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    if not input_root.exists() or not input_root.is_dir():
        print(f"Input directory does not exist or is not a directory: {input_root}", file=sys.stderr)
        return 1

    talking_head_root = (Path(__file__).resolve().parent / "talking_head").resolve()

    if not talking_head_root.exists():
        print(f"Cannot locate talking_head tool at {talking_head_root}", file=sys.stderr)
        return 1

    extensions = _normalize_extensions(args.extensions)
    if not extensions:
        print("No valid extensions provided.", file=sys.stderr)
        return 1

    if _providers_require_gpu(args.providers):
        try:
            import torch  # type: ignore

            if not torch.cuda.is_available():
                print(
                    "Requested CUDA/TensorRT providers but no CUDA device is available. "
                    "Override with --providers CPUExecutionProvider if you intend to run on CPU.",
                    file=sys.stderr,
                )
                return 1
        except ImportError:
            print(
                "PyTorch is required to validate CUDA availability. Install torch or "
                "override with --providers CPUExecutionProvider.",
                file=sys.stderr,
            )
            return 1

    videos = _discover_videos(input_root, extensions)
    if args.max_files > 0:
        videos = videos[: args.max_files]

    if not videos:
        print(f"No matching videos found under {input_root}.")
        return 0

    # build config
    cfg = RunnerConfig(
        talking_head_root=talking_head_root,
        providers=args.providers,
        fast_mode=args.fast_mode,
        skip_existing=args.skip_existing,
        min_ok_bytes=args.min_ok_bytes,
        timeout=args.timeout,
        dry_run=args.dry_run,
        verbose=args.verbose,
        omp_threads=args.omp_threads,
        trt_cache_root=Path(args.trt_cache_root).expanduser().resolve()
        if args.trt_cache_root
        else None,
        max_workers=max(1, args.workers),
    )

    # Group videos by their parent folder (relative to input_root)
    from collections import defaultdict
    folder_to_videos = defaultdict(list)
    for vid in videos:
        rel = vid.relative_to(input_root)
        folder = rel.parent  # may be Path('.')
        folder_to_videos[folder].append(vid)

    # Determine which folders to skip based on output_root
    folders_to_skip = set()
    for folder in folder_to_videos:
        out_folder = output_root / folder
        if out_folder.exists() and out_folder.is_dir():
            print(f"[SKIP FOLDER] {folder} already exists in output, skipping all contained videos.")
            folders_to_skip.add(folder)

    # Compose the work_items list, skipping videos in skipped folders
    work_items: list[tuple[int, Path, Path, str]] = []
    all_videos_to_process = []
    for folder, vids in folder_to_videos.items():
        if folder in folders_to_skip:
            continue
        all_videos_to_process.extend(vids)
    total = len(all_videos_to_process)
    successes = 0
    failures = 0

    print(f"Discovered {total} video(s) to process.")
    for index, src in enumerate(all_videos_to_process, start=1):
        rel = src.relative_to(input_root)
        dst = output_root / rel
        display_name = str(rel)
        work_items.append((index, src, dst, display_name))

    if cfg.dry_run:
        for index, _, dst, display_name in work_items:
            print(f"[{index}/{total}] {display_name} -> {dst} (dry-run)")
        return 0

    print(f"Using up to {cfg.max_workers} parallel worker(s).")
    start_time = time.time()
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=cfg.max_workers) as executor:
        future_map = {}
        for index, src, dst, display_name in work_items:
            if cfg.skip_existing and _is_ok(dst, cfg.min_ok_bytes):
                print(f"[{index}/{total}] {display_name} -> {dst} | SKIP: existing OK file")
                successes += 1
                continue
            future = executor.submit(_run_single, src, dst, display_name, cfg=cfg)
            future_map[future] = (index, display_name, dst)

        for future in as_completed(future_map):
            index, display_name, dst = future_map[future]
            try:
                ok, message = future.result()
            except Exception as exc:
                ok = False
                message = f"exception: {exc}"
            status = "OK" if ok else "FAIL"
            with lock:
                if ok:
                    successes += 1
                else:
                    failures += 1
            print(f"[{index}/{total}] {display_name} -> {dst} | {status}: {message}", flush=True)

    elapsed = time.time() - start_time
    print(f"Completed. Successes: {successes}, Failures: {failures}. Elapsed {elapsed:.1f}s.")
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
