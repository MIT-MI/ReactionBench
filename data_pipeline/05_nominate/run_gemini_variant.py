#!/usr/bin/env python3
"""Run one Gemini pre-screening variant over the Phase 0 pilot clips.

One process = one variant = one (model, media_resolution, fps) triple, so V1/V2 and
the resolution/fps ablation are the same code path with different flags.

The prompt carries the Stage A anchored 0-3 scale verbatim so the model
and the human raters are answering the same question, and it holds the model to the
same nomination budget as V0 (top-2, 2-4s each) so Precision@2 compares like with like.
Returning an empty list is explicitly allowed -- that is the no-highlight test.

Resumable: already-processed ids in the output JSONL are skipped.

Example:
  python data_pipeline/05_nominate/run_gemini_variant.py --variant V2_flash_low \
      --model gemini-3.6-flash --media_resolution LOW --fps 1
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "common"))
import rvb_config as cfg  # noqa: E402  (loads .env, puts every stage dir on sys.path)

import argparse
import json
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from google import genai
from google.genai import types

from build_pilot_frame import default_data_root

# gemini-3.6-flash list price, USD per 1M tokens (Batch is half). Used only for the
# cost line in the run log -- keep in sync with ai.google.dev/gemini-api/docs/pricing.
PRICES: Dict[str, tuple[float, float]] = {
    "gemini-3.6-flash": (1.50, 7.50),
    "gemini-3.5-flash": (1.50, 9.00),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3.1-pro-preview": (2.00, 12.00),
}

MODALITIES_AV = "You can see the reviewer and hear the audio."
MODALITIES_V = ("This clip has no audio track, so judge from the picture alone. "
                "Do not speculate about sounds you cannot hear.")
EVIDENCE_AV = "Name the visual cue, the audio cue, or both."
EVIDENCE_V = "Name the visual cue."

ANCHORS = """0 = calm viewing: no visible or audible reaction.
1 = mild: slight smile, nod, small head movement.
2 = clear: audible laugh, exclamation, raised brows with a forward lean.
3 = peak: scream, jump, hands to face, recoiling backwards."""

PROMPT = """You are screening a reaction video for the single strongest moments of reviewer reaction.

The clip is {duration:.1f} seconds long. {modalities}

Rate reaction strength on this behaviourally anchored scale:
{anchors}

Nominate at most {top_k} moments, strongest first. Each nominated moment must be
between {min_window:.0f} and {max_window:.0f} seconds long and the moments must not overlap.
Give start and end as seconds from the start of the clip, as decimal numbers.

For each nomination give the anchored score (0-3) and one sentence of evidence that
cites what you actually observed. {evidence_hint}

If no moment in this clip reaches 2 on the scale, return an empty list. Do not pad the
list to reach {top_k}; nominating a weak moment is worse than nominating nothing."""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "nominations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_s": {"type": "number"},
                    "end_s": {"type": "number"},
                    "score": {"type": "integer"},
                    "evidence": {"type": "string"},
                },
                "required": ["start_s", "end_s", "score", "evidence"],
            },
        },
    },
    "required": ["nominations"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", required=True, help="variant name, e.g. V2_flash_low")
    p.add_argument("--model", default="gemini-3.6-flash")
    p.add_argument("--media_resolution", default="LOW", choices=["LOW", "MEDIUM", "HIGH"],
                   help="HIGH is billed and tokenised identically to MEDIUM for video")
    p.add_argument("--fps", type=float, default=1.0, help="video sampling rate; 1.0 is the API default")
    p.add_argument("--strip_audio", action="store_true",
                   help="send the video with its audio track removed. Gives a vision-only LLM arm, "
                        "which separates model capability from modality access: V0 is vision-only "
                        "too, so without this arm a win for the LLM cannot be attributed to either.")
    p.add_argument("--mute_cache", type=Path, default=cfg.work("mute_clips"))
    p.add_argument("--data_root", type=Path, default=default_data_root())
    p.add_argument("--ids_file", type=Path, default=cfg.work("pilot_ids.txt"))
    p.add_argument("--sample_csv", type=Path, default=cfg.work("pilot_sample.csv"))
    p.add_argument("--out_dir", type=Path, default=cfg.work("nominations"))
    p.add_argument("--log_dir", type=Path, default=cfg.LOG_DIR)
    p.add_argument("--top_k", type=int, default=2)
    p.add_argument("--min_window_s", type=float, default=2.0)
    p.add_argument("--max_window_s", type=float, default=4.0)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None, help="process only the first N ids (prompt debugging)")
    p.add_argument("--project", default=os.environ.get("GOOGLE_CLOUD_PROJECT"))
    p.add_argument("--location", default=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"))
    p.add_argument("--dry_run", action="store_true", help="count tokens and print the projected cost, call nothing")
    return p.parse_args()


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("error") is None:
            done.add(record["id"])
    return done


def muted_clip(src: Path, cache_dir: Path) -> Path:
    """Copy of the clip with the audio stream dropped, cached on disk."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / src.name
    if dest.exists() and dest.stat().st_mtime >= src.stat().st_mtime:
        return dest
    tmp = dest.with_suffix(".tmp.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-an", "-c:v", "copy",
         "-movflags", "+faststart", str(tmp)],
        check=True, capture_output=True, timeout=900,
    )
    tmp.replace(dest)
    return dest


def build_request(clip_path: Path, duration: float, args: argparse.Namespace):
    video = types.Part(
        inline_data=types.Blob(data=clip_path.read_bytes(), mime_type="video/mp4"),
        video_metadata=types.VideoMetadata(fps=args.fps) if args.fps != 1.0 else None,
    )
    prompt = PROMPT.format(
        duration=duration, anchors=ANCHORS, top_k=args.top_k,
        min_window=args.min_window_s, max_window=args.max_window_s,
        modalities=MODALITIES_V if args.strip_audio else MODALITIES_AV,
        evidence_hint=EVIDENCE_V if args.strip_audio else EVIDENCE_AV,
    )
    return [types.Content(role="user", parts=[video, types.Part(text=prompt)])]


def main() -> int:
    args = parse_args()
    started = datetime.now(timezone.utc)
    if not args.project:
        _sys.exit("set GOOGLE_CLOUD_PROJECT (or pass --project); Vertex AI needs a project id")
    os.environ.update(GOOGLE_CLOUD_PROJECT=args.project, GOOGLE_CLOUD_LOCATION=args.location,
                      GOOGLE_GENAI_USE_VERTEXAI="true")
    client = genai.Client()

    ids = [line.strip() for line in args.ids_file.read_text().splitlines() if line.strip()]
    if args.limit:
        ids = ids[: args.limit]

    durations: Dict[str, float] = {}
    if args.sample_csv.exists():
        import pandas as pd
        frame = pd.read_csv(args.sample_csv, dtype={"id": str})
        durations = dict(zip(frame["id"], frame["duration_s"]))

    out_path = args.out_dir / f"{args.variant}.jsonl"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path)
    todo = [i for i in ids if i not in done]
    print(f"[{args.variant}] {len(ids)} clips, {len(done)} already done, {len(todo)} to run "
          f"({args.model}, {args.media_resolution}, fps={args.fps})", flush=True)

    config = types.GenerateContentConfig(
        temperature=args.temperature,
        media_resolution=getattr(types.MediaResolution, f"MEDIA_RESOLUTION_{args.media_resolution}"),
        response_mime_type="application/json",
        response_schema=RESPONSE_SCHEMA,
    )

    if args.dry_run:
        total_in = 0
        for clip_id in todo:
            contents = build_request(args.data_root / "face_origin" / f"{clip_id}.mp4",
                                     durations.get(clip_id, 60.0), args)
            counted = client.models.count_tokens(
                model=args.model, contents=contents,
                config=types.CountTokensConfig(generation_config=types.GenerationConfig(
                    media_resolution=config.media_resolution)))
            total_in += counted.total_tokens
        price_in, price_out = PRICES.get(args.model, (1.50, 7.50))
        est = total_in / 1e6 * price_in + len(todo) * 250 / 1e6 * price_out
        print(f"[{args.variant}] dry run: {total_in:,} input tokens over {len(todo)} clips "
              f"-> ~${est:.2f} standard, ~${est / 2:.2f} batch")
        return 0

    lock = threading.Lock()
    stats = {"ok": 0, "error": 0, "empty": 0, "flagged_noms": 0, "in_tokens": 0, "out_tokens": 0}
    errors: List[dict] = []

    def run_one(clip_id: str) -> None:
        clip_path = args.data_root / "face_origin" / f"{clip_id}.mp4"
        if args.strip_audio:
            clip_path = muted_clip(clip_path, args.mute_cache)
        duration = float(durations.get(clip_id, 60.0))
        record = {"variant": args.variant, "id": clip_id, "model": args.model,
                  "media_resolution": args.media_resolution, "fps": args.fps,
                  "audio": not args.strip_audio}
        try:
            response = client.models.generate_content(
                model=args.model, contents=build_request(clip_path, duration, args), config=config)
            payload = json.loads(response.text)
            noms = []
            for rank, nom in enumerate(payload.get("nominations", [])[: args.top_k], start=1):
                start, end = float(nom["start_s"]), float(nom["end_s"])
                # Flag rather than silently clamp: a nomination past the end of the clip is a
                # model failure the pilot needs to count, not a rounding error to paper over.
                flags = []
                if start < 0 or end > duration + 0.5:
                    flags.append("out_of_range")
                if end <= start:
                    flags.append("non_positive_span")
                if not flags and not (args.min_window_s - 0.5 <= end - start <= args.max_window_s + 0.5):
                    flags.append("window_violation")
                noms.append({
                    "rank": rank,
                    "start_s": round(start, 3),
                    "end_s": round(end, 3),
                    "score": int(nom["score"]),
                    "evidence": str(nom["evidence"]),
                    "flags": flags,
                })
            record.update(nominations=noms, error=None, usage={
                "input": response.usage_metadata.prompt_token_count,
                "output": response.usage_metadata.candidates_token_count,
            })
        except Exception as exc:  # network, quota, schema violation -- logged, never silent
            record.update(nominations=[], error=f"{type(exc).__name__}: {exc}"[:400])

        with lock:
            with out_path.open("a") as fh:
                fh.write(json.dumps(record) + "\n")
            if record["error"]:
                stats["error"] += 1
                errors.append({"id": clip_id, "error": record["error"]})
                print(f"[{args.variant}] {clip_id} FAILED: {record['error'][:120]}", flush=True)
            else:
                stats["ok"] += 1
                stats["empty"] += int(not record["nominations"])
                stats["flagged_noms"] += sum(1 for n in record["nominations"] if n["flags"])
                stats["in_tokens"] += record["usage"]["input"] or 0
                stats["out_tokens"] += record["usage"]["output"] or 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(run_one, todo))

    price_in, price_out = PRICES.get(args.model, (1.50, 7.50))
    cost = stats["in_tokens"] / 1e6 * price_in + stats["out_tokens"] / 1e6 * price_out
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    print(f"[{args.variant}] ok={stats['ok']} error={stats['error']} empty_lists={stats['empty']} "
          f"flagged_noms={stats['flagged_noms']} in={stats['in_tokens']:,} out={stats['out_tokens']:,} "
          f"cost=${cost:.2f} in {elapsed:.0f}s")

    args.log_dir.mkdir(parents=True, exist_ok=True)
    log = {
        "variant": args.variant, "model": args.model, "media_resolution": args.media_resolution,
        "fps": args.fps, "temperature": args.temperature, "top_k": args.top_k,
        "strip_audio": args.strip_audio,
        "window_s": [args.min_window_s, args.max_window_s],
        "started_utc": started.isoformat(), "elapsed_s": elapsed,
        "n_requested": len(todo), **stats, "cost_usd_standard": round(cost, 4),
        "cost_usd_batch": round(cost / 2, 4), "errors": errors,
        "prompt": PROMPT.format(
            duration=0.0, anchors=ANCHORS, top_k=args.top_k,
            min_window=args.min_window_s, max_window=args.max_window_s,
            modalities=MODALITIES_V if args.strip_audio else MODALITIES_AV,
            evidence_hint=EVIDENCE_V if args.strip_audio else EVIDENCE_AV,
        ),
    }
    log_path = args.log_dir / f"{args.variant}_{started.strftime('%Y%m%dT%H%M%SZ')}.json"
    log_path.write_text(json.dumps(log, indent=2))
    print(f"[{args.variant}] wrote {out_path} and {log_path}")
    return 1 if stats["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
