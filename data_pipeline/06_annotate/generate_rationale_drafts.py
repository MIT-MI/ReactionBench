#!/usr/bin/env python3
"""Generate rationale DRAFTS for the Stage C set with an open Qwen-VL model on Tinker.

Drafts are seeds that humans correct — the point of using an open model (not Gemini)
is to keep the drafting model off the headline evaluation and avoid circularity. Each draft is
one sentence explaining WHY the reviewer reacted, from frames sampled across the stimulus event
window (recut overlay for stim_source=recut). Vision is real: Qwen3.5 reads the frames.

Uses the TK_API_KEY_{1..5} keys round-robin for throughput, disable-thinking renderer for clean
output, and tracks prefill/sample tokens to report cost. Resumable via the output JSONL.

  export TINKER_API_KEY=... (or pass --keys)
  python data_pipeline/06_annotate/generate_rationale_drafts.py --limit 3    # smoke
  python data_pipeline/06_annotate/generate_rationale_drafts.py
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "common"))
import rvb_config as cfg  # noqa: E402  (loads .env, puts every stage dir on sys.path)

import argparse
import asyncio
import base64
import csv
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import tinker
from tinker_cookbook import renderers
from tinker_cookbook.image_processing_utils import get_image_processor

# Qwen3.5-9B list price, USD per 1M tokens (prefill = input incl. image tokens; sample = output).
PRICES = {"Qwen/Qwen3.5-9B": (0.66, 1.995), "Qwen/Qwen3.6-27B": (1.32, 3.99)}

PROMPT = ("These frames are from a short video clip that a person was watching, and they had a "
          "strong reaction to it. In ONE concrete sentence, explain WHY they reacted — name the "
          "specific thing that happens in the video that caused it. State it factually, no hedging, "
          "no preamble.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = cfg.work("stimulus_ranking")
    p.add_argument("--set_csv", type=Path, default=d / "rationale_set.csv")
    p.add_argument("--out", type=Path, default=d / "rationale_drafts.jsonl")
    p.add_argument("--stim_dir", type=Path, default=cfg.DATA_ROOT / "stimuli")
    p.add_argument("--recut_dir", type=Path, default=cfg.RECUT_ROOT / "stimuli")
    p.add_argument("--model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--renderer", default="qwen3_5_disable_thinking")
    p.add_argument("--frames", type=int, default=6)
    p.add_argument("--max_tokens", type=int, default=90)
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--concurrency", type=int, default=5)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def load_keys() -> list[str]:
    keys = [os.environ[k] for k in (f"TK_API_KEY_{i}" for i in range(1, 6)) if os.environ.get(k)]
    return keys or ([os.environ["TINKER_API_KEY"]] if os.environ.get("TINKER_API_KEY") else [])


def load_done(path: Path) -> set:
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
            if not r.get("error"):
                out.add(r["candidate_id"])
        except json.JSONDecodeError:
            pass
    return out


def frame_uris(row: dict, stim_dir: Path, recut_dir: Path, n: int) -> list[str]:
    src = (recut_dir if row.get("stim_source") == "recut" else stim_dir) / f'{row["video_id"]}.mp4'
    s, e = float(row["stim_s"]), float(row["stim_e"])
    a, b = max(0.0, s - 0.5), e + 0.5
    times = [a + (b - a) * i / max(1, n - 1) for i in range(n)]
    uris = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, t in enumerate(times):
            jpg = f"{tmp}/{i}.jpg"
            r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", str(src),
                "-frames:v", "1", "-vf", "scale=448:-2", jpg], capture_output=True)
            if r.returncode == 0 and os.path.exists(jpg):
                uris.append("data:image/jpeg;base64," + base64.b64encode(Path(jpg).read_bytes()).decode())
    return uris


async def main_async(args) -> int:
    started = datetime.now(timezone.utc)
    keys = load_keys()
    if not keys:
        print("no TK_API_KEY_{1..5} / TINKER_API_KEY in env"); return 1

    # one sampling client per key (key read from env at construction); one shared renderer.
    clients = []
    tokenizer = image_proc = None
    for k in keys:
        os.environ["TINKER_API_KEY"] = k
        sc = tinker.ServiceClient()
        scl = sc.create_sampling_client(base_model=args.model)
        if tokenizer is None:
            tokenizer, image_proc = scl.get_tokenizer(), get_image_processor(args.model)
        clients.append(scl)
    renderer = renderers.get_renderer(args.renderer, tokenizer, image_proc, model_name=args.model)
    stop = renderer.get_stop_sequences()
    print(f"[drafts] model={args.model} keys={len(clients)} frames={args.frames}", flush=True)

    rows = list(csv.DictReader(open(args.set_csv)))
    done = load_done(args.out)
    todo = [r for r in rows if r["candidate_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[drafts] {len(rows)} in set, {len(done)} done, {len(todo)} to run", flush=True)

    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    stats = {"ok": 0, "error": 0, "prefill": 0, "sample": 0}
    loop = asyncio.get_event_loop()

    async def do(i: int, row: dict) -> None:
        async with sem:
            rec = {"candidate_id": row["candidate_id"], "video_id": row["video_id"]}
            try:
                uris = await loop.run_in_executor(None, frame_uris, row, args.stim_dir, args.recut_dir, args.frames)
                if not uris:
                    raise RuntimeError("no frames extracted")
                content = [{"type": "text", "text": PROMPT}] + [{"type": "image", "image": u} for u in uris]
                mi = renderer.build_generation_prompt([{"role": "user", "content": content}])
                res = await clients[i % len(clients)].sample_async(
                    mi, num_samples=1,
                    sampling_params=tinker.SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens, stop=stop))
                toks = res.sequences[0].tokens
                parsed, _ = renderer.parse_response(toks)
                draft = renderers.get_text_content(parsed).strip() if hasattr(renderers, "get_text_content") else parsed["content"]
                if isinstance(draft, list):
                    draft = " ".join(p.get("text", "") for p in draft if isinstance(p, dict)).strip()
                rec.update(draft=draft, prefill=int(getattr(mi, "length", 0)), sample=len(toks), error=None)
                async with lock:
                    stats["ok"] += 1; stats["prefill"] += rec["prefill"]; stats["sample"] += rec["sample"]
            except Exception as ex:  # noqa: BLE001
                rec.update(error=f"{type(ex).__name__}: {ex}")
                async with lock:
                    stats["error"] += 1
            async with lock:
                with open(args.out, "a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (stats["ok"] + stats["error"]) % 20 == 0:
                print(f"  {stats['ok']} ok / {stats['error']} err", flush=True)

    await asyncio.gather(*[do(i, r) for i, r in enumerate(todo)])

    pin, pout = PRICES.get(args.model, (0.66, 1.995))
    cost = stats["prefill"] / 1e6 * pin + stats["sample"] / 1e6 * pout
    print(json.dumps({"ok": stats["ok"], "error": stats["error"], "prefill_tokens": stats["prefill"],
        "sample_tokens": stats["sample"], "cost_usd": round(cost, 3),
        "seconds": round((datetime.now(timezone.utc) - started).total_seconds(), 1)}, indent=2))
    return 0


def main() -> int:
    return asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
