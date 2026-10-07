#!/usr/bin/env python3
"""Transcode the pilot clips into something a browser can actually decode.

Roughly 15% of `face_origin` is MPEG-4 Part 2, which no browser supports -- the video
element just shows nothing, with no error worth the name. This builds an H.264/AAC copy
with the moov atom moved to the front so seeking works from the first byte.

Clips that are already H.264 are remuxed rather than re-encoded (`-c:v copy`), so nothing
is degraded and the pass is fast. Idempotent: an up-to-date cache entry is left alone.

    python data_pipeline/05_nominate/pilot_review/prepare_web_clips.py
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "common"))
import rvb_config as cfg  # noqa: E402  (loads .env, puts every stage dir on sys.path)

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = cfg.PIPELINE_ROOT
from build_pilot_frame import default_data_root  # noqa: E402

BROWSER_VIDEO = {"h264", "vp8", "vp9", "av1"}
BROWSER_AUDIO = {"aac", "mp3", "opus", "vorbis"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", type=Path, default=default_data_root())
    p.add_argument("--sample_csv", type=Path, default=cfg.work("pilot_sample.csv"))
    p.add_argument("--cache_dir", type=Path, default=cfg.work("review/web_clips"))
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def transcode(src: Path, dest: Path, vcodec: Optional[str], acodec: Optional[str]) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp.mp4")
    video_args = ["-c:v", "copy"] if vcodec in BROWSER_VIDEO else [
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
    ]
    audio_args = ["-c:a", "copy"] if acodec == "aac" else ["-c:a", "aac", "-b:a", "128k"]
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
           *video_args, *audio_args, "-movflags", "+faststart", str(tmp)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=900)
    except subprocess.CalledProcessError as exc:
        # A stream copy can fail where a real encode succeeds; fall back before giving up.
        if "-c:v" in video_args and video_args[1] == "copy":
            return transcode(src, dest, "force_encode", acodec)
        tmp.unlink(missing_ok=True)
        return {"id": src.stem, "ok": False, "error": exc.stderr.decode()[-300:]}
    except subprocess.SubprocessError as exc:
        tmp.unlink(missing_ok=True)
        return {"id": src.stem, "ok": False, "error": str(exc)[:300]}
    tmp.replace(dest)
    return {"id": src.stem, "ok": True, "reencoded": vcodec not in BROWSER_VIDEO}


def main() -> int:
    args = parse_args()
    sample = pd.read_csv(args.sample_csv, dtype={"id": str})
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    todo = []
    for _, row in sample.iterrows():
        src = args.data_root / "face_origin" / f"{row['id']}.mp4"
        dest = args.cache_dir / f"{row['id']}.mp4"
        if dest.exists() and not args.force and dest.stat().st_mtime >= src.stat().st_mtime:
            continue
        todo.append((src, dest, row.get("vcodec"), row.get("acodec")))

    print(f"[web] {len(sample)} clips, {len(todo)} to build into {args.cache_dir}", flush=True)
    if not todo:
        return 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda t: transcode(*t), todo))

    failed = [r for r in results if not r["ok"]]
    reencoded = sum(1 for r in results if r.get("reencoded"))
    print(f"[web] built {len(results) - len(failed)}/{len(results)} "
          f"({reencoded} re-encoded, {len(results) - len(failed) - reencoded} remuxed)")
    for r in failed:
        print(f"[web] FAILED {r['id']}: {r['error']}", file=sys.stderr)
    (args.cache_dir / "_build_report.json").write_text(json.dumps(results, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
