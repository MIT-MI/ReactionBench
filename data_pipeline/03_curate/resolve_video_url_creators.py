#!/usr/bin/env python3
"""Resolve the per-video creator keys in youtuber_id_map.json to real YouTube channels.

86 of the 115 creator keys -- effectively the whole ``game`` domain -- are
``watch?v=`` URLs rather than channels, and the game manifests carry no ``youtuber``
value. Any "N distinct creators" statistic computed on those keys counts videos, not
people, which is exactly the creator-concentration claim reviewers can check.

Fetches metadata only (no video download), caches per URL so reruns are cheap, and
never invents a channel: unresolved URLs stay unresolved and are listed in the report.

Outputs:
  $RVB_WORK_DIR/video_url_channel_map.json   url -> {channel, channel_id, uploader, title}
  $RVB_WORK_DIR/creator_resolution_report.json
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "common"))
import rvb_config as cfg  # noqa: E402  (loads .env, puts every stage dir on sys.path)

import argparse
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import yt_dlp

from build_pilot_frame import default_data_root


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", type=Path, default=default_data_root())
    p.add_argument("--meta_dir", type=Path, default=None, help="default: <data_root>/_meta")
    p.add_argument("--out_dir", type=Path, default=cfg.WORK_DIR)
    p.add_argument("--sleep_s", type=float, default=1.0, help="delay between requests")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def fetch(url: str) -> Optional[dict]:
    opts = {"quiet": True, "no_warnings": True, "skip_download": True, "extract_flat": False}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"[:300]}
    return {
        "channel": info.get("channel"),
        "channel_id": info.get("channel_id"),
        "channel_url": info.get("channel_url"),
        "uploader": info.get("uploader"),
        "uploader_id": info.get("uploader_id"),
        "title": info.get("title"),
        "duration": info.get("duration"),
    }


def main() -> int:
    args = parse_args()
    started = datetime.now(timezone.utc)
    meta_dir = args.meta_dir or (args.data_root / "_meta")

    raw = json.loads((meta_dir / "youtuber_id_map.json").read_text())
    url_keys = sorted(k for k in raw if "watch?v=" in k)
    if args.limit:
        url_keys = url_keys[: args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.out_dir / "video_url_channel_map.json"
    cache: Dict[str, dict] = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    todo = [u for u in url_keys if u not in cache or cache[u].get("error")]
    print(f"[resolve] {len(url_keys)} video-url creator keys, {len(todo)} to fetch", flush=True)

    for i, url in enumerate(todo, start=1):
        cache[url] = fetch(url) or {"error": "empty response"}
        status = cache[url].get("channel") or cache[url].get("error", "?")
        print(f"[resolve] {i}/{len(todo)} {url.split('=')[-1]} -> {str(status)[:70]}", flush=True)
        cache_path.write_text(json.dumps(cache, indent=2))
        time.sleep(args.sleep_s)

    resolved = {u: v for u, v in cache.items() if v.get("channel_id")}
    failed = {u: v.get("error") for u, v in cache.items() if not v.get("channel_id")}
    channels = Counter(v["channel_id"] for v in resolved.values())

    # How many clips sit behind each real channel, versus the per-video key count.
    clips_per_channel: Counter = Counter()
    for url, v in resolved.items():
        clips_per_channel[v["channel_id"]] += len(raw[url])

    report = {
        "generated_utc": started.isoformat(),
        "n_video_url_keys": len(url_keys),
        "n_resolved": len(resolved),
        "n_failed": len(failed),
        "n_distinct_real_channels": len(channels),
        "collapse_ratio": round(len(channels) / max(len(resolved), 1), 3),
        "channels": [
            {
                "channel_id": cid,
                "channel": next(v["channel"] for v in resolved.values() if v["channel_id"] == cid),
                "n_video_keys": n,
                "n_clips": clips_per_channel[cid],
            }
            for cid, n in channels.most_common()
        ],
        "failed": failed,
    }
    (args.out_dir / "creator_resolution_report.json").write_text(json.dumps(report, indent=2))

    print(f"\n[resolve] {len(resolved)}/{len(url_keys)} resolved into "
          f"{len(channels)} distinct channels ({len(failed)} failed)")
    for row in report["channels"][:10]:
        print(f"  {row['channel']}: {row['n_video_keys']} video keys, {row['n_clips']} clips")
    print(f"[resolve] wrote {cache_path} and {args.out_dir}/creator_resolution_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
