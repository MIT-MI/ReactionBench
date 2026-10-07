#!/usr/bin/env python3
"""
Build a CSV manifest of candidate music reaction videos from a list of YouTube playlists.

Input CSV columns (UTF-8):
    playlist_url,channel,playlist_name

Output CSV columns:
    youtuber,playlist,video_link,video_title,label,music_category,duration,num_reactors,duration_seconds,is_music_reaction_video,selected

Key features:
- Uses yt-dlp (no API key) to iterate each playlist and fetch per-video metadata (title, url, duration).
- Uses Gemini (Vertex AI via google-genai) to:
    1) classify whether a video is a music reaction video and return a confidence.
       If confidence < threshold => is_music_reaction_video = "unknown".
       Else => "yes" or "no" depending on the class.
    2) For videos classified as music reactions with sufficient confidence, predict a music_category
       (e.g., classical, kpop, hip hop, rock, etc.).
- Heuristically estimates the number of reactors from the title/playlist/channel using Gemini.

Environment:
  pip install -U yt-dlp pandas google-genai python-dotenv
  export GOOGLE_CLOUD_PROJECT=...        # Required for Vertex Gemini calls
  export GOOGLE_CLOUD_LOCATION=us-central1  # Optional, defaults to us-central1
  gcloud auth application-default login   # Or set GOOGLE_APPLICATION_CREDENTIALS
  export GEMINI_MODEL=gemini-2.0-flash   # Optional override (default set in script)

Example:
  python build_music_reactions_manifest.py \
      --input all_playlists_music.csv \
      --output music_reactions_manifest.csv \
      --confidence-threshold 0.72

Notes:
- "label" is hard-coded to "Music reactions" as requested.
- "selected" defaults to False.
- The script does NOT download videos; it only collects metadata.
- If Gemini is unavailable or errors, the script falls back to conservative defaults and logs warnings.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "common"))
import rvb_config as cfg  # noqa: E402  (loads .env, puts every stage dir on sys.path)
import argparse
import csv
import dataclasses
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

# Lazy imports to allow running even if optional deps are missing until needed.
try:
    import yt_dlp
except Exception as e:  # pragma: no cover
    yt_dlp = None  # type: ignore

# Gemini client (Vertex AI)
try:
    from google import genai  # type: ignore
    from google.genai.types import GenerateContentConfig, Part  # type: ignore
except Exception:
    genai = None  # type: ignore
    GenerateContentConfig = None  # type: ignore
    Part = None  # type: ignore


# ----------------------------
# Gemini helpers
# ----------------------------

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
DEFAULT_CONFIDENCE_THRESHOLD = 0.72

MUSIC_CATEGORY_CANDIDATES = [
    "classical", "kpop", "jpop", "c-pop", "pop", "rock", "metal", "hip hop",
    "r&b", "jazz", "edm", "house", "techno", "trance", "indie", "folk",
    "country", "blues", "reggae", "latin", "afrobeats", "soundtrack/score",
    "punk", "gospel", "world", "other"
]


@dataclass
class GeminiContext:
    client: Any
    model: str
    config: Any


def _init_gemini() -> Optional[GeminiContext]:
    """Initialize a Vertex Gemini client if available and environment configured."""
    if genai is None or GenerateContentConfig is None or Part is None:
        return None

    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        sys.stderr.write("[Warn] Set GOOGLE_CLOUD_PROJECT to enable Vertex Gemini calls.\n")
        return None
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

    try:
        client = genai.Client(vertexai=True, project=project, location=location)
    except Exception as exc:
        sys.stderr.write(f"[Warn] Failed to initialize Vertex Gemini client: {exc}\n")
        return None

    config = GenerateContentConfig(
        temperature=0.0,
        top_p=1.0,
        max_output_tokens=1024,
        response_mime_type="application/json",
    )
    return GeminiContext(client=client, model=GEMINI_MODEL, config=config)


def gemini_json_call(ctx: Optional[GeminiContext], prompt: str, max_retries: int = 3, sleep_s: float = 2.0) -> Optional[Dict[str, Any]]:
    """Call Gemini, requesting a STRICT JSON object. Returns parsed dict or None."""
    if ctx is None or Part is None:
        return None

    sys_prompt = (
        "You are a precise JSON API. Respond with a single JSON object only. "
        "Do not include any extra commentary, code fences, or text outside the JSON."
    )
    user_prompt = prompt + "\n\nReturn ONLY valid JSON."

    for attempt in range(1, max_retries + 1):
        try:
            parts = [
                Part.from_text(text=sys_prompt),
                Part.from_text(text=user_prompt),
            ]
            resp = ctx.client.models.generate_content(
                model=ctx.model,
                contents=parts,
                config=ctx.config,
            )
            text = getattr(resp, "text", None)
            if not text and getattr(resp, "candidates", None):
                first = resp.candidates[0]
                if getattr(first, "content", None) and getattr(first.content, "parts", None):
                    texts = [
                        getattr(part, "text", "")
                        for part in first.content.parts
                        if getattr(part, "text", "")
                    ]
                    text = "\n".join(t for t in texts if t)
            if not text:
                raise ValueError("Empty response from Gemini")
            # Try to locate a JSON object in the text
            m = re.search(r"\{[\s\S]*\}$", text.strip())
            json_str = m.group(0) if m else text.strip()
            return json.loads(json_str)
        except Exception as e:
            if attempt >= max_retries:
                sys.stderr.write(f"[Gemini] Failed after {attempt} attempts: {e}\n")
                return None
            time.sleep(sleep_s * attempt)
    return None


def classify_is_music_reaction(ctx: Optional[GeminiContext], *, title: str, playlist: str, channel: str) -> Tuple[str, float]:
    """Return (label, confidence) where label in {"yes","no","unknown"}."""
    prompt = f"""
Task: Determine if a YouTube video is a MUSIC REACTION video.
Definition: A music reaction video features creators listening to a SONG/ALBUM or MUSIC VIDEO and reacting, often titled like "First Time Reaction", "Album Reaction", "Track-By-Track", "Review/Reaction", or "Listening Party". Videos that are pure news, tutorials, instrument playthroughs, lyrics-only, or general commentary without reacting to music DO NOT qualify.

Input fields:
- title: {title}
- playlist: {playlist}
- channel: {channel}

Respond in JSON with fields:
  {{
    "is_music_reaction": "yes" | "no",
    "confidence": a float in [0,1]
  }}
"""
    out = gemini_json_call(ctx, prompt)
    if not out:
        return ("unknown", 0.0)
    label = (out.get("is_music_reaction") or "").strip().lower()
    conf = float(out.get("confidence") or 0.0)
    if label not in {"yes", "no"}:
        return ("unknown", conf)
    return (label, conf)


def predict_music_category(ctx: Optional[GeminiContext], *, title: str, playlist: str, channel: str) -> str:
    """Return one best-guess category string from MUSIC_CATEGORY_CANDIDATES."""
    candidates = ", ".join(MUSIC_CATEGORY_CANDIDATES)
    prompt = f"""
Task: For a confirmed MUSIC REACTION video, guess the music category (genre or broad bucket) from this fixed set:
[{candidates}]

Choose the closest label. If completely unclear, return "other".

Input:
- title: {title}
- playlist: {playlist}
- channel: {channel}

Respond as JSON: {{"music_category": "<one_of_the_list>"}}
"""
    out = gemini_json_call(ctx, prompt)
    if not out:
        return "other"
    raw = (out.get("music_category") or "other").strip().lower()
    # Simple normalization
    norm = {
        "k-pop": "kpop", "k pop": "kpop", "korean pop": "kpop",
        "j-pop": "jpop", "j pop": "jpop", "japanese pop": "jpop",
        "c pop": "c-pop", "c-pop": "c-pop", "chinese pop": "c-pop",
        "hip-hop": "hip hop",
        "soundtrack": "soundtrack/score", "score": "soundtrack/score"
    }.get(raw, raw)
    return norm if norm in MUSIC_CATEGORY_CANDIDATES else "other"


def estimate_num_reactors(ctx: Optional[GeminiContext], *, title: str, playlist: str, channel: str) -> Optional[int]:
    """Heuristically estimate reactors count from textual hints using Gemini.
    Returns an integer >=1 if clear; else None.
    """
    prompt = f"""
From the given fields (title, playlist, channel), infer how many on-screen reactors likely appear.
Rules:
- If the text clearly indicates duo/two people (e.g., "Twins", "Two Guys", "Double Take"), return 2.
- If it explicitly names a trio or more, use that number.
- If a single host is implied (e.g., channel is a solo reactor's handle), return 1.
- If unclear, return null.

Input:
- title: {title}
- playlist: {playlist}
- channel: {channel}

Respond JSON only: {{"num_reactors": <int or null>}}
"""
    out = gemini_json_call(ctx, prompt)
    if not out:
        return None
    val = out.get("num_reactors")
    try:
        if val is None:
            return None
        iv = int(val)
        return iv if iv >= 1 else None
    except Exception:
        return None


# ----------------------------
# yt-dlp helpers
# ----------------------------

class QuietLogger:
    def debug(self, msg):
        pass
    def warning(self, msg):
        sys.stderr.write(f"[yt-dlp] {msg}\n")
    def error(self, msg):
        sys.stderr.write(f"[yt-dlp ERROR] {msg}\n")


def iter_playlist_videos(playlist_url: str) -> Iterable[Dict[str, Any]]:
    """Yield minimal metadata dicts for each video in a playlist using yt-dlp."""
    if yt_dlp is None:
        raise RuntimeError("yt-dlp is not installed. pip install yt-dlp")

    ydl_opts = {
        "logger": QuietLogger(),
        "quiet": True,
        "nocheckcertificate": True,
        "skip_download": True,
        "extract_flat": False,  # get full info per video to include duration
        "writesubtitles": False,
        "writeinfojson": False,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(playlist_url, download=False)
        except Exception as e:
            sys.stderr.write(f"[yt-dlp] Failed to extract playlist {playlist_url}: {e}\n")
            return

        # The playlist extraction can yield 'entries' which may be generators.
        entries = info.get("entries") or []
        for entry in entries:
            if not entry:
                continue
            # For some playlists with extract_flat, nested entries may be dicts without duration.
            # If duration missing, re-extract by video URL to get full metadata.
            if entry.get("duration") is None and entry.get("url"):
                try:
                    full = ydl.extract_info(entry["url"], download=False)
                    entry = full or entry
                except Exception:
                    pass
            yield entry


def seconds_to_hms(seconds: Optional[int]) -> str:
    try:
        if seconds is None:
            return ""
        s = int(seconds)
        h = s // 3600
        m = (s % 3600) // 60
        sec = s % 60
        return f"{h:01d}:{m:02d}:{sec:02d}" if h > 0 else f"{m:01d}:{sec:02d}"
    except Exception:
        return ""


# ----------------------------
# Main processing
# ----------------------------

@dataclass
class OutputRow:
    youtuber: str
    playlist: str
    video_link: str
    video_title: str
    label: str
    music_category: str
    duration: str
    num_reactors: Optional[int]
    duration_seconds: Optional[int]
    is_music_reaction_video: str  # yes|no|unknown
    selected: bool


def process(input_csv: str, output_csv: str, confidence_threshold: float) -> None:
    gemini_ctx = _init_gemini()
    if gemini_ctx is None:
        sys.stderr.write("[Warn] Gemini not initialized. Install google-genai and configure Vertex credentials.\n")

    df = pd.read_csv(input_csv)
    required_cols = {"playlist_url", "channel", "playlist_name"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {sorted(missing)}")

    out_rows: List[OutputRow] = []

    for idx, row in df.iterrows():
        playlist_url = str(row["playlist_url"]).strip()
        channel = str(row["channel"]).strip()
        playlist_name = str(row["playlist_name"]).strip()
        if not playlist_url:
            continue

        sys.stderr.write(f"\n[Playlist] {playlist_name} | {playlist_url}\n")

        for vid in iter_playlist_videos(playlist_url):
            title = (vid.get("title") or "").strip()
            url = vid.get("webpage_url") or vid.get("url") or ""
            dur_s = vid.get("duration")  # seconds
            dur_str = seconds_to_hms(dur_s)

            # Gemini classification
            is_label = "unknown"
            category = ""
            num_reactors = None

            lab, conf = classify_is_music_reaction(gemini_ctx, title=title, playlist=playlist_name, channel=channel)
            if lab == "unknown" or conf < confidence_threshold:
                is_label = "unknown"
            else:
                is_label = lab

            if is_label == "yes":
                category = predict_music_category(gemini_ctx, title=title, playlist=playlist_name, channel=channel)
            else:
                category = ""

            num_reactors = estimate_num_reactors(gemini_ctx, title=title, playlist=playlist_name, channel=channel)

            out_rows.append(OutputRow(
                youtuber=channel,
                playlist=playlist_name,
                video_link=url,
                video_title=title,
                label="Music reactions",
                music_category=category,
                duration=dur_str,
                num_reactors=num_reactors,
                duration_seconds=int(dur_s) if dur_s is not None else None,
                is_music_reaction_video=is_label,
                selected=False,
            ))

    # Write CSV
    # Ensure stable column order
    fieldnames = [
        "youtuber", "playlist", "video_link", "video_title", "label",
        "music_category", "duration", "num_reactors", "duration_seconds",
        "is_music_reaction_video", "selected",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in out_rows:
            writer.writerow(dataclasses.asdict(r))

    sys.stderr.write(f"\n[Done] Wrote {len(out_rows)} rows to {output_csv}\n")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build music reactions manifest from YouTube playlists.")
    p.add_argument("--input", required=True, help="Path to input CSV with playlist_url,channel,playlist_name")
    p.add_argument("--output", required=True, help="Path to write the output CSV")
    p.add_argument("--confidence-threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD,
                   help=f"Confidence threshold for 'unknown' classification (default {DEFAULT_CONFIDENCE_THRESHOLD})")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    process(args.input, args.output, args.confidence_threshold)
