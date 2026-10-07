from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "common"))
import rvb_config as cfg  # noqa: E402  (loads .env, puts every stage dir on sys.path)
import sys
import time
from typing import Dict, Iterable, List, Optional, Tuple
import pandas as pd
from yt_dlp import YoutubeDL
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

try:
    from google import genai as genai
    from google.genai.types import GenerateContentConfig
except Exception:
    genai = None

# Playlist registry (manually curated; columns: playlist_url, playlist_name, refined_category,
# base_category, is_reaction) and output catalog. Override with env vars; no CLI flags.
INPUT_CSV = os.environ.get("RVB_PLAYLISTS_CSV", "all_playlists.csv")


def load_input_items_from_csv(
    csv_path: str,
    include_non_reactions: bool = False,
) -> List[Dict]:
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise RuntimeError(f"Error loading CSV: {csv_path} -> {e}")

    df.columns = [str(c).strip() for c in df.columns]

    df = df[df.get("playlist_url").notna()]

    if "is_reaction" in df.columns:
        if include_non_reactions:
            pass
        else:
            df = df[df["is_reaction"].astype(str).str.upper() == "YES"]

    items: List[Dict] = []
    for _, row in df.iterrows():
        url = str(row.get("playlist_url", "")).strip()
        if not url:
            continue
        label = str(row.get("refined_category") or "").strip()
        if not label:
            label = str(row.get("base_category") or "").strip()
        playlist_name = str(row.get("playlist_name") or "").strip()

        items.append({
            "type": "playlist",
            "url": url,
            "label": label,
            "playlist_name": playlist_name,
        })

    return items


INPUT_ITEMS: List[Dict] = load_input_items_from_csv(INPUT_CSV, include_non_reactions=False)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV = os.environ.get("RVB_CATALOG", os.path.join(SCRIPT_DIR, "youtube_catalog-new.csv"))

YDL_OPTS_BASE = {
    "quiet": True,
    "no_warnings": True,
    "skip_download": True,
}

def sec_to_hhmmss(sec: Optional[int]) -> str:
    if sec is None or not isinstance(sec, (int, float)) or sec < 0:
        return ""
    s = int(sec)
    h = s // 3600
    m = (s % 3600) // 60
    ss = s % 60
    return f"{h:02d}:{m:02d}:{ss:02d}"

def pick_channel_identifier(info: Dict) -> Tuple[str, str]:
    """
    Return (youtuber_link_or_id, source) where source is only used for internal debugging
    Prefer channel_url / uploader_url, then channel_id / uploader_id
    """
    for k in ("channel_url", "uploader_url"):
        v = info.get(k)
        if v:
            return v, k
    for k in ("channel_id", "uploader_id"):
        v = info.get(k)
        if v:
            return v, k
    # Last fallback use 'uploader'
    v = info.get("uploader") or ""
    return v, "uploader"

def deep_extract_video(ydl: YoutubeDL, url: str) -> Optional[Dict]:
    try:
        return ydl.extract_info(url, download=False)
    except Exception as e:
        print(f"[WARN] Video parsing failed: {url}\n  -> {e}")
        return None


def deep_extract_video_standalone(url: str) -> Optional[Dict]:
    """Thread-safe way: create your own YoutubeDL instance each time."""
    try:
        with YoutubeDL(dict(YDL_OPTS_BASE, extract_flat=False)) as ydl:
            return ydl.extract_info(url, download=False)
    except Exception as e:
        print(f"[WARN] Video parsing failed: {url}\n  -> {e}")
        return None

def expand_playlist(ydl: YoutubeDL, playlist_url: str) -> Tuple[str, List[str]]:
    """
    Return (playlist_title, video_urls_in_playlist)
    First flatten the url of entries, then deep extract each video
    """
    opts = dict(YDL_OPTS_BASE)
    opts["extract_flat"] = True
    with YoutubeDL(opts) as ydl_flat:
        try:
            pl = ydl_flat.extract_info(playlist_url, download=False)
        except Exception as e:
            print(f"[WARN] Playlist parsing failed: {playlist_url}\n  -> {e}")
            return "", []
    playlist_title = pl.get("title") or ""
    entries = pl.get("entries") or []
    urls = []
    for ent in entries:
        # ent may be dict(url=...), or may be directly video information (flat)
        u = ent.get("url") if isinstance(ent, dict) else None
        if not u:
            # In some cases, flat mode will put 'id' + 'url' combination, here do a fallback
            vid_id = ent.get("id") if isinstance(ent, dict) else None
            if vid_id:
                u = f"https://www.youtube.com/watch?v={vid_id}"
        if u:
            urls.append(u)
    return playlist_title, urls

def _ensure_header(csv_path: str) -> None:
    """If the file does not exist, create and write the header (empty content)."""
    if not os.path.exists(csv_path):
        columns = [
            "youtuber", "playlist", "video_link", "video_title", "label", "duration"
        ]
        pd.DataFrame(columns=columns).to_csv(csv_path, index=False, encoding="utf-8")


def _load_existing_results(csv_path: str) -> Tuple[set, bool]:
    """
    Return (seen_video_urls, has_header)
    - If the CSV exists, read the existing video_link for deduplication and resume.
    - has_header is used to determine whether to write the header.
    """
    seen: set = set()
    has_header = False
    if os.path.exists(csv_path):
        try:
            df_exist = pd.read_csv(csv_path)
            if "video_link" in df_exist.columns:
                seen = set(df_exist["video_link"].astype(str).tolist())
                has_header = True
        except Exception:
            # If the CSV exists but cannot be read, treat it as a new file without a header
            seen = set()
            has_header = False
    return seen, has_header


def _append_rows(csv_path: str, rows: List[List[str]], has_header: bool) -> bool:
    """
    Append rows to csv_path.
    - has_header: If False, write the header first.
    Return value: whether the header is written after writing (used to update the status).
    """
    if not rows:
        return has_header
    columns = [
        "youtuber", "playlist", "video_link", "video_title", "label", "duration"
    ]
    mode = "a" if os.path.exists(csv_path) else "w"
    df_batch = pd.DataFrame(rows, columns=columns)
    df_batch.to_csv(csv_path, mode=mode, index=False, header=(not has_header), encoding="utf-8")
    return True


def process_inputs(items: List[Dict]) -> pd.DataFrame:
    rows = []
    # Ensure the CSV at least contains the header, avoid the confusion of running for a long time without a file
    _ensure_header(OUTPUT_CSV)
    seen_video_urls, has_header = _load_existing_results(OUTPUT_CSV)

    # Load existing playlists from OUTPUT_CSV for deduplication
    seen_playlists = set()
    if os.path.exists(OUTPUT_CSV):
        try:
            df_exist = pd.read_csv(OUTPUT_CSV)
            if "playlist" in df_exist.columns:
                seen_playlists = set(df_exist["playlist"].astype(str).tolist())
        except Exception:
            seen_playlists = set()

    gemini_client = build_gemini_client()
    with YoutubeDL(dict(YDL_OPTS_BASE, extract_flat=False)) as ydl:
        # First process non-playlist (if any), then process playlist sequentially with tqdm progress
        other_items = [it for it in items if it.get("type") != "playlist"]
        playlist_items = [it for it in items if it.get("type") == "playlist"]

        # Process non-playlist (usually not in CSV data)
        for item in other_items:
            item_type = item["type"]
            label = item.get("label", "")

            if item_type == "video":
                url = item["url"]
                if url in seen_video_urls:
                    continue
                info = deep_extract_video(ydl, url)
                if not info or info.get("_type") == "playlist":
                    # Some "video links" actually are mixed pages, further process
                    info = deep_extract_video(ydl, info.get("webpage_url")) if info else None
                if not info or info.get("_type") == "playlist":
                    print(f"[WARN] Skip unparseable video: {url}")
                    continue
                # Reaction video judgment
                verdict = is_reaction_video(info, client=gemini_client)
                if verdict is not True:
                    continue

                youtuber, _ = pick_channel_identifier(info)
                playlist_col = ""  # Single video has no playlist
                video_link = info.get("webpage_url") or url
                title = info.get("title") or ""
                duration = sec_to_hhmmss(info.get("duration"))

                rows.append([youtuber, playlist_col, video_link, title, label, duration])
                seen_video_urls.add(video_link)

            elif item_type == "videolist":
                for url in item["urls"]:
                    if url in seen_video_urls:
                        continue
                    info = deep_extract_video(ydl, url)
                    if not info or info.get("_type") == "playlist":
                        info = deep_extract_video(ydl, info.get("webpage_url")) if info else None
                    if not info or info.get("_type") == "playlist":
                        print(f"[WARN] Skip unparseable video: {url}")
                        continue
                    # Reaction video judgment
                    verdict = is_reaction_video(info, client=gemini_client)
                    if verdict is not True:
                        continue

                    youtuber, _ = pick_channel_identifier(info)
                    video_link = info.get("webpage_url") or url
                    title = info.get("title") or ""
                    duration = sec_to_hhmmss(info.get("duration"))

                    rows.append([youtuber, "", video_link, title, label, duration])
                    seen_video_urls.add(video_link)

        if rows:
            has_header = _append_rows(OUTPUT_CSV, rows, has_header)
            rows = []

        # Process playlist: sequential + tqdm progress
        for item in tqdm(playlist_items, desc="Playlists", unit="pl"):
            item_type = item["type"]
            label = item.get("label", "")

            if item_type == "playlist":
                pl_url = item["url"]
                pl_title, video_urls = expand_playlist(ydl, pl_url)
                if not video_urls:
                    print(f"[WARN] Playlist is empty or failed to parse: {pl_url}")
                    continue
                # Deep extract each video + parallel processing
                playlist_col_default = item.get("playlist_name") or pl_title or pl_url
                # Check if playlist name or URL is already in seen_playlists
                if playlist_col_default in seen_playlists or pl_url in seen_playlists:
                    # print(f"[WARN] Playlist '{playlist_col_default}' already exists in output CSV. Skipping.")
                    continue

                def handle_video_url(vurl: str) -> Optional[List[str]]:
                    if vurl in seen_video_urls:
                        return None
                    info = deep_extract_video_standalone(vurl)
                    if not info or info.get("_type") == "playlist":
                        return None
                    # Reaction video judgment
                    verdict = is_reaction_video(info, client=gemini_client)
                    if verdict is not True:
                        return None
                    youtuber, _ = pick_channel_identifier(info)
                    video_link = info.get("webpage_url") or vurl
                    title = info.get("title") or ""
                    duration = sec_to_hhmmss(info.get("duration"))
                    return [youtuber, playlist_col_default, video_link, title, label, duration]

                max_workers = min(8, max(1, os.cpu_count() or 4))
                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    fut_map = {ex.submit(handle_video_url, u): u for u in video_urls}
                    for fut in as_completed(fut_map):
                        row = None
                        try:
                            row = fut.result()
                        except Exception as e:
                            print(f"[WARN] Processing video failed: {fut_map[fut]} -> {e}")
                        if row:
                            # Deduplicate
                            video_link = row[2]
                            if video_link not in seen_video_urls:
                                rows.append(row)
                                seen_video_urls.add(video_link)
                # Write each playlist to disk after completion
                if rows:
                    has_header = _append_rows(OUTPUT_CSV, rows, has_header)
                    rows = []
            else:
                print(f"[WARN] Unknown type, skipped: {item_type}")

    # If rows still have remaining (theoretically should not happen), write them to disk as a fallback
    if rows:
        _append_rows(OUTPUT_CSV, rows, has_header)
        rows = []

    # Return the final complete DataFrame (including existing + new)
    try:
        df_all = pd.read_csv(OUTPUT_CSV)
        df_all.sort_values(by=["label", "youtuber", "playlist", "video_title"], inplace=True, ignore_index=True)
        return df_all
    except Exception:
        # Fallback: if reading fails, only return an empty table
        return pd.DataFrame(columns=[
            "youtuber", "playlist", "video_link", "video_title", "label", "duration"
        ])

def main():
    df = process_inputs(INPUT_ITEMS)
    print(f"✅ Incremental generation/update CSV: {OUTPUT_CSV}")
    print(f"Current total {len(df)} videos")


# ==================== Reaction judgment (Gemini)====================

REACTION_KEYWORDS = [
    r"\breaction\b",
    r"\breacts?\b",
    r"\breacting\b",
    r"first time hearing",
    r"reviewing",
    r"watching",
    r"responds? to",
]


def quick_heuristic_is_reaction(title: str, description: str) -> Optional[bool]:
    text = f"{title}\n{description}".lower()
    # Simple heuristic: if the title or description clearly contains keywords, determine as reaction
    for pat in REACTION_KEYWORDS:
        if re.search(pat, text):
            return True

    return None


def build_gemini_client():
    # Compatible with two environment variable names
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key or genai is None:
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception:
        return None


def is_reaction_video(info: Dict, client=None) -> Optional[bool]:
    """
    Use Gemini to determine if it is a reaction video.
    - Return True: reaction
    - Return False: non-reaction
    - Return None: determination failed (network/model/response exception etc.) => need upstream "skip"
    """
    title = str(info.get("title") or "")
    desc = str(info.get("description") or "")
    channel = str(info.get("uploader") or info.get("channel") or "")

    client = client or build_gemini_client()
    if client is None:
        return None

    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    prompt = (
        "You are a strict classifier. Determine if this YouTube video is a 'reaction video' "
        "(i.e., the creator watches/listens to other media and reacts to it on camera).\n"
        "Return ONLY 'YES' or 'NO'. No extra words.\n\n"
        f"Channel: {channel}\n"
        f"Title: {title}\n"
        f"Description: {desc[:2000]}\n"
    )
    try:
        res = client.models.generate_content(
            model=model,
            contents=prompt,
            config=GenerateContentConfig(temperature=0.0),
        )
        out = (res.text or "").strip().lower()
        if out == "yes":
            return True
        if out == "no":
            return False
        # Any non-strict yes/no output is considered uncertain => skip
        return None
    except Exception as e:
        print(f"[WARN] Gemini determination failed (will skip): {e}")
        return None

if __name__ == "__main__":
    main()
