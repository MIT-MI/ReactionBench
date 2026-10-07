import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "common"))
import rvb_config as cfg  # noqa: E402  (loads .env, puts every stage dir on sys.path)
import argparse
import csv
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

from yt_dlp import YoutubeDL


DEFAULT_OUTPUT_DIR = os.path.join(os.environ.get("RVB_RAW_ROOT", "data/human_reaction_data"), "patreon", "masked")
DEFAULT_MANIFEST = "manifests/patreon.csv"
DEFAULT_COOKIES_FILE = os.environ.get("YTDLP_COOKIES")  # never commit the cookies file
DEFAULT_FORMAT = (
    "bestvideo[height<=1440][vcodec^=avc1]+"
    "bestaudio[acodec^=mp4a]/best[ext=mp4]/best"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
LOGGER = logging.getLogger("patreon_downloader")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Patreon YouTube videos listed in a manifest."
    )
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        help="Path to patreon manifest CSV (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store downloaded videos (default: %(default)s).",
    )
    parser.add_argument(
        "--cookies",
        default=DEFAULT_COOKIES_FILE,
        help="Path to cookies file for yt-dlp (optional).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume partially downloaded files if present.",
    )
    parser.add_argument(
        "--max-count",
        type=int,
        help="Download at most this many entries from the manifest.",
    )
    return parser.parse_args()


def load_manifest(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = [row for row in reader]
    return rows


def sanitize_video_id(raw_id: str) -> str:
    cleaned = (raw_id or "").strip()
    if not cleaned:
        raise ValueError("Manifest row missing video id.")
    return cleaned


def sanitize_url(raw_url: str) -> str:
    url = (raw_url or "").strip()
    if not url:
        raise ValueError("Manifest row missing youtube link.")
    return url


def make_ydl_opts(
    target_path: str,
    cookies_path: Optional[str],
    resume: bool,
) -> dict:
    output_template = os.path.splitext(target_path)[0] + ".%(ext)s"
    opts: dict = {
        "format": DEFAULT_FORMAT,
        "merge_output_format": "mp4",
        "outtmpl": output_template,
        "noplaylist": True,
        "logger": LOGGER,
        "quiet": False,
        "no_warnings": True,
        "overwrites": False,
        "continuedl": resume,
    }
    if cookies_path:
        opts["cookiefile"] = cookies_path
    return opts


def download_video(youtube_url: str, target_path: str, cookies: Optional[str], resume: bool) -> bool:
    if os.path.exists(target_path):
        LOGGER.info("Skipping existing file: %s", target_path)
        return True

    ydl_opts = make_ydl_opts(target_path, cookies, resume)

    try:
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Failed to download %s -> %s: %s", youtube_url, target_path, exc)
        return False

    if os.path.exists(target_path):
        LOGGER.info("Downloaded %s", target_path)
        return True

    LOGGER.error("Expected output does not exist after download: %s", target_path)
    return False


def process_manifest(
    manifest_rows: List[Dict[str, str]],
    output_dir: str,
    cookies: Optional[str],
    resume: bool,
    max_count: Optional[int],
) -> Tuple[int, int]:
    successes = 0
    failures = 0
    for index, row in enumerate(manifest_rows):
        if max_count is not None and index >= max_count:
            break
        video_id = None
        youtube_url = None
        try:
            video_id = sanitize_video_id(row.get("video", ""))
            youtube_url = sanitize_url(row.get("youtube link", ""))
        except ValueError as exc:
            failures += 1
            LOGGER.warning("Skipping row %d due to missing data: %s", index, exc)
            continue

        target_filename = f"{video_id}.mp4"
        target_path = os.path.join(output_dir, target_filename)
        LOGGER.info("Processing %s -> %s", youtube_url, target_filename)

        if download_video(youtube_url, target_path, cookies, resume):
            successes += 1
        else:
            failures += 1
    return successes, failures


def main() -> None:
    args = parse_args()
    manifest_path = os.path.abspath(args.manifest)
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.exists(manifest_path):
        LOGGER.error("Manifest does not exist: %s", manifest_path)
        sys.exit(1)

    LOGGER.info("Loading manifest: %s", manifest_path)
    rows = load_manifest(manifest_path)
    if not rows:
        LOGGER.warning("Manifest is empty. Nothing to download.")
        return

    cookies_path: Optional[str] = args.cookies
    if cookies_path:
        cookies_path = os.path.abspath(cookies_path)
        if not os.path.isfile(cookies_path):
            LOGGER.warning("Cookies file not found, continuing without cookies: %s", cookies_path)
            cookies_path = None

    successes, failures = process_manifest(
        rows,
        output_dir,
        cookies_path,
        args.resume,
        args.max_count,
    )

    LOGGER.info("Completed. Success: %d | Failed: %d", successes, failures)
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
