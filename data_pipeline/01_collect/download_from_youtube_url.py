import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "common"))
import rvb_config as cfg  # noqa: E402  (loads .env, puts every stage dir on sys.path)
import os
import sys
import argparse
import logging
import re
import csv
from yt_dlp import YoutubeDL
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import threading
import glob

# --- Logging setup ---
# Log to the console.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout  # write straight to stdout
)
main_logger = logging.getLogger(__name__)

# yt-dlp cookies file (Netscape format), needed for age-gated / members-only videos.
# Set with --cookies or $YTDLP_COOKIES; never commit the file itself.
COOKIES_FILE = os.environ.get("YTDLP_COOKIES") or None


def _cookie_opts():
    return {"cookiefile": COOKIES_FILE} if COOKIES_FILE else {}

def sanitize_filename(filename):
    """
    Turn a string into a valid file name by removing characters
    that are not allowed on Windows or Linux/macOS.
    """
    # Drop illegal characters
    sanitized = re.sub(r'[\\/*?:"<>|]', "", filename)
    # Replace spaces to avoid shell-quoting issues (optional)
    sanitized = sanitized.replace(" ", "_")
    # Cap the file-name length
    return sanitized[:100]

def sanitize_label_for_folder(label):
    """
    Turn a label into a safe subfolder name: keep letters, digits, '_' and '-', replace the rest with '_'.
    Returns 'Unlabeled' for an empty label.
    """
    if not label:
        return "Unlabeled"
    # Strip surrounding whitespace
    label = label.strip()
    # Replace unsafe characters with '_'
    label = re.sub(r'[^0-9A-Za-z_\-]+', '_', label)
    # Fall back if nothing is left
    return label if label else "Unlabeled"

def download_youtube_subtitles(url, subs_folder, filename_stem):
    """
    Download English subtitles with yt-dlp (manual subtitles first, auto-generated as fallback) as SRT.
    The file is saved as {filename_stem}.en.srt under subs_folder.
    Returns the saved subtitle path, or None.
    Checks the local disk first and returns immediately if the file exists, without any network request.
    """
    os.makedirs(subs_folder, exist_ok=True)

    # --- Fast local skip: check the standard name first ---
    expected_path = os.path.join(subs_folder, f"{filename_stem}.en.srt")
    if os.path.exists(expected_path):
        main_logger.info(f"Subtitle already exists (skipped): {expected_path}")
        return expected_path

    # --- Fast local skip: then any {stem}.*.srt variant ---
    stem_prefix = f"{filename_stem}."
    for fname in os.listdir(subs_folder):
        if fname.startswith(stem_prefix) and fname.endswith(".srt"):
            path = os.path.join(subs_folder, fname)
            main_logger.info(f"Subtitle already exists (skipped, variant name): {path}")
            return path

    # --- Call yt-dlp only when a download is needed ---
    try:
        ydl_opts = {
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["en"],
            "skip_download": True,
            "subtitlesformat": "srt",
            "outtmpl": {"subtitle": os.path.join(subs_folder, f"{filename_stem}.%(ext)s")},
            "logger": main_logger,
            "quiet": False,
            "no_warnings": True,
            **_cookie_opts(),
        }
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # Check the standard name again after downloading
        if os.path.exists(expected_path):
            main_logger.info(f"Subtitle downloaded: {expected_path}")
            return expected_path
        else:
            # yt-dlp sometimes writes .en-orig.srt or another variant; scan subs_folder for any .srt starting with the stem
            for fname in os.listdir(subs_folder):
                if fname.startswith(stem_prefix) and fname.endswith(".srt"):
                    path = os.path.join(subs_folder, fname)
                    main_logger.info(f"Subtitle downloaded (variant name): {path}")
                    return path
            main_logger.warning("No subtitle file was produced")
            return None
    except Exception as e:
        main_logger.error(f"Error while downloading subtitles: {e}")
        return None

def download_youtube_video(url, save_folder, filename, fmt=None):
    """
    Download the video and audio at the given URL with yt-dlp and merge them
    into the given file name.
    Checks the final target path locally before making any network request.
    """
    main_logger.info(f"Processing URL: {url}")
    try:
        # --- Step 0: fast local skip before any network request ---
        os.makedirs(save_folder, exist_ok=True)
        final_path = os.path.join(save_folder, filename)
        if os.path.exists(final_path):
            main_logger.info(f"Video already exists (skipped): {final_path}")
            return final_path

        # --- Step 1: fetch video info (title and ID) ---
        # Network requests happen only when a download is needed
        with YoutubeDL({'quiet': True, 'noplaylist': True}) as ydl:
            info_dict = ydl.extract_info(url, download=False)
            video_title = info_dict.get('title', 'untitled')
            video_id = info_dict.get('id', None)

        if not video_id:
            main_logger.error(f"Could not extract a video ID from URL: {url}")
            return None

        # --- Step 2: file name and path ---
        # (final_path was already computed in step 0)

        # --- Step 3: download options ---
        # Use an -o template so yt-dlp writes the final file name directly
        output_template = os.path.join(save_folder, filename.replace('.mp4', '.%(ext)s'))

        ydl_opts = {
            # Default format: best <=1440p video stream + best audio stream, merged
            "format": fmt if fmt else "bestvideo[height<=1440][vcodec~='^((avc1|h264)|vp9)']+bestaudio/best[height<=1440]/best",
            "outtmpl": output_template,
            "merge_output_format": "mp4",
            "noplaylist": True,
            "ignoreerrors": True,
            "quiet": False,  # show yt-dlp output while downloading
            "no_warnings": True,
            "logger": main_logger,  # route yt-dlp logs through our logger
            "progress_hooks": [lambda d: print_progress(d, filename)],  # progress callback
            # Existing targets are skipped by our own check above, so yt-dlp need not decide
            "overwrites": False,
            "continuedl": True,
            **_cookie_opts(),
        }

        # --- Step 4: download ---
        main_logger.info(f"Downloading '{video_title}' to {final_path}")
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # With 'merge_output_format': 'mp4' and the output template,
        # yt-dlp handles naming and format conversion.
        if os.path.exists(final_path):
            main_logger.info(f"Downloaded: {final_path}")
            return final_path
        else:
            main_logger.error(f"Download failed. Final file not found: {final_path}")
            return None

    except Exception as e:
        main_logger.error(f"Unexpected error while processing URL {url}: {e}")
        return None

def print_progress(d, filename):
    """Simple progress-bar callback."""
    if d['status'] == 'downloading':
        percent_str = d.get('_percent_str', 'N/A').strip()
        speed_str = d.get('_speed_str', 'N/A').strip()
        eta_str = d.get('_eta_str', 'N/A').strip()
        # '\r' returns to the start of the line to refresh the status
        sys.stdout.write(f"\r -> Downloading '{filename}': {percent_str} | speed: {speed_str} | ETA: {eta_str}")
        sys.stdout.flush()
    elif d['status'] == 'finished':
        sys.stdout.write(f"\n -> File '{filename}' downloaded, post-processing...\n")
        sys.stdout.flush()

def read_csv_selected_rows(csv_path):
    """
    Read the rows marked as selected from the CSV.
    Returns a list of dicts: {"row_index": i, "video_link": url, "label": label, "local_path": str, "local_path_low": str, "subtitle_path": str, "manifest": str}
    A row counts as selected if the value is the string 'true'/'1'/'yes' (case-insensitive) or boolean True.
    """
    selected_rows = []
    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        for i, row in enumerate(rows):
            sel = row.get('selected', '').strip()
            is_selected = False
            if isinstance(sel, bool):
                is_selected = sel
            else:
                is_selected = sel.lower() in ('true', '1', 'yes')
            url = row.get('video_link', '').strip()
            label = row.get('label', '').strip()
            if is_selected and url:
                selected_rows.append({
                    "row_index": i,
                    "video_link": url,
                    "label": label,
                    "local_path": row.get('local_path', '').strip(),
                    "local_path_low": row.get('local_path_low', '').strip(),
                    "subtitle_path": row.get('subtitle_path', '').strip(),
                    "manifest": row.get('manifest', '').strip(),
                })
    return selected_rows

def _find_existing_by_stem(folder, stem, suffix=""):
    """
    Find a file in folder named stem + optional suffix (e.g. '_low') + a common video extension.
    Returns the first match, or None.
    """
    exts = ("mp4", "mkv", "webm", "mov", "m4v")
    patterns = [os.path.join(folder, f"{stem}{suffix}.{ext}") for ext in exts]
    for p in patterns:
        if os.path.exists(p):
            return p
    # Fall back to glob for unusual case or extensions
    g = glob.glob(os.path.join(folder, f"{stem}{suffix}.*"))
    return g[0] if g else None

def _expected_subfolders(base_raw_dir, base_transcript_dir, label):
    lbl = sanitize_label_for_folder(label)
    return os.path.join(base_raw_dir, lbl), os.path.join(base_transcript_dir, lbl)

def _max_existing_stem(sub_output_dir):
    """Scan a directory for files named like 0000(_low).<ext> and return the largest existing stem (int), or -1 if none."""
    if not os.path.isdir(sub_output_dir):
        return -1
    max_idx = -1
    for fname in os.listdir(sub_output_dir):
        m = re.match(r'^(\d{4})(?:_low)?\.(mp4|mkv|webm|mov|m4v)$', fname)
        if m:
            idx = int(m.group(1))
            if idx > max_idx:
                max_idx = idx
    return max_idx


def _stem_from_path(path):
    """Parse the stem (leading four digits) from an existing local path; None if it cannot be parsed."""
    try:
        bn = os.path.basename(path)
        m = re.match(r'^(\d{4})', bn)
        return m.group(1) if m else None
    except Exception:
        return None

# --- Scan the CSV for the largest index already used per label ---
def _scan_max_stem_from_csv(csv_path):
    """
    Scan the whole CSV (selected or not) and, from the four-digit stem of local_path/local_path_low,
    find the largest index used in each label subfolder (sanitized label). Returns dict[label_folder] = max_idx.
    Labels with no index at all are absent from the dict.
    """
    max_by_label = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_label = (row.get('label') or '').strip()
            label_folder = sanitize_label_for_folder(raw_label)
            for key in ('local_path', 'local_path_low'):
                p = (row.get(key) or '').strip()
                if not p:
                    continue
                stem = _stem_from_path(p)
                if stem and stem.isdigit() and len(stem) == 4:
                    idx = int(stem)
                    prev = max_by_label.get(label_folder, -1)
                    if idx > prev:
                        max_by_label[label_folder] = idx
    return max_by_label

def update_csv_paths_by_index(csv_path, updates):
    """
    Update the local_path, local_path_low and subtitle_path columns of the CSV by row index.
    updates: list[{"row_index": int, "local_path": str, "local_path_low": str, "subtitle_path": str}]
    Only the rows at these indices are updated; all others are left unchanged.
    """
    # Read all rows
    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        rows = list(reader)

    # Make sure the three columns exist
    for col in ('local_path', 'local_path_low', 'subtitle_path'):
        if col not in fieldnames:
            fieldnames.append(col)

    # Map row index -> update
    upd_map = {u['row_index']: u for u in updates}

    # Apply updates
    for idx, row in enumerate(rows):
        if idx in upd_map:
            u = upd_map[idx]
            row['local_path'] = u.get('local_path', '') or ''
            row['local_path_low'] = u.get('local_path_low', '') or ''
            row['subtitle_path'] = u.get('subtitle_path', '') or ''

    # Write back
    with open(csv_path, 'w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def _flush_updates(csv_path, buffer, lock):
    """
    Write the buffered updates back to the CSV and clear the buffer (thread-safe).
    buffer: list of updates dicts
    """
    with lock:
        if not buffer:
            return 0
        to_write = list(buffer)
        buffer.clear()
    update_csv_paths_by_index(csv_path, to_write)
    return len(to_write)

def _should_flush(last_flush_ts, flush_every_seconds, pending_count, flush_every_n):
    """
    Decide whether to flush:
    - more than flush_every_seconds since the last flush
    - or pending_count >= flush_every_n
    """
    now = time.time()
    if pending_count >= flush_every_n:
        return True, now
    if (now - last_flush_ts) >= flush_every_seconds:
        return True, now
    return False, last_flush_ts

def _process_download_job(job):
    """
    Worker task: download the full- and low-resolution videos and the subtitles for one entry.
    job fields:
      row_index, url, sub_output_dir, sub_subs_dir, stem, filename, low_filename, skip_low
    Returns a dict for the CSV update (or None on failure).
    """
    row_index = job["row_index"]
    url = job["url"]
    sub_output_dir = job["sub_output_dir"]
    sub_subs_dir = job["sub_subs_dir"]
    stem = job["stem"]
    filename = job["filename"]
    low_filename = job["low_filename"]
    skip_low = job.get("skip_low", False)
    try:
        # Full / default resolution
        local_path = download_youtube_video(url, sub_output_dir, filename)
        # Low resolution (<=480p); skipped when skip_low is True
        if skip_low:
            local_path_low = None
        else:
            low_format = "bestvideo[height<=480][vcodec~='^((avc1|h264)|vp9)']+bestaudio/best[height<=480]/best"
            local_path_low = download_youtube_video(url, sub_output_dir, low_filename, fmt=low_format)
        # Subtitles
        subtitle_path = download_youtube_subtitles(url, sub_subs_dir, filename_stem=stem)
        return {
            "row_index": row_index,
            "local_path": local_path if local_path else "",
            "local_path_low": local_path_low if local_path_low else "",
            "subtitle_path": subtitle_path if subtitle_path else "",
        }
    except Exception as e:
        main_logger.error(f"[worker failed] row {row_index} | {url} | error: {e}")
        return None

def repair_csv_paths(csv_path, base_raw_dir, base_transcript_dir):
    """
    From the existing layout <output>/raw/<label>/ and <output>/transcript/<label>/
    and the order of the selected==True rows within each label (starting at 0000),
    re-derive the files for each row and fill in local_path/local_path_low/subtitle_path.
    """
    selected_rows = read_csv_selected_rows(csv_path)
    per_label_counter = {}
    updates = []
    for item in selected_rows:
        idx = item["row_index"]
        label = item["label"]
        # Index within this label -> stem
        n = per_label_counter.get(label, 0)
        stem = f"{n:04d}"
        per_label_counter[label] = n + 1
        # Subfolders
        sub_output_dir, sub_subs_dir = _expected_subfolders(base_raw_dir, base_transcript_dir, label)
        # Find full-res / low-res video and subtitles
        local_path = _find_existing_by_stem(sub_output_dir, stem, suffix="")
        local_path_low = _find_existing_by_stem(sub_output_dir, stem, suffix="_low")
        # Subtitles are usually {stem}.en.srt; otherwise fall back to stem.*.srt
        subtitle_path = os.path.join(sub_subs_dir, f"{stem}.en.srt")
        if not os.path.exists(subtitle_path):
            g = glob.glob(os.path.join(sub_subs_dir, f"{stem}.*.srt"))
            subtitle_path = g[0] if g else ""
        updates.append({
            "row_index": idx,
            "local_path": local_path or "",
            "local_path_low": local_path_low or "",
            "subtitle_path": subtitle_path or "",
        })
    # Write back in one batch
    update_csv_paths_by_index(csv_path, updates)
    return len(updates)

def main():
    """Parse arguments and run the download loop."""
    global COOKIES_FILE
    parser = argparse.ArgumentParser(
        description="Read YouTube links from a CSV and download the videos and English subtitles (SRT). Files are numbered per label in append-only fashion: <output>/raw/<label> is scanned first and new selected rows are numbered from the current maximum + 1, so inserting or selecting rows in the middle never shifts existing numbers. The starting number is the larger of the highest number already recorded in the CSV (even if the old files were deleted) and the highest number present on disk.",
        epilog="Example: python download_from_youtube_url.py --csv /path/to/catalog.csv --output /path/to/videos"
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="CSV file with a video_link column (manifest with selected=True rows)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.environ.get("RVB_RAW_ROOT", "data/human_reaction_data"),
        help="Download root; raw/<label>/ and transcript/<label>/ are created under it. Default: $RVB_RAW_ROOT"
    )
    parser.add_argument(
        "--subs-output",
        type=str,
        default=None,
        help="Subtitle output folder. Default: <output>/transcript"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=min(8, os.cpu_count() or 4),
        help="Maximum number of parallel download threads. Default: min(8, CPU count)"
    )
    parser.add_argument(
        "--flush-every-n",
        type=int,
        default=10,
        help="Write results back to the CSV every N finished tasks (OR-ed with the time trigger). Default: 10"
    )
    parser.add_argument(
        "--flush-every-seconds",
        type=int,
        default=30,
        help="Write results back to the CSV when this many seconds have passed since the last write (OR-ed with the count trigger). Default: 30"
    )
    parser.add_argument(
        "--repair-only",
        action="store_true",
        help="Repair the CSV only: no downloads; derive each row's local paths from CSV order and the directory layout and write them back."
    )
    parser.add_argument(
        "--cookies",
        default=COOKIES_FILE,
        help="yt-dlp cookies file (Netscape format). Default: $YTDLP_COOKIES",
    )
    args = parser.parse_args()
    COOKIES_FILE = args.cookies

    # Default subtitle folder; fixed layout: <output>/raw and <output>/transcript
    base_raw_dir = os.path.join(args.output, "raw")
    if args.subs_output is None:
        args.subs_output = os.path.join(args.output, "transcript")
    base_transcript_dir = args.subs_output
    os.makedirs(base_raw_dir, exist_ok=True)
    os.makedirs(base_transcript_dir, exist_ok=True)

    if args.repair_only:
        count = repair_csv_paths(args.csv, base_raw_dir, base_transcript_dir)
        main_logger.info(f"[repair done] Filled in {count} selected rows in the CSV from existing files.")
        return

    # Read the selected videos (selected == True) from the CSV
    main_logger.info(f"Reading videos from CSV (only selected==True rows are downloaded): {args.csv}")
    selected_rows = read_csv_selected_rows(args.csv)
    main_logger.info(f"{len(selected_rows)} selected entries. Download root: {args.output}")

    # Scan every used index in the CSV first (including Done/processed rows) so numbering continues even if old files were deleted
    csv_max_by_label = _scan_max_stem_from_csv(args.csv)

    per_label_counter = {}
    jobs = []
    labels_seen = {}
    updates = []
    for item in selected_rows:
        raw_label = item['label']
        label_folder = sanitize_label_for_folder(raw_label)
        if label_folder in labels_seen:
            continue
        sub_output_dir = os.path.join(base_raw_dir, label_folder)
        os.makedirs(sub_output_dir, exist_ok=True)
        # Largest index present on disk (usually -1 if the folder was emptied)
        dir_max = _max_existing_stem(sub_output_dir)
        # Largest index recorded in the CSV for this label (-1 if never numbered)
        csv_max = csv_max_by_label.get(label_folder, -1)
        start_idx = max(dir_max, csv_max) + 1  # take the larger of the two so numbering stays contiguous
        per_label_counter[label_folder] = start_idx
        labels_seen[label_folder] = True

    for item in selected_rows:
        row_index = item['row_index']
        url = item['video_link']
        raw_label = item['label']
        label_folder = sanitize_label_for_folder(raw_label)

        # Subfolders (video and subtitles)
        sub_output_dir = os.path.join(base_raw_dir, label_folder)          # <output>/raw/<label>
        sub_subs_dir = os.path.join(base_transcript_dir, label_folder)     # <output>/transcript/<label>
        os.makedirs(sub_output_dir, exist_ok=True)
        os.makedirs(sub_subs_dir, exist_ok=True)

        # If the row already has a local path, reuse its index (whether or not the file exists); otherwise append
        existing_hd = item.get('local_path') or ''
        existing_low = item.get('local_path_low') or ''
        existing_sub = item.get('subtitle_path') or ''
        manifest = item.get('manifest', '') or ''
        # Game manifests: manifest name contains "game" (case-insensitive)
        skip_low = 'game' in manifest.lower()

        # Reuse an existing stem from the CSV, whether or not the file still exists
        reuse_stem = None
        if existing_hd:
            reuse_stem = _stem_from_path(existing_hd)
        if (not reuse_stem) and existing_low:
            reuse_stem = _stem_from_path(existing_low)

        if reuse_stem:
            stem = reuse_stem
        else:
            n = per_label_counter.get(label_folder, 0)
            stem = f"{n:04d}"
            per_label_counter[label_folder] = n + 1

        filename = f"{stem}.mp4"
        low_filename = f"{stem}_low.mp4"

        # Nothing to download if the HD file, the low-res file and the subtitles all exist
        have_hd = os.path.exists(os.path.join(base_raw_dir, label_folder, filename)) if filename else False
        have_low = os.path.exists(os.path.join(base_raw_dir, label_folder, low_filename)) if low_filename else False
        have_sub = os.path.exists(os.path.join(base_transcript_dir, label_folder, f"{stem}.en.srt")) or bool(glob.glob(os.path.join(base_transcript_dir, label_folder, f"{stem}.*.srt")))

        # The low-res file is not required when skip_low is True
        if have_hd and (have_low or skip_low) and have_sub:
            main_logger.info(f"[skip download] row {row_index} | label '{raw_label}' | all files present, stem={stem}")
            # Still write the paths back so the CSV stays in sync
            updates.append({
                "row_index": row_index,
                "local_path": os.path.join(base_raw_dir, label_folder, filename),
                "local_path_low": ("" if skip_low else os.path.join(base_raw_dir, label_folder, low_filename)),
                "subtitle_path": (os.path.join(base_transcript_dir, label_folder, f"{stem}.en.srt") if os.path.exists(os.path.join(base_transcript_dir, label_folder, f"{stem}.en.srt")) else (glob.glob(os.path.join(base_transcript_dir, label_folder, f"{stem}.*.srt"))[0] if glob.glob(os.path.join(base_transcript_dir, label_folder, f"{stem}.*.srt")) else "")),
            })
            continue

        main_logger.info(f"[plan] row {row_index} | label '{raw_label}' -> subfolder '{label_folder}' | file prefix {stem}")

        jobs.append({
            "row_index": row_index,
            "url": url,
            "sub_output_dir": sub_output_dir,
            "sub_subs_dir": sub_subs_dir,
            "stem": stem,
            "filename": filename,
            "low_filename": low_filename,
            "skip_low": skip_low,
        })

    max_workers = max(1, int(args.max_workers))
    main_logger.info(f"Starting parallel downloads: {len(jobs)} tasks, max_workers={max_workers}")

    partial_updates = []
    flush_lock = threading.Lock()
    last_flush_ts = time.time()

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_job = {executor.submit(_process_download_job, job): job for job in jobs}
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    result = future.result()
                    if result:
                        updates.append(result)
                        # Buffer the result for the next batched write-back
                        with flush_lock:
                            partial_updates.append(result)
                        main_logger.info(f"[done] row {result['row_index']} | {job['url']} | stem={job['stem']}")
                    else:
                        main_logger.warning(f"[skipped/failed] row {job['row_index']} | {job['url']}")
                except Exception as e:
                    main_logger.error(f"[exception] row {job['row_index']} | {job['url']} | error: {e}")
                # After each result, check whether a batched write-back is due
                do_flush, last_flush_ts = _should_flush(last_flush_ts, args.flush_every_seconds, len(partial_updates), args.flush_every_n)
                if do_flush:
                    wrote = _flush_updates(args.csv, partial_updates, flush_lock)
                    if wrote:
                        main_logger.info(f"[CSV incremental write] wrote {wrote} rows (count/time trigger).")
    except KeyboardInterrupt:
        main_logger.warning("Interrupted (Ctrl+C). Writing finished results back to the CSV ...")
    finally:
        # After the loop, flush whatever is left in the buffer
        wrote = _flush_updates(args.csv, partial_updates, flush_lock)
        if wrote:
            main_logger.info(f"[CSV final write] wrote {wrote} remaining rows.")

    # Update the CSV (paths are written only for the selected rows processed in this run)
    main_logger.info("Updating CSV (writing local video and subtitle paths for the selected rows processed in this run) ...")
    update_csv_paths_by_index(args.csv, updates)

    main_logger.info("All downloads finished; CSV updated.")


if __name__ == "__main__":
    main()