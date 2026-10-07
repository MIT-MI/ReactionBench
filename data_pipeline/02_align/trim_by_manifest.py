#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "common"))
import rvb_config as cfg  # noqa: E402  (loads .env, puts every stage dir on sys.path)

import argparse, csv, pathlib, re, subprocess, sys, math, shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

SEG_DEFAULT = 180  # 180 s per segment

STIMULI_PAT = "{id}_stimuli_{seg:03d}.mp4"
# There may be several face tracks (face01/face02/...); typical file name: 0000_face01_seg001.mp4
FACE_GLOB   = "{id}_face*_seg*.mp4"
FACE_RE     = re.compile(r"^(?P<id>\d{4})_(?P<face>face\d+)_seg(?P<seg>\d{3})\.mp4$")

def hms_to_seconds(s):
    parts = s.split(":")
    if len(parts) == 3:
        h, m, sec = parts
        return int(h)*3600 + int(m)*60 + int(sec)
    elif len(parts) == 2:
        m, sec = parts
        return int(m)*60 + int(sec)
    elif len(parts) == 1:
        return int(parts[0])
    else:
        raise ValueError(f"Bad time format: {s}")

def seg_bounds(seg_idx, seg_len):
    start = (seg_idx - 1) * seg_len
    end   = seg_idx * seg_len
    return start, end

def ffmpeg_trim(src, dst, offset_s, duration_s):
    # Input-side -ss (keyframe-accurate) plus -t; stream copy for fast, lossless output (drop -c copy if codecs get in the way)
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(offset_s),
        "-i", str(src),
        "-t", str(duration_s),
        "-c", "copy",
        str(dst)
    ]
    return cmd

def ffmpeg_trim_transcode(src, dst, offset_s, duration_s):
    # Fallback: re-encode for compatibility
    return [
        "ffmpeg", "-y",
        "-ss", str(offset_s),
        "-i", str(src),
        "-t", str(duration_s),
        "-c:v", "libx264",
        "-c:a", "aac",
        "-movflags", "+faststart",
        str(dst)
    ]

def plan_for_interval(rs, re, seg_len, max_seg_guess=1000):
    """Return each segment's action: DELETE / KEEP / TRIM(leading|trailing|single) plus (offset, duration)."""
    first_seg = math.floor(rs / seg_len) + 1
    last_seg  = math.ceil (re / seg_len)
    plan = {}
    for seg in range(1, max_seg_guess+1):
        seg_start, seg_end = seg_bounds(seg, seg_len)
        overlap_start = max(seg_start, rs)
        overlap_end   = min(seg_end,   re)
        if overlap_start >= overlap_end:
            action = "DELETE"
            meta = {}
        else:
            if first_seg == last_seg == seg:
                # The whole interval falls inside one segment
                action = "TRIM(single)"
                meta = {"offset": overlap_start - seg_start, "duration": overlap_end - overlap_start}
            elif seg == first_seg:
                action = "TRIM(leading)"
                meta = {"offset": overlap_start - seg_start, "duration": seg_end - overlap_start}
            elif seg == last_seg:
                action = "TRIM(trailing)"
                meta = {"offset": 0, "duration": overlap_end - seg_start}
            else:
                action = "KEEP"
                meta = {}
        plan[seg] = (action, meta)
        # Stop early once segments past last_seg are DELETE
        if seg > last_seg + 1 and action == "DELETE":
            break
    return plan

def ensure_tmp_dst(dst: pathlib.Path):
    return dst.with_suffix(dst.suffix + ".tmp__rewrite")

def process_one_video(src: pathlib.Path, action: str, meta: dict, apply: bool, backup: bool):
    if action == "DELETE":
        if not apply:
            return f"[DRY] DELETE {src}"
        if backup:
            trash = src.with_suffix(src.suffix + ".bak")
            shutil.move(src, trash)
            return f"[APPLY] MOVE->BAK {src} -> {trash}"
        else:
            src.unlink(missing_ok=True)
            return f"[APPLY] DELETE {src}"

    if action == "KEEP":
        return f"[SKIP] KEEP {src}"

    # TRIM
    offset = meta["offset"]
    duration = meta["duration"]
    tmp_dst = ensure_tmp_dst(src)
    cmd = ffmpeg_trim(src, tmp_dst, offset, duration)
    if not apply:
        return f"[DRY] TRIM {src}  (-ss {offset}s -t {duration}s) -> {src.name}"
    # Run (stream copy first, fall back to transcoding on failure)
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e1:
        # Fall back to transcoding
        cmd2 = ffmpeg_trim_transcode(src, tmp_dst, offset, duration)
        try:
            subprocess.run(cmd2, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e2:
            return f"[ERR] ffmpeg failed for {src}: copy->{e1.returncode}, transcode->{e2.returncode}. STDERR: {e2.stderr.decode(errors='ignore')[:3000]}"
    # Replace in place
    if backup:
        bak = src.with_suffix(src.suffix + ".bak")
        if bak.exists():
            bak.unlink()
        src.rename(bak)
        tmp_dst.rename(src)
        return f"[APPLY] TRIM+REPLACE {src} (backup {bak.name})"
    else:
        src.unlink(missing_ok=True)
        tmp_dst.rename(src)
        return f"[APPLY] TRIM+REPLACE {src}"

def process_id(row, seg_len, stimuli_dir, face_dir, apply, backup, only_selected):
    vid = row["id"]
    if only_selected:
        sel = str(row.get("selected","")).strip().lower() in ("true","1","yes")
        if not sel:
            return [f"[SKIP] id {vid} selected!=True"]

    rs = hms_to_seconds(row["react_start"])
    re = hms_to_seconds(row["react_end"])
    plan = plan_for_interval(rs, re, seg_len)

    logs = []

    # stimuli
    for seg, (action, meta) in plan.items():
        stim = stimuli_dir / STIMULI_PAT.format(id=vid, seg=seg)
        if stim.exists():
            logs.append(process_one_video(stim, action, meta, apply, backup))

    # face (possibly several faceXX tracks); searches subdirectories recursively (e.g. "face - 0000 - /0000_face01_seg001.mp4")
    # For each segment, match every face file and apply the same action
    for seg, (action, meta) in plan.items():
        patt = FACE_GLOB.format(id=vid)
        for f in face_dir.rglob(patt):
            m = FACE_RE.match(f.name)
            if not m:
                continue
            if int(m.group("seg")) != seg:
                continue
            logs.append(process_one_video(f, action, meta, apply, backup))

    return logs

def main():
    ap = argparse.ArgumentParser(description="Trim segments by manifest react_start/react_end.")
    ap.add_argument("--csv", required=True, help="Manifest CSV path (with id, react_start, react_end[, selected])")
    ap.add_argument("--stimuli-dir", required=True, help="Stimuli directory (contains *_stimuli_###.mp4)")
    ap.add_argument("--face-dir", required=True, help="Face directory (contains *_faceXX_seg###.mp4)")
    ap.add_argument("--segment-sec", type=int, default=SEG_DEFAULT, help="Segment length in seconds (default 180)")
    ap.add_argument("--apply", action="store_true", help="Dry run by default; files are modified/deleted only with this flag")
    ap.add_argument("--backup", action="store_true", help="With --apply, back up each original as .bak before replacing/deleting it")
    ap.add_argument("--only-selected", action="store_true", help="Only process CSV rows with selected==True")
    ap.add_argument("--workers", type=int, default=4, help="Number of worker threads")
    args = ap.parse_args()

    stimuli_dir = pathlib.Path(args.stimuli_dir)
    face_dir    = pathlib.Path(args.face_dir)

    # Read the CSV
    rows = []
    with open(args.csv, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    futures = []
    logs_all = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for r in rows:
            futures.append(ex.submit(process_id, r, args.segment_sec, stimuli_dir, face_dir, args.apply, args.backup, args.only_selected))
        for fu in as_completed(futures):
            logs = fu.result()
            logs_all.extend(logs)

    for line in logs_all:
        print(line)

    if not args.apply:
        print("\n[INFO] The above is a dry-run preview (nothing was changed). Add --apply to execute; adding --backup as well is recommended so changes can be rolled back.")

if __name__ == "__main__":
    main()