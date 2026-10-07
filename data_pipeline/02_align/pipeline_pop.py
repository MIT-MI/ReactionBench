from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "common"))
import rvb_config as cfg  # noqa: E402  (loads .env, puts every stage dir on sys.path)

import argparse
import json
import mimetypes
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import re
import time
from typing import List, Optional

import io
try:
    import fcntl  # type: ignore
except Exception:
    fcntl = None  # on non-Unix platforms

def _locked_open(path: str, mode: str):
    f = open(path, mode)
    if fcntl and ("w" in mode or "+" in mode or "a" in mode or "r" in mode):
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        except Exception:
            pass
    return f

def _atomic_replace(src_tmp: str, dst_path: str) -> None:
    os.replace(src_tmp, dst_path)

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from google import genai
from google.genai.types import GenerateContentConfig, Part
from pydantic import BaseModel, Field, field_validator

# === Audio probing helper ===
def _probe_audio_info(path: str) -> dict:
    """
    Return basic audio info using ffprobe.
    Keys: has_audio (bool), codec (str), channels (int), sample_rate (int), duration (float).
    Falls back to has_audio=False on any error.
    """
    info = {"has_audio": False, "codec": "", "channels": 0, "sample_rate": 0, "duration": 0.0}
    try:
        if shutil.which("ffprobe") is None:
            return info
        # Probe audio streams
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_name,channels,sample_rate",
            "-of", "json",
            path,
        ]
        p = subprocess.run(cmd, check=True, capture_output=True, text=True)
        data = json.loads(p.stdout or "{}")
        streams = data.get("streams", [])
        if not streams:
            return info
        s = streams[0]
        info["codec"] = str(s.get("codec_name") or "")
        try:
            info["channels"] = int(s.get("channels") or 0)
        except Exception:
            pass
        try:
            info["sample_rate"] = int(s.get("sample_rate") or 0)
        except Exception:
            pass
        # Get duration from container if available
        dcmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path,
        ]
        dp = subprocess.run(dcmd, check=True, capture_output=True, text=True)
        try:
            info["duration"] = float((dp.stdout or "0").strip() or 0.0)
        except Exception:
            info["duration"] = 0.0
        # Mark as audio-present if codec exists and either channels or sample_rate are non-zero
        info["has_audio"] = bool(info["codec"]) and (info["channels"] > 0 or info["sample_rate"] > 0)
    except Exception:
        return info
    return info

# === Audio attach/mux helpers for face outputs ===
def _attach_audio_from_source(face_video: str, source_video: str) -> bool:
    """Mux the face video (video stream) with the audio stream from source_video.
    Returns True on success, False otherwise. Uses atomic replace to avoid partial writes."""
    try:
        if not os.path.exists(face_video) or not os.path.exists(source_video):
            return False
        out_tmp = face_video + ".tmp_audio"
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-i", face_video,
            "-i", source_video,
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            "-y", out_tmp,
        ]
        subprocess.run(cmd, check=True)
        _atomic_replace(out_tmp, face_video)
        return True
    except Exception:
        try:
            if os.path.exists(out_tmp):
                os.remove(out_tmp)
        except Exception:
            pass
        return False

def _ensure_audio_attached(face_video: str, source_video: str) -> None:
    """If the face video lacks audio but the source has audio, attach it from source."""
    try:
        src_info = _probe_audio_info(source_video)
        face_info = _probe_audio_info(face_video)
        if src_info.get("has_audio") and not face_info.get("has_audio"):
            ok = _attach_audio_from_source(face_video, source_video)
            if ok:
                print(f"🔊 Attached audio from source -> {os.path.basename(face_video)}")
            else:
                print(f"Warning: failed to attach audio to {face_video}")
    except Exception as e:
        print(f"Warning: audio attach check failed for {face_video}: {e}")

class Scene(BaseModel):
    scene_number: int = Field(..., description="1,2,3,... sequential id")
    start_time: str = Field(..., description="HH:MM:SS")
    end_time: str = Field(..., description="HH:MM:SS")
    description: str = Field(..., description="short scene summary")
    is_react: bool = Field(False, description="true if it's a real-time viewer currently reacting to a pop-culture item (TikTok/meme/post/clip), false otherwise")
    is_playing: bool = Field(..., description="true if the pop-culture item is on-screen and/or its audio is playing; false if paused/muted")

    @staticmethod
    def _to_hms(v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("time must be str")
        parts = v.strip().split(":")
        if len(parts) == 2:         # MM:SS
            h, m, s = 0, parts[0], parts[1]
        elif len(parts) == 3:       # H:MM:SS or HH:MM:SS
            h, m, s = parts
        else:
            raise ValueError(f"invalid time format: {v}")
        try:
            h = int(h)
            m = int(m)
            s = int(s)
        except Exception:
            raise ValueError(f"invalid time digits: {v}")
        return f"{h:02d}:{m:02d}:{s:02d}"

    @field_validator("start_time", "end_time", mode="before")
    @classmethod
    def _norm_time(cls, v):
        return cls._to_hms(v)


# === Duration helpers & splitting ===

def _hms_to_seconds(t: str) -> int:
    t = Scene._to_hms(str(t))
    h, m, s = map(int, t.split(":"))
    return h * 3600 + m * 60 + s


def _seconds_to_hms(sec: int) -> str:
    sec = max(0, int(sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _split_long_scenes(df: pd.DataFrame, *, max_seconds: int) -> pd.DataFrame:
    """Given a scene-detection DataFrame with start_time/end_time, split any row whose
    duration exceeds max_seconds into multiple rows of length <= max_seconds.

    Adds columns: `subclip_index` (0,1,2,...) and preserves original `scene_number`.
    """
    if max_seconds is None or int(max_seconds) <= 0 or df.empty:
        # Ensure column exists even if not used
        if "subclip_index" not in df.columns:
            df["subclip_index"] = 0
        return df

    rows = []
    for _, r in df.iterrows():
        try:
            s0 = _hms_to_seconds(r["start_time"])  # normalized by Scene validator
            s1 = _hms_to_seconds(r["end_time"])    
        except Exception:
            # If timestamps are invalid, keep original row
            r = r.copy()
            r["subclip_index"] = 0
            rows.append(r)
            continue
        dur = max(0, s1 - s0)
        if dur <= max_seconds:
            r = r.copy()
            r["subclip_index"] = 0
            rows.append(r)
            continue
        # Split into chunks
        sub_idx = 0
        cur = s0
        while cur < s1:
            end = min(cur + max_seconds, s1)
            part = r.copy()
            part["start_time"] = _seconds_to_hms(cur)
            part["end_time"]   = _seconds_to_hms(end)
            part["subclip_index"] = sub_idx
            rows.append(part)
            sub_idx += 1
            cur = end
    out = pd.DataFrame(rows)
    # Keep canonical column order if possible
    cols = [c for c in ["scene_number","subclip_index","start_time","end_time","description","is_react","is_playing"] if c in out.columns]
    return out[cols + [c for c in out.columns if c not in cols]].reset_index(drop=True)


_BASE_INSTRUCTIONS_POP = """You are a **multimodal General Pop-Culture Reaction Segment Detector**. The input is a reaction video that may include short and long segments reacting to **TikTok/shorts, memes, trailer clips, influencer videos, livestream snippets, screenshots of posts (e.g., Reddit/Twitter/Instagram), news clips, or other pop-culture media**.

**Goal:** Output only the spans where the reactor is actively watch-along reacting to some external pop-culture content that is **visible on screen and/or clearly audible**. Ignore pure talky sections without source content.

**When to cut:**
- Start when external content appears (or its audio begins) and the reactor is clearly reacting to it.
- End when that content disappears, is paused for an extended tangent, switches to a different item, or the video transitions to intro/outro/sponsor.
- It is normal to have **multiple short segments** (one per item). Keep segments contiguous within each item.

**Skip/special cases (do not output):**
- Entire source window is **masked/blurred/covered**, or replaced by static placeholders/timestamps that hide the content.
- The edit is an ultra-rapid montage where items change so fast that timestamps would be meaningless (no stable contiguous spans).

**Cues (use audio & visuals):**
- On-screen phone/vertical panel (e.g., TikTok), embedded tweet/reddit screenshot with scrolling, PIP of a clip or meme; reactor gaze/gestures synchronized with content; content audio.
- Non-reaction cues: long unrelated banter, Q&A, sponsor read, tangents with no source media visible/audible.

**Timestamp rules:**
- Produce **one segment per item** of reacted content. Segments may be short (5–120s) or long if the item is long.
- Timestamps must be within video duration.

**Output format (JSON array; only reaction segments):**
Each object must include:
- `scene_number` (int; 1,2,3…)
- `start_time` (HH:MM:SS)
- `end_time` (HH:MM:SS)
- `description` (short like "reacts to TikTok dance", "laughs at meme compilation", "reads Reddit AITA post")
- `is_react` (boolean; **always `true`**)
- `is_playing` (boolean; **true when the pop-culture item is on-screen and/or its audio is playing; false if paused/muted**)
"""
class PopVisibility(BaseModel):
    contains_movie: bool = Field(..., description="true if pop-culture source content (clip/post/meme) appears (not masked/blurred)")
    reason: str = Field("", description="short explanation of the decision")


def check_pop_visibility(
    video_path: str,
    *,
    project: Optional[str] = None,
    location: str = "us-central1",
    model_id: str = "gemini-2.0-flash",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_output_tokens: int = 512,
) -> tuple[bool, str]:
    """Return (is_visible, reason) by asking Gemini whether the **pop-culture source content**
    (e.g., TikTok/shorts, memes, trailers, screenshots of posts, etc.) is actually visible/moving.

    If the source is hidden (black box, blurred mosaic, covered, static placeholder, only timestamps/frames without moving content), return False.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        return True, "No GOOGLE_CLOUD_PROJECT configured; assuming visible"

    client = genai.Client(vertexai=True, project=project, location=location)

    prompt = (
        "You are verifying **source visibility** in a pop-culture reaction video. "
        "Decide if the actual source item is shown (TikTok/shorts, memes, trailers, screenshots of posts, news clips, etc.).\n"
        "Return JSON with {\"contains_movie\": boolean, \"reason\": string}.\n\n"
        "Consider it **NOT visible** (contains_movie=false) if any apply: the source region is blacked out, covered, blurred/mosaic, a static image/placeholder, "
        "or only timestamp/frames are shown without actual moving/legible content."
    )

    parts = [
        Part.from_text(text=prompt),
        _part_from_local_file(video_path, mime_fallback="video/mp4"),
    ]

    config = GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
        response_mime_type="application/json",
        response_schema=PopVisibility,
    )

    try:
        resp = client.models.generate_content(model=model_id, contents=parts, config=config)
        parsed = resp.parsed
    except Exception:
        parsed = None

    if parsed is None:
        try:
            data = json.loads(getattr(resp, "text", ""))
            mv = PopVisibility(**data)
        except Exception as e:
            return True, f"Visibility check unparsable ({e}); proceeding by default"
    else:
        mv = parsed

    return bool(mv.contains_movie), str(mv.reason or "")

def check_pop_cut_feasibility(
    video_path: str,
    *,
    project: Optional[str] = None,
    location: str = "us-central1",
    model_id: str = "gemini-2.0-flash",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_output_tokens: int = 512,
) -> tuple[bool, str]:
    """Lightweight feasibility check: skip videos that are too chaotic to cut
    (ultra-rapid montage with no stable contiguous spans) or where the reactor never shows
    the source item long enough to form a clip.

    Returns (is_feasible, reason).
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        # If no project is configured, default to feasible to avoid false negatives
        return True, "No GOOGLE_CLOUD_PROJECT configured; assuming feasible"

    client = genai.Client(vertexai=True, project=project, location=location)

    prompt = (
        "You are assessing whether a **general pop-culture reaction** video can be cleanly clip-cut. "
        "Return JSON {\"feasible\": boolean, \"reason\": string}. Consider **NOT feasible** if: "
        "(1) it is an ultra-rapid montage with items changing every 1-2 seconds so timestamps are meaningless; "
        "(2) source items are rarely fully visible/audible; (3) long stretches of talk with ephemeral flashes of content; "
        "(4) heavy subtitles/filters obscure the source region most of the time."
    )

    parts = [Part.from_text(text=prompt), _part_from_local_file(video_path, mime_fallback="video/mp4")]
    config = GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
        response_mime_type="application/json",
        response_schema=dict,  # parse manually
    )

    try:
        resp = client.models.generate_content(model=model_id, contents=parts, config=config)
        data = json.loads(getattr(resp, "text", "{}") or "{}")
        feasible = bool(data.get("feasible", True))
        reason = str(data.get("reason", ""))
        return feasible, reason
    except Exception as e:
        return True, f"Feasibility check error: {e}; proceeding by default"


def _part_from_local_file(path: str, mime_fallback: str = "application/octet-stream") -> Part:
    """
    Load a local file into a Part. Prefer the system-inferred MIME type when available.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    mime, _ = mimetypes.guess_type(path)
    mime = mime or mime_fallback
    with open(path, "rb") as f:
        data = f.read()
    return Part.from_bytes(data=data, mime_type=mime)


def detect_scene_transitions_to_df(
    video_path: str,
    *,
    project: Optional[str] = None,
    location: str = "us-central1",
    model_id: str = "gemini-2.0-flash",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_output_tokens: int = 8000,
) -> pd.DataFrame:
    """
    Use Google GenAI (Gemini) to segment a local video into scenes for General Pop-Culture reaction videos.

    This targets pop-culture reactions where multiple short segments per item are normal.
    Upstream splitting may apply if long segments are detected.

    Parameters
    ----------
    video_path : str
        Path to the local video file (mp4/mov/etc).
    project : Optional[str]
        GCP Project ID. If not provided, it will try to read from the environment variable GOOGLE_CLOUD_PROJECT.
    location : str
        Vertex location (default: us-central1).
    model_id : str
        Model ID (default: "gemini-2.0-flash").
    temperature : float
        Generation temperature (default: 0.0, deterministic).
    top_p : float
        Nucleus sampling (default: 1.0).
    max_output_tokens : int
        Maximum number of output tokens (default: 8000).

    Returns
    -------
    pd.DataFrame
        Columns: ["scene_number","start_time","end_time","description","is_react","is_playing"]
    """
    # Initialize client (same as notebook: genai.Client(vertexai=True, ...))
    project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise ValueError("Project ID not provided. Pass 'project' or set the environment variable GOOGLE_CLOUD_PROJECT.")

    client = genai.Client(vertexai=True, project=project, location=location)

    # Assemble input parts (Pop-culture)
    parts: List[Part] = [
        Part.from_text(text=_BASE_INSTRUCTIONS_POP),
        Part.from_text(text="Apply the scene transition instructions above."),
        _part_from_local_file(video_path, mime_fallback="video/mp4"),
    ]

    # Config consistent with notebook: set JSON schema to list[Scene]
    config = GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
        response_mime_type="application/json",
        response_schema=list[Scene],
    )

    # Call the model
    response = client.models.generate_content(
        model=model_id,
        contents=parts,
        config=config,
    )

    # Prefer SDK structured parsing; if empty or failed, fall back to manual parsing and normalize times
    scenes: List[Scene] = []
    try:
        scenes = response.parsed or []
    except Exception:
        scenes = []

    if not scenes:
        try:
            items = json.loads(response.text)
            # Feed each record to Scene (triggers the time normalization/validation above)
            scenes = [Scene(**item) for item in items]
        except Exception as e:
            raise RuntimeError(
                f"Failed to parse model output into a list of Scene. The raw response is not matching JSON: {e}"
            )

    df = pd.DataFrame([s.model_dump() for s in scenes])
    df = df[["scene_number", "start_time", "end_time", "description", "is_react", "is_playing"]].sort_values(
        "scene_number"
    ).reset_index(drop=True)
    return df





def _sanitize_filename(filename: str) -> str:
    """Mirror the manifest's sanitize: remove illegal chars, replace spaces, cap length."""
    sanitized = re.sub(r'[\\/*?:"<>|]', "", filename)
    sanitized = sanitized.replace(" ", "_")
    return sanitized[:100]


def _label_key(label: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", (label or "").strip().lower()).strip("_")
    return key or "default"

def _safe_merge_persist_manifest(
    manifest_csv: str,
    working_df: pd.DataFrame,
    cols_to_update: list[str],
    key_col: str | None = None,
) -> None:
    try:
        key_col = key_col if key_col in working_df.columns else None
    except Exception:
        key_col = None
    if key_col is None:
        for cand in ("video_link", "local_path", "local_path_low"):
            if cand in working_df.columns:
                key_col = cand
                break

    if key_col and os.path.exists(manifest_csv):
        try:
            disk_df = pd.read_csv(manifest_csv)
        except Exception:
            disk_df = working_df.copy()
        if key_col not in disk_df.columns:
            merged = working_df.copy()
        else:
            upd_cols = [c for c in cols_to_update if c in working_df.columns]
            base_cols = [key_col] + upd_cols
            upd = working_df[base_cols].copy()
            merged = disk_df.merge(upd, on=key_col, how='left', suffixes=('', '_new'))
            for c in upd_cols:
                newc = c + '_new'
                if newc in merged.columns:
                    def _pick(n, o):
                        if pd.isna(n): return o
                        if isinstance(n, str) and n == "": return o
                        return n
                    merged[c] = [_pick(n, o) for n, o in zip(merged[newc], merged.get(c, [None]*len(merged)))]
                    merged.drop(columns=[newc], inplace=True)
    else:
        merged = working_df.copy()

    tmp_path = manifest_csv + ".tmp"
    with _locked_open(tmp_path, "w") as f:
        merged.to_csv(f, index=False)
    _atomic_replace(tmp_path, manifest_csv)


# Helper: check if label means "General pop-culture React" (case/space/underscore tolerant)
def _is_pop_label(v: str) -> bool:
    if v is None:
        return False
    s = str(v).strip().lower().replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    return s == "general pop-culture react"

def _infer_root_and_label(path: str) -> tuple[str, str]:
    """
    From any path under the dataset, infer:
      - root directory (the parent of 'raw'/'clips'/'stimuli'/'masked'/'face')
      - label subfolder name (immediately under the anchor like 'raw' or 'clips').
    Example:
      /data/human_reaction_data/raw/General_pop-culture_React/0000.mp4
      -> ("/data/human_reaction_data", "General_pop-culture_React")
    """
    parts = os.path.abspath(path).split(os.sep)
    anchors = ("raw", "clips", "stimuli", "masked", "face")
    for anchor in anchors:
        if anchor in parts:
            i = parts.index(anchor)
            if i >= 1 and i + 1 < len(parts):
                root = os.sep.join(parts[:i])
                label = parts[i + 1]
                return root, label
    # Fallback: go up two levels; label is the last component
    root = os.path.dirname(os.path.dirname(os.path.abspath(path)))
    label = os.path.basename(os.path.dirname(os.path.abspath(path)))
    return root, label

def _ffmpeg_video_codec_args() -> list[str]:
    """
    Choose a safe set of FFmpeg video-encoding args for broad compatibility.

    We avoid using `-preset` because some FFmpeg builds (or non-libx264 encoders)
    reject it with `Unrecognized option 'preset'`. Prefer libx264 if available,
    otherwise fall back to h264_nvenc, and finally a generic MPEG-4 encoder.
    """
    # Try to detect available encoders
    try:
        encoders = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            check=True, capture_output=True, text=True
        ).stdout.lower()
    except Exception:
        encoders = ""

    if "libx264" in encoders:
        # CRF 20 is a good default; omit -preset to avoid failures on some builds.
        return ["-c:v", "libx264", "-crf", "20"]
    if "h264_nvenc" in encoders:
        # Constant QP-ish control; avoid preset flags to maximize compatibility.
        return ["-c:v", "h264_nvenc", "-cq", "23"]
    # Very safe fallback (widely supported), quality is lower but reliable.
    return ["-c:v", "mpeg4", "-qscale:v", "5"]


# === Bounding box extraction for SOURCE MEDIA panel region ===

def _ffprobe_video_duration_seconds(video_path: str) -> float:
    """Return video duration in seconds using ffprobe; fallback to 0 on failure."""
    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return float(out)
    except Exception:
        return 0.0


def _ffprobe_video_size(video_path: str) -> tuple[int, int]:
    """Return (width, height) of the first video stream using ffprobe; fallback to (0,0)."""
    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=s=x:p=0",
                video_path,
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        w_str, h_str = out.split("x")
        return int(w_str), int(h_str)
    except Exception:
        return 0, 0


def _extract_representative_frame(video_path: str, out_dir: Optional[str] = None) -> tuple[str, int, int]:
    """Extract a representative frame JPG from the video and return (image_path, width, height).

    Heuristic: grab the middle frame (t = duration/2). If duration unavailable, use t=1s.
    """
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg/ffprobe not found. Please install them, e.g., `sudo apt-get install ffmpeg`.")

    duration = _ffprobe_video_duration_seconds(video_path)
    timestamp = max(0.0, duration / 2.0) if duration > 0 else 1.0

    tmp_root = out_dir or tempfile.mkdtemp(prefix="bbox_frame_")
    os.makedirs(tmp_root, exist_ok=True)
    img_path = os.path.join(tmp_root, "frame.jpg")

    # Extract a single video frame as JPEG
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-ss",
        f"{timestamp:.3f}",
        "-i",
        video_path,
        "-frames:v",
        "1",
        "-q:v",
        "2",
        "-y",
        img_path,
    ]
    subprocess.run(cmd, check=True)

    # Prefer the actual video stream dimensions; if missing, try to probe the image itself
    width, height = _ffprobe_video_size(video_path)
    if width <= 0 or height <= 0:
        try:
            out = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "stream=width,height",
                    "-of",
                    "csv=s=x:p=0",
                    img_path,
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            w_str, h_str = out.split("x")
            width, height = int(w_str), int(h_str)
        except Exception:
            pass

    return img_path, width, height


def _call_gemini_bbox_mv(image_path: str, *, project: Optional[str], location: str, model_id: str) -> list[int]:
    """Call Gemini to get one bounding box [ymin, xmin, ymax, xmax] in 0–1000 coords. Returns 4 ints.

    This follows the user's suggested prompt/regex-extract style for robustness across model variants.
    """
    client = genai.Client(vertexai=True, project=project, location=location)

    prompt = (
        "This image is a reaction video layout. One region shows the streamer/reactor face camera, "
        "and another region shows the **SOURCE MEDIA panel** (e.g., TikTok vertical video, embedded meme/clip, trailer, or a screenshot of a post). "
        "Return the bounding box for the entire SOURCE MEDIA panel (include its visible border if any) "
        "in [ymin, xmin, ymax, xmax], with all four integers normalized to a 0-1000 coordinate space. "
        "Return ONLY the JSON array of four integers."
    )

    parts: List[Part] = [
        Part.from_text(text=prompt),
        _part_from_local_file(image_path, mime_fallback="image/jpeg"),
    ]

    # Keep config minimal; we'll parse text with a regex for robustness
    config = GenerateContentConfig(temperature=0.0, top_p=1.0, max_output_tokens=256)

    response = client.models.generate_content(
        model=model_id,
        contents=parts,
        config=config,
    )

    response_text = getattr(response, "text", "") or ""
    match = re.search(r"\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]", response_text)
    if not match:
        # Try a secondary, simpler prompt once
        alt_prompt = (
            "Return ONLY the bounding box for the SOURCE MEDIA panel as [ymin, xmin, ymax, xmax] in 0-1000 integers."
        )
        response = client.models.generate_content(
            model=model_id,
            contents=[Part.from_text(text=alt_prompt), _part_from_local_file(image_path, mime_fallback="image/jpeg")],
            config=config,
        )
        response_text = getattr(response, "text", "") or ""
        match = re.search(r"\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]", response_text)
        if not match:
            raise ValueError(f"Could not find JSON array in response: {response_text}")

    box = json.loads(match.group(0))
    if not (isinstance(box, list) and len(box) == 4):
        raise ValueError(f"Unexpected bbox format: {box}")
    return [int(box[0]), int(box[1]), int(box[2]), int(box[3])]


def bounding_box_extraction(
    video_path: str,
    *,
    df_path: Optional[str] = None,
    project: Optional[str] = None,
    location: str = "us-central1",
    model_id: str = "gemini-2.0-flash"
) -> tuple[int, int, int, int, str]:
    """Extract the bounding box of the SOURCE MEDIA panel (e.g. a vertical TikTok, an embedded meme/clip, a post screenshot) from a reaction clip.

    Workflow:
    1) Grab a representative frame (middle of the video) as a JPEG.
    2) Ask Gemini to return a single bbox as [ymin, xmin, ymax, xmax] in 0–1000 coords.
    3) Convert that normalized box to **pixel coordinates** relative to the frame and return `(xmin, ymin, xmax, ymax, masked_out_path)`.

    Returns
    -------
    (xmin, ymin, xmax, ymax, masked_out_path) : tuple
        - xmin, ymin, xmax, ymax: integer pixel coordinates of the detected SOURCE MEDIA panel region.
        - masked_out_path: path to the blacked-out (masked) clip if generated, else "".
        If detection fails, returns (0, 0, width, height, "") or (0, 0, 0, 0, "") on error.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    # === Reuse cached bbox from CSV to avoid repeated Gemini calls ===
    cached_bbox = None
    if df_path is not None and os.path.exists(df_path):
        try:
            df_cached = pd.read_csv(df_path)
            needed = ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
            if all(col in df_cached.columns for col in needed):
                mask = df_cached["clip_path"] == video_path if "clip_path" in df_cached.columns else None
                row = df_cached.loc[mask].iloc[0] if mask is not None and mask.any() else None
                # Accept only non-empty numeric values
                if row is not None and all(str(row[c]).strip() != "" for c in needed):
                    try:
                        x1_c = int(float(row["bbox_x1"]))
                        y1_c = int(float(row["bbox_y1"]))
                        x2_c = int(float(row["bbox_x2"]))
                        y2_c = int(float(row["bbox_y2"]))
                        # Basic sanity: positive area
                        if x2_c > x1_c and y2_c > y1_c:
                            cached_bbox = (x1_c, y1_c, x2_c, y2_c)
                            print(f"bounding_box_extraction: Reusing cached bbox for {os.path.basename(video_path)} -> {cached_bbox}")
                    except Exception:
                        cached_bbox = None
        except Exception as e:
            print(f"bounding_box_extraction: failed to read cached bbox from {df_path}: {e}")

    # If we have a cached bbox, skip Gemini and jump straight to stimuli/masked generation using it
    if cached_bbox is not None:
        x1, y1, x2, y2 = cached_bbox
        # Proceed to the existing stimuli/masked generation block below
    else:
        # Resolve project/location from env if not provided
        project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project:
            print("bounding_box_extraction: missing GCP project (set GOOGLE_CLOUD_PROJECT or pass project=)")
            # Try best-effort fallback: use full-frame
            _, w_fallback, h_fallback = _extract_representative_frame(video_path)
            x1, y1, x2, y2 = 0, 0, max(0, w_fallback), max(0, h_fallback)
        else:
            try:
                frame_path, width, height = _extract_representative_frame(video_path)
            except Exception as e:
                print(f"bounding_box_extraction: failed to extract frame: {e}")
                x1, y1, x2, y2 = 0, 0, 0, 0
            else:
                if width <= 0 or height <= 0:
                    print("bounding_box_extraction: unknown frame size; using full frame")
                    x1, y1, x2, y2 = 0, 0, 0, 0
                else:
                    try:
                        ymin, xmin, ymax, xmax = _call_gemini_bbox_mv(frame_path, project=project, location=location, model_id=model_id)
                    except Exception as e:
                        print(f"bounding_box_extraction: model bbox call failed: {e}")
                        x1, y1, x2, y2 = 0, 0, width, height
                    else:
                        def _clamp(v: int) -> int:
                            return max(0, min(1000, int(v)))
                        ymin = _clamp(ymin); xmin = _clamp(xmin); ymax = _clamp(ymax); xmax = _clamp(xmax)
                        # Convert to pixels
                        x1 = int(round(xmin / 1000.0 * width))
                        y1 = int(round(ymin / 1000.0 * height))
                        x2 = int(round(xmax / 1000.0 * width))
                        y2 = int(round(ymax / 1000.0 * height))
                        # Sanity fixes
                        x1, x2 = max(0, min(x1, width)), max(0, min(x2, width))
                        y1, y2 = max(0, min(y1, height)), max(0, min(y2, height))
                        if x2 < x1: x1, x2 = x2, x1
                        if y2 < y1: y1, y2 = y2, y1

    # --- Additional stimuli crop logic ---
    masked_out_path = ""
    try:
        # Infer root and label from the input clip (which should live under /clips/<label>/...)
        root_root, label_folder = _infer_root_and_label(video_path)
        label_folder = _sanitize_filename(label_folder)

        clip_dir = os.path.dirname(video_path)
        clip_dir_base = os.path.basename(clip_dir)  # e.g., "{stem}_clips"
        hi_stem = clip_dir_base[:-6] if clip_dir_base.endswith("_clips") else clip_dir_base

        stimuli_out_dir = os.path.join(root_root, 'stimuli', label_folder, f"{hi_stem}_stimuli")
        os.makedirs(stimuli_out_dir, exist_ok=True)
        masked_out_dir = os.path.join(root_root, 'masked', label_folder, f"{hi_stem}_masked")
        os.makedirs(masked_out_dir, exist_ok=True)
        crop_w = max(1, x2 - x1)
        crop_h = max(1, y2 - y1)
        stimuli_out_path = os.path.join(stimuli_out_dir, os.path.basename(video_path))
        masked_out_path = os.path.join(masked_out_dir, os.path.basename(video_path))
        vcodec_args = _ffmpeg_video_codec_args()
        cmd = [
            'ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin',
            '-i', video_path,
            '-filter:v', f'crop={crop_w}:{crop_h}:{x1}:{y1}',
            *vcodec_args,
            '-an',
            '-y', stimuli_out_path,
        ]
        try:
            subprocess.run(cmd, check=True)
        except Exception as e:
            print(f"Warning: failed to crop stimuli video for {video_path}: {e}")
        # Write masked version (black out the SOURCE MEDIA panel region)
        try:
            cmd_masked = [
                'ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin',
                '-i', video_path,
                '-filter:v', f'drawbox=x={x1}:y={y1}:w={crop_w}:h={crop_h}:color=black@1:t=fill',
                *vcodec_args,
                '-an',
                '-y', masked_out_path,
            ]
            subprocess.run(cmd_masked, check=True)
        except Exception as e:
            print(f"Warning: failed to write masked video for {video_path}: {e}")
        # Update CSV if applicable
        if df_path is not None and os.path.exists(df_path):
            try:
                df = pd.read_csv(df_path)
                # Ensure output dir columns exist
                if "stimuli_out_dir" not in df.columns:
                    df["stimuli_out_dir"] = ""
                if "masked_out_dir" not in df.columns:
                    df["masked_out_dir"] = ""
                # Ensure bbox columns exist
                for col in ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]:
                    if col not in df.columns:
                        df[col] = ""
                # Update rows that match the video_path
                if "clip_path" in df.columns:
                    mask = df["clip_path"] == video_path
                    if mask.any():
                        df.loc[mask, "stimuli_out_dir"] = stimuli_out_dir
                        df.loc[mask, "masked_out_dir"] = masked_out_dir
                        # Also persist bbox coordinates for this clip
                        df.loc[mask, "bbox_x1"] = x1
                        df.loc[mask, "bbox_y1"] = y1
                        df.loc[mask, "bbox_x2"] = x2
                        df.loc[mask, "bbox_y2"] = y2
                else:
                    df["stimuli_out_dir"] = stimuli_out_dir
                    df["masked_out_dir"] = masked_out_dir
                    # If no clip_path, update all rows with bbox info
                    df["bbox_x1"] = x1
                    df["bbox_y1"] = y1
                    df["bbox_x2"] = x2
                    df["bbox_y2"] = y2
                df.to_csv(df_path, index=False)
            except Exception as e:
                print(f"Warning: failed to update stimuli_out_dir/masked_out_dir/bbox in {df_path}: {e}")
    except Exception as e:
        print(f"Warning: stimuli crop logic failed for {video_path}: {e}")

    return (x1, y1, x2, y2, masked_out_path)


def bounding_box_extraction_batch(
    video_paths: List[str],
    *,
    df_path: Optional[str] = None,
    project: Optional[str] = None,
    location: str = "us-central1",
    model_id: str = "gemini-2.0-flash",
    max_workers: int = 6,
) -> dict[str, tuple[int, int, int, int, str]]:
    """Run bounding_box_extraction in parallel for a list of videos.

    Returns a dict mapping input video path -> (x1, y1, x2, y2, masked_clip_path).
    masked_clip_path may be "" if masking failed; callers may fall back to the original clip.
    """
    if not video_paths:
        return {}

    results: dict[str, tuple[int, int, int, int, str]] = {}

    def _worker(vp: str) -> tuple[str, tuple[int, int, int, int, str]]:
        try:
            x1, y1, x2, y2, masked = bounding_box_extraction(
                vp, df_path=df_path, project=project, location=location, model_id=model_id
            )
            print(f"BBox ✅ {os.path.basename(vp)} -> ({x1},{y1},{x2},{y2}), masked={bool(masked)}")
            return vp, (x1, y1, x2, y2, masked)
        except Exception as e:
            print(f"BBox ❌ {os.path.basename(vp)} -> {e}")
            return vp, (0, 0, 0, 0, "")

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_worker, vp): vp for vp in video_paths}
        for fut in as_completed(futs):
            vp, val = fut.result()
            results[vp] = val

    return results


def talking_head_extraction(
    video_path: str,
    *,
    df_path: Optional[str] = None,
    skip_if_exists: bool = True,
    min_ok_bytes: int = 16384,
    fast_mode: bool = True,
    providers: Optional[str] = "TensorrtExecutionProvider,CUDAExecutionProvider",
    trt_cache_dir: Optional[str] = None,
) -> str:
    """Extract talking head (head-and-shoulders crop) from a video clip.

    Creates a 'face' folder structure mirroring the clips folders and ensures the
    resulting face video includes the corresponding audio track from the input clip
    (same time span), attaching it if the ghs_crop output is video-only.

    This function can take either a regular clip or a masked clip produced by bounding_box_extraction.
    When a masked clip is provided, the CSV update will attempt to route the face_out_dir to the row
    corresponding to the original unmasked clip when possible.

    Parameters
    ----------
    video_path : str
        Path to the input video clip (can be a masked clip from bounding_box_extraction)
    df_path : Optional[str]
        Path to the CSV file to update with face folder path
    skip_if_exists : bool
        If True, skip processing if output already exists and is valid
    min_ok_bytes : int
        Minimum file size to consider a face video "good"
    fast_mode : bool
        If True, use optimized settings for speed over accuracy
    providers : Optional[str]
        Execution providers string for ONNX Runtime (default: "TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider").
    trt_cache_dir : Optional[str]
        Directory to cache TensorRT engines; improves throughput for repeated model inferences.

    Returns
    -------
    str
        Path to the output face video file
    """
    start_time = time.time()
    print(f"🎭 Starting face extraction for {os.path.basename(video_path)} at {time.strftime('%H:%M:%S')}")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input video not found: {video_path}")

    # Determine folder structure using the new layout: <root>/<category>/<label>/...
    root_root, label_folder = _infer_root_and_label(video_path)
    label_folder = _sanitize_filename(label_folder)
    clip_dir = os.path.dirname(video_path)
    clip_dir_base = os.path.basename(clip_dir)  # e.g., "video_name_clips"

    face_stem = clip_dir_base[:-6] + "_face" if clip_dir_base.endswith("_clips") else clip_dir_base + "_face"
    face_out_dir = os.path.join(root_root, 'face', label_folder, face_stem)
    os.makedirs(face_out_dir, exist_ok=True)

    # Output file path
    face_out_path = os.path.join(face_out_dir, os.path.basename(video_path))

    # Prefer TensorRT on large NVIDIA GPUs when available; fall back to CUDA, then CPU
    providers = providers or "TensorrtExecutionProvider,CUDAExecutionProvider"
    print(f"🔧 Using execution providers: {providers}")
    # Enforce GPU-only execution
    if "CUDAExecutionProvider" in providers or "TensorrtExecutionProvider" in providers:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU-only mode requested but no CUDA-capable GPU is available. Aborting.")
    def _is_ok(p: str) -> bool:
        try:
            return os.path.exists(p) and os.path.getsize(p) >= max(1, int(min_ok_bytes))
        except Exception:
            return False

    # Skip if output already exists and is valid
    if skip_if_exists and _is_ok(face_out_path):
        elapsed = time.time() - start_time
        print(f"⏩ Face extraction already exists: {face_out_path} (checked in {elapsed:.2f}s)")
        # Still update CSV if needed
        _update_csv_face_path(df_path, video_path, face_out_dir)
        # Ensure audio is present (attach from source clip if missing)
        _ensure_audio_attached(face_out_path, video_path)
        return face_out_path

    # Check if ghs_crop tool is available
    try:
        # Import the talking_head ghs_crop module
        talking_head_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "talking_head")
        if talking_head_root not in sys.path:
            sys.path.insert(0, talking_head_root)

        # Optimize parameters for speed
        if fast_mode:
            # More aggressive sampling for speed on large GPUs
            step = "10"            # sample every 8th frame
            conf_thres = "0.6"    # allow slightly lower confidence
            det_size = "384"      # modest detector size to keep recall
            safety_scale = "1.5"
        else:
            step = "3"
            conf_thres = "0.7"
            det_size = "640"
            safety_scale = "1.3"

        # Run ghs_crop command using subprocess for better control
        cmd = [
            sys.executable, "-m", "ghs_crop",
            "--input", video_path,
            "--output", face_out_path,
            "--step", step,
            "--conf-thres", conf_thres,
            "--det-size", det_size,
            "--headroom-bias", "0.05",
            "--safety-scale", safety_scale,
            "--face-area-percentile", "60",
            "--providers", providers,
        ]

        # Change to the talking_head directory to ensure proper imports
        orig_cwd = os.getcwd()
        try:
            os.chdir(talking_head_root)
            # Set environment variables to optimize performance
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "4"  # Limit CPU threads to avoid oversubscription
            env["MKL_NUM_THREADS"] = "4"
            # Enable ONNX Runtime TensorRT EP optimizations when available
            if trt_cache_dir is None:
                trt_cache_dir = os.path.join(face_out_dir, ".trt_cache")
            os.makedirs(trt_cache_dir, exist_ok=True)
            env["ORT_TENSORRT_ENGINE_CACHE_ENABLE"] = "1"
            env["ORT_TENSORRT_ENGINE_CACHE_PATH"] = trt_cache_dir
            env["ORT_TENSORRT_FP16_ENABLE"] = "1"           # allow FP16 kernels
            env["ORT_TENSORRT_MAX_WORKSPACE_SIZE"] = str(8 * 1024 * 1024 * 1024)  # 8GB
            # Reduce host mem fragmentation on very parallel runs
            env["CUDA_MODULE_LOADING"] = "LAZY"
            # Disallow CPU fallback within ORT by omitting CPU EP and hinting GPU usage
            env["ORT_DISABLE_CPU"] = "1"
            env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

            ghs_start = time.time()
            print(f"🔄 Running ghs_crop command for {os.path.basename(video_path)}...")

            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                env=env,
                timeout=300,  # 5 minute timeout per video
            )

            ghs_elapsed = time.time() - ghs_start
            total_elapsed = time.time() - start_time
            print(f"✅ Face extraction completed: {face_out_path} (ghs_crop: {ghs_elapsed:.2f}s, total: {total_elapsed:.2f}s)")
            # Attach/mux audio from the source clip into the face output if missing
            _ensure_audio_attached(face_out_path, video_path)
        finally:
            os.chdir(orig_cwd)

    except subprocess.TimeoutExpired:
        print(f"Warning: face extraction timed out for {video_path}")
        # Create empty file to mark as attempted
        with open(face_out_path, 'w') as f:
            f.write("")
    except subprocess.CalledProcessError as e:
        print(f"Warning: face extraction failed for {video_path}: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        # Create empty file to mark as attempted
        with open(face_out_path, 'w') as f:
            f.write("")
    except Exception as e:
        print(f"Warning: face extraction failed for {video_path}: {e}")
        # Create empty file to mark as attempted
        with open(face_out_path, 'w') as f:
            f.write("")

    # Update CSV with face folder path
    csv_start = time.time()
    _update_csv_face_path(df_path, video_path, face_out_dir)
    csv_elapsed = time.time() - csv_start

    total_elapsed = time.time() - start_time
    print(f"📊 CSV update completed in {csv_elapsed:.2f}s, total time: {total_elapsed:.2f}s")

    return face_out_path


def talking_head_extraction_batch(
    video_paths: List[str],
    *,
    df_path: Optional[str] = None,
    skip_if_exists: bool = True,
    min_ok_bytes: int = 16384,
    fast_mode: bool = True,
    providers: Optional[str] = "TensorrtExecutionProvider,CUDAExecutionProvider",
    trt_cache_dir: Optional[str] = None,
    max_workers: int = 2,
) -> List[str]:
    """Batch process multiple videos for talking head extraction.

    This is more efficient than calling talking_head_extraction individually
    as it can process multiple videos in parallel and reuse model loading.

    Parameters
    ----------
    video_paths : List[str]
        List of video paths to process
    df_path : Optional[str]
        Path to the CSV file to update with face folder paths
    skip_if_exists : bool
        If True, skip processing if output already exists and is valid
    min_ok_bytes : int
        Minimum file size to consider a face video "good"
    fast_mode : bool
        If True, use optimized settings for speed over accuracy
    providers : Optional[str]
        Execution providers string for ONNX Runtime (default: "TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider").
    trt_cache_dir : Optional[str]
        Directory to cache TensorRT engines; improves throughput for repeated model inferences.
    max_workers : int
        Maximum number of parallel workers

    Returns
    -------
    List[str]
        List of output face video paths
    """
    if not video_paths:
        return []

    # Filter videos that need processing
    videos_to_process = []
    output_paths = []

    def _is_ok(p: str) -> bool:
        try:
            return os.path.exists(p) and os.path.getsize(p) >= max(1, int(min_ok_bytes))
        except Exception:
            return False

    for video_path in video_paths:
        if not os.path.exists(video_path):
            print(f"Warning: video not found: {video_path}")
            output_paths.append("")
            continue

        root_root, label_folder = _infer_root_and_label(video_path)
        label_folder = _sanitize_filename(label_folder)

        clip_dir = os.path.dirname(video_path)
        clip_dir_base = os.path.basename(clip_dir)

        face_stem = clip_dir_base[:-6] + "_face" if clip_dir_base.endswith("_clips") else clip_dir_base + "_face"
        face_out_dir = os.path.join(root_root, 'face', label_folder, face_stem)
        os.makedirs(face_out_dir, exist_ok=True)
        face_out_path = os.path.join(face_out_dir, os.path.basename(video_path))
        output_paths.append(face_out_path)

        # Check if we need to process this video
        if skip_if_exists and _is_ok(face_out_path):
            print(f"Face extraction already exists: {face_out_path}")
            _update_csv_face_path(df_path, video_path, face_out_dir)
            _ensure_audio_attached(face_out_path, video_path)
        else:
            videos_to_process.append((video_path, face_out_path))

    if not videos_to_process:
        print("All face extractions already exist, skipping batch processing")
        return output_paths

    batch_start = time.time()
    print(f"🚀 Batch processing {len(videos_to_process)} videos for face extraction at {time.strftime('%H:%M:%S')}")

    # Use the process_masked_videos_optimized.py approach for batch processing
    try:
        talking_head_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "talking_head")
        if talking_head_root not in sys.path:
            sys.path.insert(0, talking_head_root)

        # Use top-level imports for ThreadPoolExecutor, as_completed, threading

        lock = threading.Lock()
        processed_count = 0
        failed_count = 0

        def process_single_video_optimized(video_input_output):
            nonlocal processed_count, failed_count
            video_path, face_out_path = video_input_output

            single_start = time.time()
            print(f"🎬 Starting {os.path.basename(video_path)} at {time.strftime('%H:%M:%S')}")
            # Print which execution providers are being used for this video
            print(f"🔧 Using execution providers for {os.path.basename(video_path)}: {providers or 'default (CPU)'}")
            if ("CUDAExecutionProvider" in (providers or "")) or ("TensorrtExecutionProvider" in (providers or "")):
                if not torch.cuda.is_available():
                    raise RuntimeError("GPU-only mode requested but no CUDA-capable GPU is available. Aborting.")

            try:
                # Optimize parameters for speed
                if fast_mode:
                    step = "8"
                    conf_thres = "0.6"
                    det_size = "384"
                    safety_scale = "1.5"
                else:
                    step = "3"
                    conf_thres = "0.7"
                    det_size = "640"
                    safety_scale = "1.3"

                cmd = [
                    sys.executable, "-m", "ghs_crop",
                    "--input", video_path,
                    "--output", face_out_path,
                    "--step", step,
                    "--conf-thres", conf_thres,
                    "--det-size", det_size,
                    "--headroom-bias", "0.05",
                    "--safety-scale", safety_scale,
                    "--face-area-percentile", "60",
                    "--providers", (providers or "TensorrtExecutionProvider,CUDAExecutionProvider"),
                ]

                # Set environment variables to optimize performance
                env = os.environ.copy()
                env["OMP_NUM_THREADS"] = "2"  # Fewer threads for parallel processing
                env["MKL_NUM_THREADS"] = "2"
                # Enable ORT TensorRT EP engine caching/fp16 for speed
                _cache_dir = trt_cache_dir or os.path.join(os.path.dirname(face_out_path), ".trt_cache")
                os.makedirs(_cache_dir, exist_ok=True)
                env["ORT_TENSORRT_ENGINE_CACHE_ENABLE"] = "1"
                env["ORT_TENSORRT_ENGINE_CACHE_PATH"] = _cache_dir
                env["ORT_TENSORRT_FP16_ENABLE"] = "1"
                env["ORT_TENSORRT_MAX_WORKSPACE_SIZE"] = str(8 * 1024 * 1024 * 1024)
                env["CUDA_MODULE_LOADING"] = "LAZY"
                env["ORT_DISABLE_CPU"] = "1"
                env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

                orig_cwd = os.getcwd()
                try:
                    os.chdir(talking_head_root)
                    ghs_start = time.time()
                    result = subprocess.run(
                        cmd,
                        check=True,
                        capture_output=True,
                        text=True,
                        env=env,
                        timeout=300,
                    )
                    ghs_elapsed = time.time() - ghs_start

                    # Ensure face output carries audio from the input clip
                    _ensure_audio_attached(face_out_path, video_path)

                    # Update CSV
                    csv_start = time.time()
                    face_out_dir = os.path.dirname(face_out_path)
                    _update_csv_face_path(df_path, video_path, face_out_dir)
                    csv_elapsed = time.time() - csv_start

                    single_elapsed = time.time() - single_start

                    with lock:
                        processed_count += 1
                        print(f"✅ Face extraction completed ({processed_count}/{len(videos_to_process)}): {os.path.basename(video_path)} "
                              f"(ghs: {ghs_elapsed:.2f}s, csv: {csv_elapsed:.2f}s, total: {single_elapsed:.2f}s)")

                    return True, video_path, f"Success ({single_elapsed:.2f}s)"

                finally:
                    os.chdir(orig_cwd)

            except Exception as e:
                single_elapsed = time.time() - single_start
                with lock:
                    failed_count += 1
                    print(f"❌ Face extraction failed ({failed_count}): {os.path.basename(video_path)} - {e} (took {single_elapsed:.2f}s)")

                # Create empty file to mark as attempted
                try:
                    with open(face_out_path, 'w') as f:
                        f.write("")
                except Exception:
                    pass

                return False, video_path, f"{str(e)} ({single_elapsed:.2f}s)"

        # Process videos in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_video = {
                executor.submit(process_single_video_optimized, video_input_output): video_input_output
                for video_input_output in videos_to_process
            }

            for future in as_completed(future_to_video):
                try:
                    success, video_path, message = future.result()
                except Exception as e:
                    video_path = future_to_video[future][0]
                    print(f"Exception in batch processing {video_path}: {e}")

        batch_elapsed = time.time() - batch_start
        print(f"🎯 Batch face extraction completed: {processed_count} success, {failed_count} failed (total batch time: {batch_elapsed:.2f}s)")
        if processed_count > 0:
            avg_time = batch_elapsed / processed_count
            print(f"📈 Average time per successful extraction: {avg_time:.2f}s")

    except Exception as e:
        print(f"Warning: batch face extraction failed, falling back to individual processing: {e}")
        # Fallback to individual processing
        for video_path, face_out_path in videos_to_process:
            try:
                talking_head_extraction(
                    video_path,
                    df_path=df_path,
                    skip_if_exists=skip_if_exists,
                    min_ok_bytes=min_ok_bytes,
                    fast_mode=fast_mode,
                )
            except Exception as e:
                print(f"Individual fallback also failed for {video_path}: {e}")

    # Optional: clear CUDA cache after batch completes
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return output_paths


def _update_csv_face_path(df_path: Optional[str], video_path: str, face_out_dir: str) -> None:
    """
    Update the CSV file with the face folder path.
    If video_path is a masked clip (e.g., from bounding_box_extraction), and no row matches clip_path == video_path,
    the function will attempt to map it back to the original clip path by replacing '/masked/{stem}_masked/filename.mp4'
    with '/clips/{stem}_clips/filename.mp4' for the immediate parent directory. If still no match, fallback to previous behavior.
    """
    if df_path is None or not os.path.exists(df_path):
        return

    try:
        df = pd.read_csv(df_path)

        # Add face_out_dir column if it doesn't exist
        if "face_out_dir" not in df.columns:
            df["face_out_dir"] = ""

        # Update rows that match the video path
        if "clip_path" in df.columns:
            mask = df["clip_path"] == video_path
            if not mask.any():
                # Try mapping masked path back to original clip path
                # e.g., /masked/{stem}_masked/filename.mp4 -> /clips/{stem}_clips/filename.mp4
                masked_parts = video_path.split(os.sep)
                try:
                    idx = masked_parts.index("masked")
                    if idx + 1 < len(masked_parts):
                        masked_dir = masked_parts[idx + 1]
                        if masked_dir.endswith("_masked"):
                            orig_dir = masked_dir[:-7] + "_clips"
                            candidate_orig = os.sep.join(
                                masked_parts[:idx] + ["clips", orig_dir] + masked_parts[idx + 2 :]
                            )
                            mask = df["clip_path"] == candidate_orig
                except Exception:
                    pass
            if mask.any():
                df.loc[mask, "face_out_dir"] = face_out_dir
            else:
                # If still no match, fallback: update all rows (legacy fallback)
                df["face_out_dir"] = face_out_dir
        else:
            # If no clip_path column, update all rows (fallback)
            df["face_out_dir"] = face_out_dir

        df.to_csv(df_path, index=False)
        print(f"Updated CSV with face_out_dir: {face_out_dir}")

    except Exception as e:
        print(f"Warning: failed to update face_out_dir in {df_path}: {e}")


def cut_reaction_clips_from_csv(root_dir: str, video_path: str, csv_path: str, max_clip_seconds: int | None = None) -> tuple[str, int]:
    """Cut reaction clips from a CSV for a given video.

    Reads `csv_path`, filters rows where both `is_react` and `is_playing` are true,
    and uses ffmpeg to cut clips from `input_video` into an output directory named
    `<video_stem>_clips` under `root_dir`.

    Adds a `clip_path` column to the CSV with the output clip file paths.
    Optionally splits long scenes if `max_clip_seconds` is set.

    Returns (out_dir, number_of_new_clips).
    """
    stem = os.path.splitext(os.path.basename(video_path))[0]
    input_video = video_path if os.path.isabs(video_path) else os.path.join(root_dir, video_path)
    root_root, label_folder = _infer_root_and_label(input_video)
    label_folder = _sanitize_filename(label_folder)
    out_dir = os.path.join(root_root, 'clips', label_folder, f"{stem}_clips")

    if not os.path.exists(csv_path):
        print(f"Skipping cutting: CSV not found for {video_path}: {csv_path}")
        return out_dir, 0

    df = pd.read_csv(csv_path)
    if "clip_path" not in df.columns:
        df["clip_path"] = ""
    # persist CSV immediately after ensuring clip_path column exists
    try:
        df.to_csv(csv_path, index=False)
    except Exception as e:
        print(f"Warning: failed to persist CSV after adding clip_path column: {e}")

    # If the CSV doesn't yet contain subclip_index (older runs), optionally split here as a fallback
    if (max_clip_seconds and int(max_clip_seconds) > 0) and ("subclip_index" not in df.columns or df["subclip_index"].isna().all()):
        try:
            df = _split_long_scenes(df, max_seconds=int(max_clip_seconds))
            # Persist the split back so downstream steps are consistent
            df.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"Warning: failed to split long scenes in {csv_path}: {e}")

    def _to_bool(v):
        if isinstance(v, bool):
            return v
        if pd.isna(v):
            return False
        return str(v).strip().lower() in {"true", "1", "yes", "y", "t"}

    if "is_react" not in df.columns or "is_playing" not in df.columns:
        print(f"Skipping {csv_path}: required columns missing")
        return out_dir, 0

    mask = df["is_react"].map(_to_bool) & df["is_playing"].map(_to_bool)
    selected = df[mask].reset_index(drop=True)

    if os.path.isdir(out_dir):
        existing_clips = [f for f in os.listdir(out_dir) if f.lower().endswith(".mp4")]
        if len(existing_clips) == len(selected):
            print(f"Skipping cutting for {video_path}: {len(existing_clips)} clips already present")
            return out_dir, 0

    os.makedirs(out_dir, exist_ok=True)
    print(f"Cutting clips for {csv_path} -> {out_dir}")

    written = 0
    for idx, row in selected.iterrows():
        start = str(row["start_time"]).strip()
        end = str(row["end_time"]).strip()
        sub_i = int(row.get("subclip_index", 0)) if pd.notna(row.get("subclip_index", 0)) else 0
        # Keep deterministic names that reflect splitting
        out_name = f"{idx:04d}_{sub_i:02d}.mp4" if sub_i > 0 else f"{idx:04d}.mp4"
        out_path = os.path.join(out_dir, out_name)
        if os.path.exists(out_path):
            print(f"  Skipping existing clip {out_path}")
            row_sel_idx = selected.index[idx]
            df.loc[row_sel_idx, "clip_path"] = out_path
            if "split_start_time" not in df.columns:
                df["split_start_time"] = ""
            if "split_end_time" not in df.columns:
                df["split_end_time"] = ""
            df.loc[row_sel_idx, "split_start_time"] = start
            df.loc[row_sel_idx, "split_end_time"] = end
            # persist CSV immediately after updating clip_path
            try:
                df.to_csv(csv_path, index=False)
            except Exception as e:
                print(f"  Warning: failed to persist CSV for existing clip {out_path}: {e}")
            continue

        vcodec_args = _ffmpeg_video_codec_args()
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-ss", start, "-to", end, "-i", input_video,
            *vcodec_args,
            "-c:a", "aac", "-b:a", "192k",
            "-y", out_path,
        ]
        try:
            subprocess.run(cmd, check=True)
            print(f"  Wrote {out_path}")
            written += 1
            row_sel_idx = selected.index[idx]
            df.loc[row_sel_idx, "clip_path"] = out_path
            if "split_start_time" not in df.columns:
                df["split_start_time"] = ""
            if "split_end_time" not in df.columns:
                df["split_end_time"] = ""
            df.loc[row_sel_idx, "split_start_time"] = start
            df.loc[row_sel_idx, "split_end_time"] = end
            # persist CSV immediately after updating clip_path
            try:
                df.to_csv(csv_path, index=False)
            except Exception as e:
                print(f"  Warning: failed to persist CSV after writing {out_path}: {e}")
        except subprocess.CalledProcessError as e:
            print(f"  Failed to cut {out_path}: {e}")

    # Save updated DataFrame with clip_path column back to CSV
    df.to_csv(csv_path, index=False)

    return out_dir, written


def process_from_manifest(
    manifest_csv: str,
    *,
    use_batch_face_extraction: bool = True,
    mode: str = "auto",
    max_clip_seconds: int | None = None,
    label: str = "General pop-culture React",
    fast_skip_local: bool = False,
) -> None:
    """
    Process pop-culture reaction videos listed in a CSV manifest.

    When `fast_skip_local=True`, skip decisions are made purely from local filesystem artifacts and the manifest is not modified.
    When writing progress to the manifest (default), progress columns are namespaced with the label key (e.g., `general_pop_culture_react_df_path`).

    Expected columns in the CSV:
    - selected       : bool -> only rows with True are processed; others are ignored
    - local_path_low : str  -> (present/valid only for selected=True rows) used for detect_scene_transitions_to_df()
    - local_path     : str  -> (present/valid only for selected=True rows) used for cut_reaction_clips_from_csv()
    - subtitle_path  : str  -> (optional; present/valid only for selected=True rows)

    Side effects (on selected=True rows only):
    - Writes the scene detection CSV (df_path) next to local_path_low's directory.
    - Cuts clips and writes them under a sibling 'clips' directory (same behavior as folder mode).
    - Updates the manifest with namespaced columns for this label: `{label_key}_df_path` and `{label_key}_out_dir`.

    - If `mode == "face-only"`, skip scene detection, cutting, and other heavy stages; **only** run face extraction on existing clips (preferring masked clips when available). The CSV next to `local_path_low` will still be updated with `face_out_dir`.
    """
    if not os.path.exists(manifest_csv):
        print(f"Error: Manifest CSV does not exist: {manifest_csv}")
        sys.exit(1)

    dfm = pd.read_csv(manifest_csv)

    label_key = _label_key(label)
    col_df_path = f"{label_key}_df_path"
    col_out_dir = f"{label_key}_out_dir"

    def _maybe_persist_manifest():
        if fast_skip_local:
            return
        try:
            _safe_merge_persist_manifest(
                manifest_csv,
                dfm,
                cols_to_update=[col_df_path, col_out_dir],
                key_col="video_link",
            )
        except Exception as e:
            print(f"Warning: failed to safely persist manifest (merge): {e}")

    # --- Validate presence of 'selected' column ---
    if "selected" not in dfm.columns:
        print("Error: Manifest must include a 'selected' column (True/False).")
        sys.exit(1)

    # Enforce: only process label == 'General pop-culture React'
    if "label" not in dfm.columns:
        print("Error: Manifest must include a 'label' column to enforce Pop-culture-only processing.")
        return

    # Ensure output/progress columns exist (only when not in fast-skip mode)
    if not fast_skip_local:
        for col in [col_df_path, col_out_dir]:
            if col not in dfm.columns:
                dfm[col] = ""

    # Helper: robust boolean cast (mirrors helper used elsewhere)
    def _to_bool(v):
        if isinstance(v, bool):
            return v
        if pd.isna(v):
            return False
        return str(v).strip().lower() in {"true", "1", "yes", "y", "t"}

    # Filter to rows explicitly selected == True AND label matches exactly
    sel_mask = dfm["selected"].map(_to_bool)
    df_sel = dfm[sel_mask & (dfm["label"].astype(str).str.strip() == label)].copy()
    if df_sel.empty:
        print(f"No rows marked selected=True and label='{label}'; nothing to process.")
        return

    print(f"Processing {len(df_sel)} selected rows (out of {len(dfm)} total).")
    if fast_skip_local:
        print(f"Fast-skip mode enabled (local-only). Manifest will not be modified. Label filter: '{label}' [{_label_key(label)}]")
    else:
        print(f"Manifest progress columns will be namespaced under '{_label_key(label)}' and merged atomically (key=video_link).")
        print("Manifest writes use safe merge on key 'video_link' with atomic replace; other labels' columns are preserved.")
    if str(mode).strip().lower() == "face-only":
        print("Running in FACE-ONLY mode: only talking head extraction will be executed.")
        # Prepare optional per-clip face extraction report
        report_rows: list[dict] = []
        _report_min_ok_bytes = 16384

    # Cache env configs
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION")
    model_id = os.environ.get("GOOGLE_CLOUD_MODEL_ID")

    # Collect all clips for global batch processing
    all_clips_for_face_extraction = []

    # Iterate only over selected rows; leave unselected rows untouched
    for idx, row in df_sel.iterrows():
        # Guard: local_path_low/local_path should exist for selected rows
        low_path = str(row.get("local_path_low", "")).strip()
        hi_path = str(row.get("local_path", "")).strip()

        if not low_path or not os.path.exists(low_path):
            print(f"[{idx}] Skipping: selected row missing/invalid local_path_low: {low_path}")
            continue
        if not hi_path or not os.path.exists(hi_path):
            print(f"[{idx}] Skipping: selected row missing/invalid local_path: {hi_path}")
            continue

        # Pre-compute expected artifact locations (for local fast-skip)
        low_dir = os.path.dirname(low_path)
        low_stem = os.path.splitext(os.path.basename(low_path))[0]
        hi_dir = os.path.dirname(hi_path)
        hi_stem = os.path.splitext(os.path.basename(hi_path))[0]

        expected_df_path = os.path.join(low_dir, f"{low_stem}_scene_transitions.csv")
        root_root, label_folder = _infer_root_and_label(hi_dir)
        label_folder = _sanitize_filename(label_folder)
        expected_out_dir = os.path.join(root_root, 'clips', label_folder, f"{hi_stem}_clips")

        # === FACE-ONLY fast path ===
        if str(mode).strip().lower() == "face-only":
            # For face-only mode, we only run talking head extraction. We prefer masked clips
            # if they exist; otherwise, fall back to the regular clips.
            videos_for_row: List[str] = []

            # Resolve masked and clips directories for this video
            try:
                root_root, label_folder = _infer_root_and_label(hi_dir)
                label_folder = _sanitize_filename(label_folder)
                hi_stem = os.path.splitext(os.path.basename(hi_path))[0]
                masked_dir = os.path.join(root_root, 'masked', label_folder, f"{hi_stem}_masked")
            except Exception:
                masked_dir = ""

            def _collect_mp4s(d: str) -> List[str]:
                try:
                    return sorted([
                        os.path.join(d, f)
                        for f in os.listdir(d)
                        if f.lower().endswith('.mp4')
                    ])
                except Exception:
                    return []

            # Prefer masked clips if present; else use normal clips
            if masked_dir and os.path.isdir(masked_dir):
                videos_for_row = _collect_mp4s(masked_dir)
            if not videos_for_row and os.path.isdir(expected_out_dir):
                videos_for_row = _collect_mp4s(expected_out_dir)

            if not videos_for_row:
                print(f"[{idx}] Face-only: no clips found (masked or clips) for {hi_path}; skipping row.")
                continue

            print(f"[{idx}] Face-only: extracting faces for {len(videos_for_row)} clip(s).")
            try:
                # Process per-row so that df_path updates the correct CSV for this video
                face_outputs = talking_head_extraction_batch(
                    videos_for_row,
                    df_path=expected_df_path,
                    skip_if_exists=True,
                    fast_mode=True,
                    max_workers=2,
                )
                # Record batch results for the face-only report
                if 'report_rows' in locals():
                    for clip_path, face_out_path in zip(videos_for_row, face_outputs or []):
                        try:
                            size = os.path.getsize(face_out_path) if face_out_path and os.path.exists(face_out_path) else 0
                        except Exception:
                            size = 0
                        status = "success" if size >= _report_min_ok_bytes else "fail"
                        report_rows.append({
                            "clip_path": clip_path,
                            "face_out_path": face_out_path or "",
                            "status": status,
                            "size_bytes": int(size),
                        })
            except Exception as e:
                print(f"[{idx}] Face-only: batch face extraction failed: {e}")
                # Fallback: try individual processing and still record report entries
                if 'report_rows' in locals():
                    for clip in videos_for_row:
                        try:
                            single_out = talking_head_extraction(
                                clip,
                                df_path=expected_df_path,
                                skip_if_exists=True,
                                fast_mode=True,
                            )
                            try:
                                size = os.path.getsize(single_out) if single_out and os.path.exists(single_out) else 0
                            except Exception:
                                size = 0
                            status = "success" if size >= _report_min_ok_bytes else "fail"
                            report_rows.append({
                                "clip_path": clip,
                                "face_out_path": single_out or "",
                                "status": status,
                                "size_bytes": int(size),
                            })
                        except Exception as e2:
                            print(f"[{idx}] Face-only: individual failed for {clip}: {e2}")
                            report_rows.append({
                                "clip_path": clip,
                                "face_out_path": "",
                                "status": "fail",
                                "size_bytes": 0,
                            })
            # Face-only mode should not run any other steps for this row
            continue

        if fast_skip_local:
            has_df = os.path.exists(expected_df_path)
            has_clips = os.path.isdir(expected_out_dir) and any(f.lower().endswith('.mp4') for f in os.listdir(expected_out_dir))
            if has_df and has_clips:
                print(f"[{idx}] ⏩ Fast-skip (local artifacts present): {hi_stem}")
                continue

        # 0) Visibility pre-check on the low-res video: skip if the pop-culture source is masked/blurred/omitted
        try:
            is_visible, reason = check_pop_visibility(
                low_path,
                project=project,
                location=location or "us-central1",
                model_id=(model_id or os.environ.get("GOOGLE_CLOUD_MODEL_ID") or "gemini-2.0-flash"),
            )
        except Exception as e:
            print(f"[{idx}] Visibility check failed ({e}); proceeding by default")
            is_visible, reason = True, "exception"

        if not is_visible:
            print(f"[{idx}] Skipping {hi_path}: pop-culture source not visible (reason: {reason})")
            _maybe_persist_manifest()
            continue

        # New: lightweight feasibility check — skip if too chaotic to cut clean segments
        try:
            feasible, why = check_pop_cut_feasibility(
                low_path,
                project=project,
                location=location or "us-central1",
                model_id=(model_id or os.environ.get("GOOGLE_CLOUD_MODEL_ID") or "gemini-2.0-flash"),
            )
        except Exception as e:
            feasible, why = True, f"exception: {e}"

        if not feasible:
            print(f"[{idx}] Skipping {hi_path}: not feasible to cut clean segments (reason: {why})")
            _maybe_persist_manifest()
            continue

        df_path = os.path.join(low_dir, f"{low_stem}_scene_transitions.csv")
        if not fast_skip_local:
            dfm.at[idx, col_df_path] = df_path
            _maybe_persist_manifest()

        # 1) Scene detection on the low-res file (skip if CSV already exists)
        if os.path.exists(df_path):
            print(f"[{idx}] Skipping scene detection; CSV exists: {df_path}")
        else:
            # For manual mode, prompt for start/end at runtime; otherwise, use manifest columns or auto-detect
            if mode == "manual":
                print(f"[{idx}] Manual mode selected. Please input start and end time for {hi_path}")
                manual_start = input("Enter start time (HH:MM:SS): ").strip()
                manual_end   = input("Enter end time (HH:MM:SS): ").strip()
                if not (manual_start and manual_end):
                    print(f"[{idx}] ERROR: manual mode requires both start and end times. Skipping.")
                    continue
                print(f"[{idx}] Using MANUAL pop-culture reaction window: {manual_start} -> {manual_end}")
                df_detect = pd.DataFrame([{
                    "scene_number": 1,
                    "start_time": manual_start,
                    "end_time": manual_end,
                    "description": "pop-culture watch-along",
                    "is_react": True,
                    "is_playing": True,
                }])
                # Apply optional duration threshold splitting
                if max_clip_seconds and int(max_clip_seconds) > 0:
                    df_detect = _split_long_scenes(df_detect, max_seconds=int(max_clip_seconds))
                df_detect.to_csv(df_path, index=False)
            else:  # auto
                start_candidates = [str(row.get(k, "")).strip() for k in ("movie_start", "react_start", "start_time")]
                end_candidates   = [str(row.get(k, "")).strip() for k in ("movie_end", "react_end", "end_time")]
                manual_start = next((v for v in start_candidates if v), "")
                manual_end   = next((v for v in end_candidates if v), "")
                if manual_start and manual_end:
                    print(f"[{idx}] Using manifest-provided window: {manual_start} -> {manual_end}")
                    df_detect = pd.DataFrame([{
                        "scene_number": 1,
                        "start_time": manual_start,
                        "end_time": manual_end,
                        "description": "pop-culture watch-along",
                        "is_react": True,
                        "is_playing": True,
                    }])
                    # Apply optional duration threshold splitting
                    if max_clip_seconds and int(max_clip_seconds) > 0:
                        df_detect = _split_long_scenes(df_detect, max_seconds=int(max_clip_seconds))
                else:
                    print(f"[{idx}] Detecting pop-culture reaction segments for {low_path}")
                    try:
                        df_detect = detect_scene_transitions_to_df(
                            video_path=low_path,
                            project=project,
                            location=location,
                            model_id=model_id,
                        )
                        # Apply optional duration threshold splitting
                        if max_clip_seconds and int(max_clip_seconds) > 0:
                            df_detect = _split_long_scenes(df_detect, max_seconds=int(max_clip_seconds))
                    except Exception as e:
                        print(f"[{idx}] Pop-culture scene detection failed for {low_path}: {e}")
                        continue
                df_detect.to_csv(df_path, index=False)

        # 2) Cut clips from the high-res file using the detection CSV
        out_dir, written = cut_reaction_clips_from_csv(
            root_dir=hi_dir,
            video_path=os.path.basename(hi_path),
            csv_path=df_path,
            max_clip_seconds=max_clip_seconds,
        )
        if not fast_skip_local:
            dfm.at[idx, col_out_dir] = out_dir
            _maybe_persist_manifest()
        print(f"[{idx}] Cut {written} clips to {out_dir}")

        # Collect the emitted clips for downstream steps
        passing_clips = []
        if not os.path.isdir(out_dir):
            print(f"[{idx}] No out_dir found (no clips?): {out_dir}")
        else:
            for clip_file in os.listdir(out_dir):
                if not clip_file.endswith(".mp4"):
                    continue
                full_clip = os.path.join(out_dir, clip_file)
                passing_clips.append(full_clip)

        # === Parallel bounding-box extraction for passing clips ===
        prepared_clips_for_face = []
        if passing_clips:
            print(f"[{idx}] Running parallel bounding-box extraction for {len(passing_clips)} clips...")
            bbox_map = bounding_box_extraction_batch(
                passing_clips,
                df_path=df_path,
                project=project,
                location=location,
                model_id=model_id or os.environ.get("GOOGLE_CLOUD_MODEL_ID") or "gemini-2.0-flash",
                max_workers=6,
            )
        else:
            bbox_map = {}

        for full_clip in passing_clips:
            x1, y1, x2, y2, masked_clip = bbox_map.get(full_clip, (0, 0, 0, 0, ""))
            print(f"Bounding box extraction: {x1}, {y1}, {x2}, {y2}; masked={masked_clip}")
            prepared_clips_for_face.append(masked_clip if masked_clip else full_clip)

        # Collect clips for global batch processing or process immediately (no duplicate bbox calls)
        if use_batch_face_extraction:
            for clip_path in prepared_clips_for_face:
                all_clips_for_face_extraction.append((clip_path, df_path))
        else:
            if prepared_clips_for_face:
                print(f"[{idx}] Batch processing {len(prepared_clips_for_face)} clips for face extraction")
                talking_head_extraction_batch(
                    prepared_clips_for_face,
                    df_path=df_path,
                    fast_mode=True,
                    max_workers=6,
                )

        # Persist manifest after each selected row to avoid losing progress
        if not fast_skip_local:
            _maybe_persist_manifest()

    # If running face-only mode, write the per-clip report and exit early
    if str(mode).strip().lower() == "face-only":
        try:
            if 'report_rows' in locals() and report_rows:
                report_df = pd.DataFrame(report_rows, columns=["clip_path","face_out_path","status","size_bytes"])
                report_path = os.path.splitext(os.path.abspath(manifest_csv))[0] + f"_{_label_key(label)}_face_report.csv"
                with _locked_open(report_path, "w") as f:
                    report_df.to_csv(f, index=False)
                print(f"[Face-only] Wrote face report: {report_path} ({len(report_df)} rows)")
            else:
                print("[Face-only] No clips processed; no face report written.")
        except Exception as e:
            print(f"[Face-only] Failed to write face report CSV: {e}")
        return

    # Global batch processing for face extraction (selected rows only)
    if use_batch_face_extraction and all_clips_for_face_extraction:
        print(f"\n=== Global Batch Face Extraction ===")
        print(f"Processing {len(all_clips_for_face_extraction)} clips across selected videos")

        # Group clips by df_path for efficient CSV updates
        clips_by_csv = {}
        for full_clip, df_path in all_clips_for_face_extraction:
            if df_path not in clips_by_csv:
                clips_by_csv[df_path] = []
            clips_by_csv[df_path].append(full_clip)

        for df_path, clips in clips_by_csv.items():
            print(f"Batch processing {len(clips)} clips for CSV: {df_path}")
            try:
                talking_head_extraction_batch(
                    clips,
                    df_path=df_path,
                    fast_mode=True,
                    providers="TensorrtExecutionProvider,CUDAExecutionProvider",
                    trt_cache_dir=os.path.join(os.path.dirname(df_path), ".trt_cache"),
                    max_workers=6,
                )
            except Exception as e:
                print(f"Warning: global batch face extraction failed for {df_path}: {e}")
                for clip in clips:
                    try:
                        talking_head_extraction(clip, df_path=df_path, fast_mode=True)
                    except Exception as e2:
                        print(f"Individual fallback also failed for {clip}: {e2}")

        print(f"=== Global Batch Face Extraction Complete ===\n")

    if not fast_skip_local:
        _maybe_persist_manifest()


def main():
    """Main function to process videos either from a manifest CSV or by scanning a directory."""
    parser = argparse.ArgumentParser(description="Process pop-culture reaction videos/watch-alongs")
    parser.add_argument(
        "--manifest",
        help="CSV manifest containing video paths (uses local_path_low for detection and local_path for cutting)",
        default=None,
    )
    parser.add_argument(
        "--label",
        default="General pop-culture React",
        help="Only process rows whose label exactly matches this string (prevents cross-label interference)",
    )
    parser.add_argument(
        "--fast-skip-local",
        action="store_true",
        help="Make skip decisions purely from local artifacts and do not write to the manifest",
    )
    parser.add_argument(
        "--mode", choices=["manual","auto", "face-only"], default="auto",
        help="Pop-culture segmentation mode. 'manual' requires start/end per video in the manifest; 'auto' will use manifest overrides if present, otherwise run Gemini detection."
    )
    parser.add_argument(
        "--max-clip-seconds",
        type=int,
        default=int(os.environ.get("MAX_CLIP_SECONDS", "0")),
        help="If >0, split any detected scene longer than this many seconds into subclips to avoid OOM (applies to cutting/masking/stimuli/face).",
    )
    parser.add_argument(
        "root_dir",
        nargs="?",
        help="(Legacy mode) Root directory containing the video files to process",
        default=None,
    )
    args = parser.parse_args()

    if args.manifest:
        print(f"Processing videos from manifest: {args.manifest}")
        process_from_manifest(
            args.manifest,
            use_batch_face_extraction=True,
            mode=args.mode,
            max_clip_seconds=args.max_clip_seconds,
            label=args.label,
            fast_skip_local=args.fast_skip_local,
        )
        return

    # Legacy folder-scanning mode (kept for backward compatibility)
    if not args.root_dir:
        print("Error: either provide --manifest /path/to/manifest.csv or a root_dir")
        sys.exit(1)

    root_dir = args.root_dir

    # Validate that the root directory exists
    if not os.path.exists(root_dir):
        print(f"Error: Root directory does not exist: {root_dir}")
        sys.exit(1)

    if not os.path.isdir(root_dir):
        print(f"Error: Root directory is not a directory: {root_dir}")
        sys.exit(1)

    # Enforce: only process folders whose inferred label is 'General pop-culture React'
    _root_root, _label = _infer_root_and_label(root_dir)
    if not _is_pop_label(_label):
        print(f"Skipping legacy folder: inferred label '{_label}' is not 'General pop-culture React'.")
        return

    print(f"Processing videos in directory: {root_dir}")

    # walk through all videos in folder
    for video_path in os.listdir(root_dir):
        if not video_path.endswith(".mp4"):
            continue

        df_path = os.path.join(root_dir, f"{video_path.replace('.mp4', '')}_scene_transitions.csv")
        if os.path.exists(df_path):
            print(f"Skipping scene detection for {video_path} because CSV already exists")
        else:
            print(f"Processing scene detection for {video_path}")
            df_detect = detect_scene_transitions_to_df(
                video_path=os.path.join(root_dir, video_path),
                project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
                location=os.environ.get("GOOGLE_CLOUD_LOCATION"),
                model_id=os.environ.get("GOOGLE_CLOUD_MODEL_ID")
            )
            # Optional split in legacy mode
            if args.max_clip_seconds and args.max_clip_seconds > 0:
                df_detect = _split_long_scenes(df_detect, max_seconds=args.max_clip_seconds)
            df_detect.to_csv(df_path, index=False)

        # Cut clips from CSVs using helper
        out_dir, written = cut_reaction_clips_from_csv(root_dir=root_dir, video_path=video_path, csv_path=df_path, max_clip_seconds=args.max_clip_seconds)
        print(f"Cut {written} clips from {video_path} to {out_dir}")

        # Collect all clips (Pop-culture: accept all)
        _clips_for_bbox = []
        for clip_path in os.listdir(out_dir):
            if not clip_path.endswith(".mp4"):
                continue
            print(f"Processing {clip_path}")
            full_clip = os.path.join(out_dir, clip_path)
            _clips_for_bbox.append(full_clip)

        # After the loop, batch bbox and then stage for face extraction
        if _clips_for_bbox:
            print(f"Running parallel bounding-box extraction for {len(_clips_for_bbox)} clips")
            bbox_map = bounding_box_extraction_batch(
                _clips_for_bbox,
                df_path=df_path,
                project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
                location=os.environ.get("GOOGLE_CLOUD_LOCATION") or "us-central1",
                model_id=os.environ.get("GOOGLE_CLOUD_MODEL_ID") or "gemini-2.0-flash",
                max_workers=6,
            )

            _clips_for_batch_face = []
            for full_clip in _clips_for_bbox:
                x1, y1, x2, y2, masked_clip = bbox_map.get(full_clip, (0, 0, 0, 0, ""))
                print(f"Bounding box extraction: {x1}, {y1}, {x2}, {y2}; masked={masked_clip}")
                _clips_for_batch_face.append(masked_clip if masked_clip else full_clip)
            # Run talking head extraction
            if _clips_for_batch_face:
                print(f"Batch processing {len(_clips_for_batch_face)} clips for face extraction")
                talking_head_extraction_batch(
                    _clips_for_batch_face,
                    df_path=df_path,
                    fast_mode=True,
                    max_workers=6,
                )

if __name__ == "__main__":
    main()
