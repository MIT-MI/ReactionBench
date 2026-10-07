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
import statistics
from typing import List, Optional

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from google import genai
from google.genai.types import GenerateContentConfig, Part
from pydantic import BaseModel, Field, field_validator


# --- Utility: ensure the face clip keeps the original audio track ---
def _mux_original_audio_into_face(face_video_path: str, audio_source_path: str, *, min_ok_bytes: int = 16384) -> None:
    """Mux the audio from `audio_source_path` into `face_video_path` if the face clip lacks audio.
    Keeps video stream as-is (copy) and re-encodes audio to AAC. Safe no-op if either file missing.
    """
    try:
        if not (os.path.exists(face_video_path) and os.path.getsize(face_video_path) >= max(1, int(min_ok_bytes))):
            return
        if not os.path.exists(audio_source_path):
            return
        # Probe whether face_video_path already has an audio stream
        has_audio = False
        try:
            prob = subprocess.run([
                'ffprobe','-v','error','-select_streams','a','-show_entries','stream=index','-of','csv=p=0',
                face_video_path
            ], check=True, capture_output=True, text=True)
            has_audio = bool((prob.stdout or '').strip())
        except Exception:
            has_audio = False
        if has_audio:
            return  # already has audio
        # Mux audio from the source into face clip
        tmp_out = face_video_path + '.tmp_audio.mp4'
        try:
            subprocess.run([
                'ffmpeg','-hide_banner','-loglevel','error','-nostdin',
                '-i', face_video_path, '-i', audio_source_path,
                '-map','0:v:0','-map','1:a:0?',
                '-c:v','copy','-c:a','aac','-b:a','160k',
                '-shortest','-movflags','+faststart','-y', tmp_out
            ], check=True)
            if os.path.exists(tmp_out) and os.path.getsize(tmp_out) >= max(1, int(min_ok_bytes)):
                os.replace(tmp_out, face_video_path)
            else:
                try:
                    os.remove(tmp_out)
                except Exception:
                    pass
        except Exception:
            try:
                if os.path.exists(tmp_out):
                    os.remove(tmp_out)
            except Exception:
                pass
    except Exception:
        return

def _ensure_talking_head_deps() -> None:
    """Preflight check for talking_head Python dependencies."""
    missing = []
    reasons = []

    try:
        import cv2  # type: ignore
        if not hasattr(cv2, "CV_8U"):
            reasons.append("`cv2` imported but is missing `CV_8U` (OpenCV not fully installed).")
            missing.append("opencv-python-headless>=4.8")
    except Exception as e:  # pragma: no cover - defensive
        reasons.append(f"Failed to import OpenCV: {e}")
        missing.append("opencv-python-headless>=4.8")

    try:
        import albumentations  # type: ignore  # noqa: F401
    except Exception as e:  # pragma: no cover - defensive
        reasons.append(f"Failed to import albumentations: {e}")
        missing.append("albumentations")

    try:
        import insightface  # type: ignore  # noqa: F401
    except Exception as e:  # pragma: no cover - defensive
        reasons.append(f"Failed to import insightface: {e}")
        missing.append("insightface>=0.7")

    if missing:
        detail = "; ".join(reasons) if reasons else "missing required packages."
        req_path = os.path.join(os.path.dirname(__file__), "talking_head", "requirements.txt")
        raise RuntimeError(
            "talking_head dependencies are unavailable. "
            f"Install them with `pip install -r {req_path}` inside the active environment. "
            f"Details: {detail}"
        )


def _resolve_talking_head_root() -> str:
    """Locate the talking_head tool in a user-agnostic way."""
    candidates = []
    env_override = os.environ.get("TALKING_HEAD_ROOT")
    if env_override:
        candidates.append(env_override)
    candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "talking_head"))

    missing = []
    denied = []

    for candidate in candidates:
        if not candidate:
            continue
        candidate = os.path.abspath(candidate)
        if not os.path.exists(candidate):
            missing.append(candidate)
            continue
        if not os.path.isdir(candidate):
            missing.append(candidate)
            continue
        if not os.access(candidate, os.R_OK | os.X_OK):
            denied.append(candidate)
            continue
        return candidate

    if denied:
        raise PermissionError(
            f"talking_head tool exists but is not accessible. Check permissions for: {', '.join(denied)}"
        )
    raise FileNotFoundError(
        "Unable to locate talking_head tool. Set TALKING_HEAD_ROOT or ensure data_pipeline/02_align/talking_head exists."
    )


class Scene(BaseModel):
    scene_number: int = Field(..., description="1,2,3,... sequential id")
    start_time: str = Field(..., description="HH:MM:SS")
    end_time: str = Field(..., description="HH:MM:SS")
    description: str = Field(..., description="short scene summary")
    is_react: bool = Field(False, description="true if the streamer is actively reacting to gameplay")
    is_playing: bool = Field(..., description="true if gameplay is ongoing (match/live play/highlights)")

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


_BASE_INSTRUCTIONS_GAME = """You are a **multimodal Game Stream Reaction Segment Detector**. The input is typically a game livestream/VOD where a streamer reacts while playing or watching gameplay.

**Task:** Identify the continuous **reaction window(s)** where the streamer is actively reacting to gameplay (watch‑along or live play). Prefer **one long segment** from the start of reacting to the end, unless there is a true intermission.

**When to cut:**
- Boundaries only at **(1) reaction starts** (the streamer begins actively playing/watching) and **(2) reaction ends** (end screen/outro/post‑game talk unrelated to the play).
- Ignore editor jump cuts, short tangents, and overlays that do not stop the watch‑along.

**Common layouts:**
- **Layout A (Face‑full / Game‑window small):** Person fills most of the frame; game is a smaller window. (Use game window bbox for `stimuli` crop, mask that game region on the full frame for `masked`, and face pipeline can run on the original or masked‑out clip.)
- **Layout B (Game‑full / Face‑cam small):** Game fills the full frame; face‑cam is a small window at a corner. (Detect **person bbox**; output the **person crop** into the `masked` folder; use that cropped clip for the `face` pipeline; and create `stimuli` by masking the person region out of the full frame.)

**Cues (use audio & visuals):**
- Gameplay UI/HUD, minimap, health bars, input sounds; streamer speech synced with on‑screen events; picture‑in‑picture (PiP) windows.
- Non‑reaction cues: intro banter/sponsor/readme, post‑game debrief not tied to footage.

**Timestamp rules:**
- Prefer a single continuous segment; split only on clear breaks/intermissions.
- Timestamps must be within video duration.

**Output format (JSON array; only reaction segments):**
Each object must include:
- `scene_number` (int; 1,2,3…)
- `start_time` (HH:MM:SS)
- `end_time` (HH:MM:SS)
- `description` (short summary like "playing match / reacting to gameplay highlights")
- `is_react` (boolean; **always `true`**)
- `is_playing` (boolean; **set `true`** if gameplay is ongoing, even if music/OST/ducked).
"""


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



# === Module-level helpers for time conversion ===
def _hms_to_seconds(t: str) -> float:
    parts = [int(p) for p in str(t).strip().split(":")]
    if len(parts) == 2:
        h, m, s = 0, parts[0], parts[1]
    elif len(parts) == 3:
        h, m, s = parts
    else:
        raise ValueError(f"Invalid time format: {t}")
    return int(h) * 3600 + int(m) * 60 + int(s)

def _seconds_to_hms(sec: float) -> str:
    s = max(0, int(round(sec)))
    h = s // 3600
    m = (s % 3600) // 60
    s = s % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


# === Scene transition detection helpers ===

def _detect_scene_transitions_single(
    video_path: str,
    *,
    project: Optional[str] = None,
    location: str = "us-central1",
    model_id: str = "gemini-2.0-flash",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_output_tokens: int = 8000,
) -> pd.DataFrame:
    # Initialize client (same as notebook: genai.Client(vertexai=True, ...))
    project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise ValueError("Project ID not provided. Pass 'project' or set the environment variable GOOGLE_CLOUD_PROJECT.")

    client = genai.Client(vertexai=True, project=project, location=location)

    # Assemble input parts (Game only)
    parts: List[Part] = [
        Part.from_text(text=_BASE_INSTRUCTIONS_GAME),
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


def detect_scene_transitions_to_df(
    video_path: str,
    *,
    project: Optional[str] = None,
    location: str = "us-central1",
    model_id: str = "gemini-2.0-flash",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_output_tokens: int = 8000,
    max_payload_bytes: int = 480_000_000,
    chunk_overlap_s: float = 0.0,
) -> pd.DataFrame:
    """
    Size‑aware wrapper: if `video_path` exceeds `max_payload_bytes`, split into **temporal chunks**
    and run detection per chunk. We **do not** re‑assemble the chunks into a big video; we only
    stitch the **timestamps** in the returned DataFrame by adding the proper offsets.

    Chunks are created with ffmpeg stream‑copy (`-c copy`) for speed and fidelity.
    """
    try:
        fsize = os.path.getsize(video_path)
    except Exception:
        fsize = 0

    if fsize <= max_payload_bytes or max_payload_bytes <= 0:
        # Small enough → single pass
        return _detect_scene_transitions_single(
            video_path,
            project=project,
            location=location,
            model_id=model_id,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
        )

    # Need chunking
    duration = _ffprobe_video_duration_seconds(video_path)
    if duration <= 0:
        # Unknown duration: fallback to single (will likely fail upstream with a clear error)
        return _detect_scene_transitions_single(
            video_path,
            project=project,
            location=location,
            model_id=model_id,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
        )

    # Estimate number of chunks based on file size; leave ~10% headroom per chunk
    import math
    chunks = max(1, int(math.ceil(fsize / (max_payload_bytes * 0.9))))
    chunk_len = duration / chunks
    print(f"Scene detection: splitting '{os.path.basename(video_path)}' into {chunks} chunk(s) of ~{chunk_len:.1f}s each due to size {fsize/1e6:.1f} MB > limit {max_payload_bytes/1e6:.1f} MB. No re‑merge will be performed.")

    tmp_root = tempfile.mkdtemp(prefix="scene_chunks_")
    made = []
    try:
        # Create chunks on disk (stream‑copy)
        for i in range(chunks):
            start = max(0.0, i * chunk_len - (chunk_overlap_s if i > 0 else 0.0))
            # Ensure the last chunk runs to the end
            end = duration if i == chunks - 1 else min(duration, (i + 1) * chunk_len + (chunk_overlap_s if i < chunks - 1 else 0.0))
            seg_len = max(0.01, end - start)
            cpath = os.path.join(tmp_root, f"chunk_{i:03d}.mp4")
            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
                "-ss", f"{start:.3f}", "-i", video_path,
                "-t", f"{seg_len:.3f}",
                "-map", "0:v:0", "-map", "0:a:0?",
                "-c", "copy",
                "-movflags", "+faststart",
                "-y", cpath,
            ]
            subprocess.run(cmd, check=True)
            made.append((cpath, start))

        # Run detection per chunk and accumulate with offset
        dfs = []
        for cpath, offset in made:
            try:
                df_chunk = _detect_scene_transitions_single(
                    cpath,
                    project=project,
                    location=location,
                    model_id=model_id,
                    temperature=temperature,
                    top_p=top_p,
                    max_output_tokens=max_output_tokens,
                )
                if not df_chunk.empty:
                    # Offset timestamps by chunk start
                    df_chunk = df_chunk.copy()
                    df_chunk["start_time"] = df_chunk["start_time"].map(lambda t: _seconds_to_hms(_hms_to_seconds(str(t)) + offset))
                    df_chunk["end_time"]   = df_chunk["end_time"].map(lambda t: _seconds_to_hms(_hms_to_seconds(str(t)) + offset))
                    dfs.append(df_chunk)
            except Exception as e:
                print(f"Warning: scene detection failed on {os.path.basename(cpath)}: {e}")

        if not dfs:
            return pd.DataFrame(columns=["scene_number","start_time","end_time","description","is_react","is_playing"])  # empty

        # Concatenate and renumber scene_number
        out = pd.concat(dfs, ignore_index=True)
        out = out.sort_values(by=["start_time"]).reset_index(drop=True)
        out["scene_number"] = range(1, len(out) + 1)
        return out[["scene_number","start_time","end_time","description","is_react","is_playing"]]
    finally:
        try:
            shutil.rmtree(tmp_root, ignore_errors=True)
        except Exception:
            pass





def _sanitize_filename(filename: str) -> str:
    """Mirror the manifest's sanitize: remove illegal chars, replace spaces, cap length."""
    sanitized = re.sub(r'[\\/*?:"<>|]', "", filename)
    sanitized = sanitized.replace(" ", "_")
    return sanitized[:100]


# Helper: check if label is exactly 'game'
def _is_game_label(v: str) -> bool:
    """Return True iff the provided label is exactly 'game' (case-insensitive)."""
    if v is None:
        return False
    s = str(v).strip().lower()
    return s == "game"

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
    Choose a safe and size‑efficient set of FFmpeg video-encoding args.

    Preference: libx264 (CRF 26, veryfast), fallbacks: h264_nvenc (CQ 28), libx265 (CRF 30),
    and as a last resort MPEG‑4 with qscale 7. These defaults are tuned to *reduce* size vs source.
    """
    try:
        encoders = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            check=True, capture_output=True, text=True
        ).stdout.lower()
    except Exception:
        encoders = ""

    # Prefer H.264 software encoder with a modest CRF to keep files small
    if "libx264" in encoders:
        # Use a slightly higher CRF than before to reduce size; veryfast is widely supported.
        return ["-c:v", "libx264", "-crf", "26", "-preset", "veryfast"]

    # If NVIDIA NVENC is available, use constant‑quality mode with a conservative CQ
    if "h264_nvenc" in encoders:
        return ["-c:v", "h264_nvenc", "-cq", "28", "-rc", "vbr"]

    # If x265 is available, it can yield smaller files at similar quality
    if "libx265" in encoders:
        return ["-c:v", "libx265", "-crf", "30", "-preset", "fast"]

    # Very safe fallback; keep quality modest to avoid ballooning sizes
    return ["-c:v", "mpeg4", "-qscale:v", "7"]


# === Bounding box extraction for game PiP region ===

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





# === Representative frame extraction and Gemini bbox (single inference) ===

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


def _call_gemini_bbox_game(
    image_path: str,
    *,
    project: Optional[str],
    location: str,
    model_id: str,
    layout: str = "game_full"
) -> list[int]:
    """Call Gemini to get one bounding box [ymin, xmin, ymax, xmax] (0–1000 coords) for the gameplay window or face-cam window.
    Returns 4 ints.
    """
    client = genai.Client(vertexai=True, project=project, location=location)

    if layout == "game_full":
        prompt = (
            "This image is a game reaction/stream layout where the gameplay fills the full frame, "
            "and a small face-camera (PiP) window appears in a corner. "
            "Return the bounding box of the face-camera window (the region showing the person’s face/upper body), "
            "not the gameplay area. "
            "Output ONLY a JSON array of four integers [ymin, xmin, ymax, xmax] in a 0–1000 normalized coordinate space."
        )
    else:
        prompt = (
            "This image is a game reaction/stream layout. One region is the gameplay window (game feed), "
            "which may be full-screen or a PiP window with UI/HUD/minimap; another region is the face camera. "
            "Return the bounding box for the entire gameplay/game feed panel (include its visible border if any) "
            "in [ymin, xmin, ymax, xmax], with all four integers normalized to a 0-1000 coordinate space. "
            "Return ONLY the JSON array of four integers."
        )

    parts: List[Part] = [
        Part.from_text(text=prompt),
        _part_from_local_file(image_path, mime_fallback="image/jpeg"),
    ]

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
            "Return ONLY the bounding box for the gameplay/game feed panel as [ymin, xmin, ymax, xmax] in 0-1000 integers."
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


def _detect_layout_for_video(
    ref_video_path: str,
    *,
    project: Optional[str],
    location: str,
    model_id: str,
) -> str:
    """Detect stream layout for a video frame using Gemini. Returns one of:
    - "game_full_face_small"
    - "face_full_game_small"
    Falls back to "game_full_face_small" on failure.
    """
    project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        return "game_full_face_small"

    try:
        frame_path, width, height = _extract_representative_frame(ref_video_path)
    except Exception:
        return "game_full_face_small"
    if width <= 0 or height <= 0:
        return "game_full_face_small"

    client = genai.Client(vertexai=True, project=project, location=location)
    prompt = (
        "Classify the stream layout in this image.\n"
        "Return EXACTLY one token without explanation: \n"
        "- game_full_face_small  (gameplay fills frame; face-cam is a small PiP)\n"
        "- face_full_game_small  (person fills frame; gameplay is a smaller window)\n"
    )
    parts: List[Part] = [Part.from_text(text=prompt), _part_from_local_file(frame_path, mime_fallback="image/jpeg")]
    try:
        response = client.models.generate_content(
            model=model_id,
            contents=parts,
            config=GenerateContentConfig(temperature=0.0, top_p=1.0, max_output_tokens=16),
        )
        text = (getattr(response, "text", "") or "").strip().lower()
        if "face_full_game_small" in text and "game_full_face_small" not in text:
            return "face_full_game_small"
        if "game_full_face_small" in text and "face_full_game_small" not in text:
            return "game_full_face_small"
        # If ambiguous, prefer game_full_face_small as safer default
        return "game_full_face_small"
    except Exception:
        return "game_full_face_small"


def _compute_global_bbox_or_fallback(
    ref_video_path: str,
    *,
    project: Optional[str],
    location: str,
    model_id: str,
    layout: Optional[str] = None,
) -> tuple[int, int, int, int]:
    """Compute one global bbox for the whole video via Gemini; fallback to full-frame on failure.

    If layout is provided:
      - "game_full_face_small": detect face-cam bbox
      - "face_full_game_small": detect gameplay window bbox
    """
    project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        # Fallback: full frame
        _, w_fallback, h_fallback = _extract_representative_frame(ref_video_path)
        return 0, 0, max(0, w_fallback), max(0, h_fallback)

    try:
        frame_path, width, height = _extract_representative_frame(ref_video_path)
    except Exception:
        width, height = _ffprobe_video_size(ref_video_path)
        if width <= 0 or height <= 0:
            return 0, 0, 0, 0
        return 0, 0, width, height

    if width <= 0 or height <= 0:
        return 0, 0, 0, 0

    try:
        # Map canonical layout -> bbox target for Gemini
        lay = (layout or "game_full_face_small").strip().lower()
        mode = "game_full" if lay == "game_full_face_small" else "face_full"
        ymin, xmin, ymax, xmax = _call_gemini_bbox_game(
            frame_path, project=project, location=location, model_id=model_id, layout=mode
        )
    except Exception:
        return 0, 0, width, height

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
    return x1, y1, x2, y2


def _compute_bbox_from_clip_multiframe(
    clip_path: str,
    *,
    project: Optional[str],
    location: str,
    model_id: str,
    layout: str = "game_full",
    num_frames: int = 5,
) -> tuple[int, int, int, int]:
    """Compute a robust bbox from multiple frames of a clip using Gemini.

    layout: "game_full" -> detect face-cam bbox; any other value -> detect gameplay window bbox.
    Returns pixel bbox (x1,y1,x2,y2). Falls back to full-frame on failure.
    """
    width, height = _ffprobe_video_size(clip_path)
    if width <= 0 or height <= 0:
        return 0, 0, 0, 0

    duration = _ffprobe_video_duration_seconds(clip_path)
    # Choose evenly spaced timestamps, avoid exact 0 and end
    times: list[float] = []
    nf = max(1, int(num_frames))
    if duration > 0:
        for i in range(nf):
            t = (i + 1) / float(nf + 1) * duration
            times.append(max(0.0, min(duration, t)))
    else:
        times = [1.0]

    tmp_dir = tempfile.mkdtemp(prefix="bbox_multi_")
    boxes_px: list[tuple[int,int,int,int]] = []
    try:
        for idx, t in enumerate(times):
            img_path = os.path.join(tmp_dir, f"frame_{idx:02d}.jpg")
            try:
                subprocess.run([
                    'ffmpeg','-hide_banner','-loglevel','error','-nostdin',
                    '-ss', f"{t:.3f}", '-i', clip_path,
                    '-frames:v', '1', '-q:v', '2', '-y', img_path
                ], check=True)
            except Exception:
                continue

            try:
                ymin, xmin, ymax, xmax = _call_gemini_bbox_game(
                    img_path,
                    project=project,
                    location=location,
                    model_id=model_id,
                    layout=("game_full" if layout == "game_full_face_small" or layout == "game_full" else "face_full"),
                )
                # Clamp and convert to pixels
                def _clamp(v: int) -> int:
                    return max(0, min(1000, int(v)))
                ymin = _clamp(ymin); xmin = _clamp(xmin); ymax = _clamp(ymax); xmax = _clamp(xmax)
                x1 = int(round(xmin / 1000.0 * width))
                y1 = int(round(ymin / 1000.0 * height))
                x2 = int(round(xmax / 1000.0 * width))
                y2 = int(round(ymax / 1000.0 * height))
                # Sanity fixes
                x1, x2 = max(0, min(x1, width)), max(0, min(x2, width))
                y1, y2 = max(0, min(y1, height)), max(0, min(y2, height))
                if x2 < x1: x1, x2 = x2, x1
                if y2 < y1: y1, y2 = y2, y1
                boxes_px.append((x1, y1, x2, y2))
            except Exception:
                continue

        if not boxes_px:
            return 0, 0, width, height

        # Take median across frames
        xs1 = [b[0] for b in boxes_px]
        ys1 = [b[1] for b in boxes_px]
        xs2 = [b[2] for b in boxes_px]
        ys2 = [b[3] for b in boxes_px]
        x1m = int(round(statistics.median(xs1)))
        y1m = int(round(statistics.median(ys1)))
        x2m = int(round(statistics.median(xs2)))
        y2m = int(round(statistics.median(ys2)))
        return x1m, y1m, x2m, y2m
    finally:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass


def _apply_bbox_to_clip(
    video_path: str,
    bbox: tuple[int,int,int,int],
    layout: str = "game_full_face_small",
    *,
    skip_if_exists: bool = True,
    min_ok_bytes: int = 16384,
) -> tuple[str, str]:
    """Given a clip, a global bbox (x1,y1,x2,y2), and layout, write per-clip outputs.

    Layout semantics:
      - game_full_face_small: masked = crop(face-cam region); stimuli = full frame with face region masked
      - face_full_game_small: stimuli = crop(game window);    masked  = full frame with game region masked

    Returns (stimuli_out_path, masked_out_path).
    """
    x1, y1, x2, y2 = bbox
    root_root, label_folder = _infer_root_and_label(video_path)
    label_folder = _sanitize_filename(label_folder)
    clip_dir = os.path.dirname(video_path)
    clip_dir_base = os.path.basename(clip_dir)  # e.g., "{stem}_clips"
    hi_stem = clip_dir_base[:-6] if clip_dir_base.endswith("_clips") else clip_dir_base
    stimuli_out_dir = os.path.join(root_root, 'stimuli', label_folder, f"{hi_stem}_stimuli")
    masked_out_dir  = os.path.join(root_root, 'masked',  label_folder, f"{hi_stem}_masked")
    os.makedirs(stimuli_out_dir, exist_ok=True)
    os.makedirs(masked_out_dir,  exist_ok=True)
    stimuli_out_path = os.path.join(stimuli_out_dir, os.path.basename(video_path))
    masked_out_path  = os.path.join(masked_out_dir,  os.path.basename(video_path))
    vcodec_args = _ffmpeg_video_codec_args()

    crop_w = max(1, x2 - x1); crop_h = max(1, y2 - y1)
    vw, vh = _ffprobe_video_size(video_path)
    is_fullframe_crop = (x1 == 0 and y1 == 0 and vw > 0 and vh > 0 and abs(crop_w - vw) <= 1 and abs(crop_h - vh) <= 1)

    def _is_ok(p: str) -> bool:
        try:
            return bool(p) and os.path.exists(p) and os.path.getsize(p) >= max(1, int(min_ok_bytes))
        except Exception:
            return False

    if skip_if_exists:
        stimuli_ok = _is_ok(stimuli_out_path)
        masked_ok = True if is_fullframe_crop else _is_ok(masked_out_path)
        if stimuli_ok and masked_ok:
            try:
                print(
                    f"BBox apply: outputs already exist, skipping: {stimuli_out_path}"
                    + ("" if is_fullframe_crop else f", {masked_out_path}")
                )
            except Exception:
                pass
            return (stimuli_out_path, "" if is_fullframe_crop else masked_out_path)

    # Normalize layout
    lay = (layout or "game_full_face_small").strip().lower()

    if lay == "face_full_game_small":
        # Stimuli: crop game window; Masked: full frame with game region masked
        # Handle full-frame bbox gracefully
        # masked
        try:
            if is_fullframe_crop:
                masked_out_path = ""
            else:
                subprocess.run([
                    'ffmpeg','-hide_banner','-loglevel','error','-nostdin','-i', video_path,
                    '-filter:v', f'drawbox=x={x1}:y={y1}:w={crop_w}:h={crop_h}:color=black@1:t=fill',
                    *vcodec_args,
                    '-map','0:v:0',
                    '-map','0:a:0?',
                    '-c:a','aac','-b:a','160k',
                    '-movflags','+faststart',
                    '-shortest',
                    '-y', masked_out_path
                ], check=True)
        except Exception as e:
            print(f"Warning: failed to write masked (drawbox game) for {video_path}: {e}")

        # stimuli
        try:
            if is_fullframe_crop:
                subprocess.run([
                    'ffmpeg','-hide_banner','-loglevel','error','-nostdin','-i', video_path,
                    '-map','0:v:0','-map','0:a:0?',
                    '-c','copy',
                    '-movflags','+faststart',
                    '-y', stimuli_out_path
                ], check=True)
            else:
                subprocess.run([
                    'ffmpeg','-hide_banner','-loglevel','error','-nostdin','-i', video_path,
                    '-filter:v', f'crop={crop_w}:{crop_h}:{x1}:{y1}',
                    *vcodec_args,
                    '-map','0:v:0',
                    '-map','0:a:0?',
                    '-c:a','aac','-b:a','160k',
                    '-movflags','+faststart',
                    '-shortest',
                    '-y', stimuli_out_path
                ], check=True)
        except Exception as e:
            print(f"Warning: failed to write stimuli (crop game) for {video_path}: {e}")
    else:
        # game_full_face_small (default): Masked: crop face-cam; Stimuli: full frame with face region masked
        # masked (face crop)
        if is_fullframe_crop:
            masked_out_path = ""
        else:
            try:
                subprocess.run([
                    'ffmpeg','-hide_banner','-loglevel','error','-nostdin','-i', video_path,
                    '-filter:v', f'crop={crop_w}:{crop_h}:{x1}:{y1}',
                    *vcodec_args,
                    '-map','0:v:0',
                    '-map','0:a:0?',
                    '-c:a','aac','-b:a','160k',
                    '-movflags','+faststart',
                    '-shortest',
                    '-y', masked_out_path
                ], check=True)
            except Exception as e:
                print(f"Warning: failed to write masked (face crop) for {video_path}: {e}")

        # stimuli
        try:
            if is_fullframe_crop:
                subprocess.run([
                    'ffmpeg','-hide_banner','-loglevel','error','-nostdin','-i', video_path,
                    '-map','0:v:0','-map','0:a:0?',
                    '-c','copy',
                    '-movflags','+faststart',
                    '-y', stimuli_out_path
                ], check=True)
            else:
                subprocess.run([
                    'ffmpeg','-hide_banner','-loglevel','error','-nostdin','-i', video_path,
                    '-filter:v', f'drawbox=x={x1}:y={y1}:w={crop_w}:h={crop_h}:color=black@1:t=fill',
                    *vcodec_args,
                    '-map','0:v:0',
                    '-map','0:a:0?',
                    '-c:a','aac','-b:a','160k',
                    '-movflags','+faststart',
                    '-shortest',
                    '-y', stimuli_out_path
                ], check=True)
        except Exception as e:
            print(f"Warning: failed to write stimuli (mask region) for {video_path}: {e}")

    return stimuli_out_path, masked_out_path


# === Window split helper (<=10 minutes per segment) ===

def _split_window_to_segments(start_hms: str, end_hms: str, *, max_len_s: int = 180) -> list[tuple[str, str]]:
    s0 = _hms_to_seconds(start_hms)
    s1 = _hms_to_seconds(end_hms)
    if s1 <= s0:
        return [(start_hms, end_hms)]
    out: list[tuple[str, str]] = []
    cur = s0
    while cur < s1:
        nxt = min(s1, cur + max(1, int(max_len_s)))
        out.append((_seconds_to_hms(cur), _seconds_to_hms(nxt)))
        cur = nxt
    return out


def bounding_box_extraction(
    video_path: str,
    *,
    df_path: Optional[str] = None,
    project: Optional[str] = None,
    location: str = "us-central1",
    model_id: str = "gemini-2.0-flash",
    skip_if_exists: bool = True,
    min_ok_bytes: int = 16384,
) -> tuple[int, int, int, int, str]:
    """Extract and apply bbox for game reactions **based on CSV-provided layout**.

    This version does **not** perform any automatic layout detection.
    It expects the CSV at `df_path` to contain:
      - `layout`: one of the accepted strings (case-insensitive):
          * "face_full_game_small" (aliases: "face_full", "layout_a", "a", "face-full")
          * "game_full_face_small" (aliases: "game_full", "layout_b", "b", "game-full")
      - `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2`: integer pixel coordinates for the small PiP region.
      - `clip_path`: path of the clip this row corresponds to (used to locate the row). If the current
        `video_path` points into `/masked/.../_masked/`, we will try to map back to the original `/clips/.../_clips/` path.

    Behavior by layout:
      - face_full_game_small → stimuli=crop(game window), masked=drawbox(black out game window on full frame)
      - game_full_face_small → masked=crop(face-cam region), stimuli=drawbox(black out face-cam region on full frame)

    Returns (x1, y1, x2, y2, masked_out_path). If bbox or layout is missing, we fall back to full-frame pass-through
    with layout treated as "game_full_face_small" and return (0,0,width,height,"") where width/height may be unknown (0,0).
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Read layout/bbox from CSV
    layout_val = None
    bbox_vals: tuple[int,int,int,int] | None = None
    if df_path is not None and os.path.exists(df_path):
        try:
            df = pd.read_csv(df_path)
            # Find row by exact clip_path match; if not found and video is from masked folder, map to original clip path
            row = None
            if "clip_path" in df.columns:
                m = (df["clip_path"].astype(str) == video_path)
                if not m.any():
                    # Try masked→clips reverse mapping
                    parts = video_path.split(os.sep)
                    try:
                        idx = parts.index("masked")
                        if idx + 1 < len(parts):
                            masked_dir = parts[idx + 1]
                            if masked_dir.endswith("_masked"):
                                orig_dir = masked_dir[:-7] + "_clips"
                                candidate = os.sep.join(parts[:idx] + ["clips", orig_dir] + parts[idx + 2 :])
                                m = (df["clip_path"].astype(str) == candidate)
                    except Exception:
                        pass
                if m.any():
                    row = df.loc[m].iloc[0]

            if row is not None:
                layout_val = str(row.get("layout", "")).strip().lower() or None
                try:
                    x1 = int(float(row.get("bbox_x1", 0)))
                    y1 = int(float(row.get("bbox_y1", 0)))
                    x2 = int(float(row.get("bbox_x2", 0)))
                    y2 = int(float(row.get("bbox_y2", 0)))
                    if x2 > x1 and y2 > y1:
                        bbox_vals = (x1, y1, x2, y2)
                except Exception:
                    bbox_vals = None
        except Exception as e:
            print(f"bounding_box_extraction: failed to read bbox/layout from {df_path}: {e}")

    # Normalize layout aliases → canonical keys
    def _norm_layout(val: Optional[str]) -> str:
        v = (val or "").strip().lower()
        if v in {"face_full_game_small", "face_full", "layout_a", "a", "face-full"}:
            return "face_full_game_small"
        if v in {"game_full_face_small", "game_full", "layout_b", "b", "game-full"}:
            return "game_full_face_small"
        # default
        return "game_full_face_small"

    layout = _norm_layout(layout_val)

    # Prepare output dirs
    masked_out_path = ""
    try:
        root_root, label_folder = _infer_root_and_label(video_path)
        label_folder = _sanitize_filename(label_folder)
        clip_dir = os.path.dirname(video_path)
        clip_dir_base = os.path.basename(clip_dir)
        hi_stem = clip_dir_base[:-6] if clip_dir_base.endswith("_clips") else clip_dir_base
        stimuli_out_dir = os.path.join(root_root, 'stimuli', label_folder, f"{hi_stem}_stimuli")
        masked_out_dir  = os.path.join(root_root, 'masked',  label_folder, f"{hi_stem}_masked")
        os.makedirs(stimuli_out_dir, exist_ok=True)
        os.makedirs(masked_out_dir,  exist_ok=True)
        stimuli_out_path = os.path.join(stimuli_out_dir, os.path.basename(video_path))
        masked_out_path  = os.path.join(masked_out_dir,  os.path.basename(video_path))
        vcodec_args = _ffmpeg_video_codec_args()

        # If bbox missing, attempt multi-frame Gemini bbox from the first clip (0000) and write back to CSV
        if bbox_vals is None:
            first_clip = os.path.join(clip_dir, "0000.mp4")
            if not os.path.exists(first_clip):
                try:
                    mp4s = sorted([f for f in os.listdir(clip_dir) if f.lower().endswith('.mp4')])
                    if mp4s:
                        first_clip = os.path.join(clip_dir, mp4s[0])
                except Exception:
                    pass
            # Compute bbox via multi-frame helper; layout default is normalized layout above
            try:
                bx1, by1, bx2, by2 = _compute_bbox_from_clip_multiframe(
                    first_clip if os.path.exists(first_clip) else video_path,
                    project=project,
                    location=location,
                    model_id=model_id,
                    layout=layout,
                    num_frames=12,
                )
                if bx2 > bx1 and by2 > by1:
                    bbox_vals = (bx1, by1, bx2, by2)
                    # Write bbox back to CSV for all rows missing bbox
                    if df_path is not None and os.path.exists(df_path):
                        try:
                            df2 = pd.read_csv(df_path)
                            for col in ["bbox_x1","bbox_y1","bbox_x2","bbox_y2","layout"]:
                                if col not in df2.columns:
                                    df2[col] = ""
                            # Fill where empty or invalid
                            def _empty(v):
                                try:
                                    return str(v).strip() == "" or float(v) == 0
                                except Exception:
                                    return True
                            m_empty = (
                                df2["bbox_x1"].map(_empty) |
                                df2["bbox_y1"].map(_empty) |
                                df2["bbox_x2"].map(_empty) |
                                df2["bbox_y2"].map(_empty)
                            )
                            df2.loc[m_empty, ["bbox_x1","bbox_y1","bbox_x2","bbox_y2"]] = [bx1, by1, bx2, by2]
                            df2["layout"] = layout
                            df2.to_csv(df_path, index=False)
                            print(f"Wrote inferred bbox ({bx1},{by1},{bx2},{by2}) from {os.path.basename(first_clip) if os.path.exists(first_clip) else os.path.basename(video_path)} back to CSV")
                        except Exception as e:
                            print(f"Warning: failed to persist inferred bbox to CSV: {e}")
                else:
                    print("Warning: inferred bbox invalid; falling back to pass-through.")
            except Exception as e:
                print(f"Warning: multi-frame bbox inference failed: {e}")
                # proceed to pass-through below if still missing
            if bbox_vals is None:
                # safe pass-through: stimuli copy, masked empty
                try:
                    subprocess.run([
                        'ffmpeg','-hide_banner','-loglevel','error','-nostdin','-i', video_path,
                        '-map','0:v:0','-map','0:a:0?','-c','copy','-movflags','+faststart','-an','-y', stimuli_out_path
                    ], check=True)
                except Exception as e:
                    print(f"Warning: failed to write stimuli (passthrough): {e}")
                return (0, 0, 0, 0, masked_out_path)

        x1, y1, x2, y2 = bbox_vals
        crop_w = max(1, x2 - x1); crop_h = max(1, y2 - y1)
        vw, vh = _ffprobe_video_size(video_path)
        is_fullframe_crop = (x1 == 0 and y1 == 0 and vw > 0 and vh > 0 and abs(crop_w - vw) <= 1 and abs(crop_h - vh) <= 1)

        # Optional skip if outputs already exist
        def _is_ok(p: str) -> bool:
            try:
                return bool(p) and os.path.exists(p) and os.path.getsize(p) >= max(1, int(min_ok_bytes))
            except Exception:
                return False
        if skip_if_exists:
            stimuli_ok = _is_ok(stimuli_out_path)
            masked_ok = True if is_fullframe_crop else _is_ok(masked_out_path)
            if stimuli_ok and masked_ok:
                # Update CSV with directories and normalized layout string if possible
                if df_path is not None and os.path.exists(df_path):
                    try:
                        df2 = pd.read_csv(df_path)
                        for col in ["stimuli_out_dir","masked_out_dir","bbox_x1","bbox_y1","bbox_x2","bbox_y2","layout"]:
                            if col not in df2.columns:
                                df2[col] = ""
                        if "clip_path" in df2.columns:
                            mm = (df2["clip_path"].astype(str) == video_path)
                            if not mm.any():
                                parts = video_path.split(os.sep)
                                try:
                                    idx = parts.index("masked")
                                    if idx + 1 < len(parts):
                                        masked_dir = parts[idx + 1]
                                        if masked_dir.endswith("_masked"):
                                            orig_dir = masked_dir[:-7] + "_clips"
                                            candidate = os.sep.join(parts[:idx] + ["clips", orig_dir] + parts[idx + 2 :])
                                            mm = (df2["clip_path"].astype(str) == candidate)
                                except Exception:
                                    pass
                            if mm.any():
                                df2.loc[mm, "stimuli_out_dir"] = os.path.dirname(stimuli_out_path)
                                df2.loc[mm, "masked_out_dir"]  = os.path.dirname(masked_out_path) if not is_fullframe_crop else ""
                                df2.loc[mm, ["bbox_x1","bbox_y1","bbox_x2","bbox_y2"]] = [x1,y1,x2,y2]
                                df2.loc[mm, "layout"] = layout
                        else:
                            df2["stimuli_out_dir"] = os.path.dirname(stimuli_out_path)
                            df2["masked_out_dir"]  = os.path.dirname(masked_out_path) if not is_fullframe_crop else ""
                            df2["bbox_x1"], df2["bbox_y1"], df2["bbox_x2"], df2["bbox_y2"] = x1, y1, x2, y2
                            df2["layout"] = layout
                        df2.to_csv(df_path, index=False)
                    except Exception:
                        pass
                print(f"BBox skip-existing: {os.path.basename(video_path)}")
                return (x1, y1, x2, y2, "" if is_fullframe_crop else masked_out_path)

        if layout == "face_full_game_small":
            # A) Face is full; game is small window →
            #   - stimuli: crop the GAME window (x1,y1,x2,y2)
            #   - masked: black out the GAME window on the full frame
            if is_fullframe_crop:
                # Graceful fallback: stimuli copy, masked empty
                try:
                    subprocess.run([
                        'ffmpeg','-hide_banner','-loglevel','error','-nostdin','-i', video_path,
                        '-map','0:v:0','-map','0:a:0?','-c','copy','-movflags','+faststart','-an','-y', stimuli_out_path
                    ], check=True)
                except Exception as e:
                    print(f"Warning: failed to write stimuli (full-frame passthrough): {e}")
                masked_out_path = ""
            else:
                try:
                    cmd = ['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-i', video_path,
                           '-filter:v', f'crop={crop_w}:{crop_h}:{x1}:{y1}', *vcodec_args, '-an','-y', stimuli_out_path]
                    subprocess.run(cmd, check=True)
                except Exception as e:
                    print(f"Warning: failed to crop game stimuli: {e}")
                try:
                    cmd_mask = ['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-i', video_path,
                                '-filter:v', f'drawbox=x={x1}:y={y1}:w={crop_w}:h={crop_h}:color=black@1:t=fill',
                                *vcodec_args,'-an','-y', masked_out_path]
                    subprocess.run(cmd_mask, check=True)
                except Exception as e:
                    print(f"Warning: failed to drawbox game mask: {e}")
        else:
            # B) Game is full; face-cam is small window →
            #   - masked: person crop (save cropped face-cam region)
            #   - stimuli: black out the person region on the full frame
            if is_fullframe_crop:
                # Graceful fallback: stimuli copy, masked empty
                try:
                    subprocess.run([
                        'ffmpeg','-hide_banner','-loglevel','error','-nostdin','-i', video_path,
                        '-map','0:v:0','-map','0:a:0?','-c','copy','-movflags','+faststart','-an','-y', stimuli_out_path
                    ], check=True)
                except Exception as e:
                    print(f"Warning: failed to write stimuli (full-frame passthrough): {e}")
                masked_out_path = ""
            else:
                try:
                    cmd_person = ['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-i', video_path,
                                  '-filter:v', f'crop={crop_w}:{crop_h}:{x1}:{y1}', *vcodec_args, '-an','-y', masked_out_path]
                    subprocess.run(cmd_person, check=True)
                except Exception as e:
                    print(f"Warning: failed to crop person masked clip: {e}")
                try:
                    cmd_game = ['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-i', video_path,
                                '-filter:v', f'drawbox=x={x1}:y={y1}:w={crop_w}:h={crop_h}:color=black@1:t=fill',
                                *vcodec_args,'-an','-y', stimuli_out_path]
                    subprocess.run(cmd_game, check=True)
                except Exception as e:
                    print(f"Warning: failed to write game stimuli with masked person: {e}")

        # Update CSV with directories and normalized layout string
        if df_path is not None and os.path.exists(df_path):
            try:
                df2 = pd.read_csv(df_path)
                for col in ["stimuli_out_dir","masked_out_dir","bbox_x1","bbox_y1","bbox_x2","bbox_y2","layout"]:
                    if col not in df2.columns:
                        df2[col] = ""
                if "clip_path" in df2.columns:
                    mm = (df2["clip_path"].astype(str) == video_path)
                    if not mm.any():
                        parts = video_path.split(os.sep)
                        try:
                            idx = parts.index("masked")
                            if idx + 1 < len(parts):
                                masked_dir = parts[idx + 1]
                                if masked_dir.endswith("_masked"):
                                    orig_dir = masked_dir[:-7] + "_clips"
                                    candidate = os.sep.join(parts[:idx] + ["clips", orig_dir] + parts[idx + 2 :])
                                    mm = (df2["clip_path"].astype(str) == candidate)
                        except Exception:
                            pass
                    if mm.any():
                        df2.loc[mm, "stimuli_out_dir"] = os.path.dirname(stimuli_out_path)
                        df2.loc[mm, "masked_out_dir"]  = os.path.dirname(masked_out_path)
                        df2.loc[mm, ["bbox_x1","bbox_y1","bbox_x2","bbox_y2"]] = [x1,y1,x2,y2]
                        df2.loc[mm, "layout"] = layout
                else:
                    df2["stimuli_out_dir"] = os.path.dirname(stimuli_out_path)
                    df2["masked_out_dir"]  = os.path.dirname(masked_out_path)
                    df2["bbox_x1"], df2["bbox_y1"], df2["bbox_x2"], df2["bbox_y2"] = x1, y1, x2, y2
                    df2["layout"] = layout
                df2.to_csv(df_path, index=False)
            except Exception as e:
                print(f"Warning: failed to update stimuli/masked/bbox/layout in {df_path}: {e}")
    except Exception as e:
        print(f"Warning: layout/masking pipeline failed for {video_path}: {e}")

    return (x1 if bbox_vals else 0, y1 if bbox_vals else 0, x2 if bbox_vals else 0, y2 if bbox_vals else 0, masked_out_path)


def bounding_box_extraction_batch(
    video_paths: List[str],
    *,
    df_path: Optional[str] = None,
    project: Optional[str] = None,
    location: str = "us-central1",
    model_id: str = "gemini-2.0-flash",
    max_workers: int = 6,
    skip_if_exists: bool = True,
    min_ok_bytes: int = 16384,
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
                vp,
                df_path=df_path,
                project=project,
                location=location,
                model_id=model_id,
                skip_if_exists=skip_if_exists,
                min_ok_bytes=min_ok_bytes,
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


#
# === Debug helpers for subprocess runs ===
def _write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", errors="ignore") as f:
        f.write(text or "")

def _tail(text: str, n: int = 40) -> str:
    lines = (text or "").splitlines()
    return "\n".join(lines[-n:])

def _providers_fallback_list(providers: Optional[str]) -> list[str]:
    base = (providers or "").strip() or "TensorrtExecutionProvider,CUDAExecutionProvider"
    order = [base, "CUDAExecutionProvider", "CPUExecutionProvider"]
    # de-dup while preserving order
    seen, out = set(), []
    for p in order:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def talking_head_extraction(
    video_path: str,
    *,
    df_path: Optional[str] = None,
    skip_if_exists: bool = True,
    min_ok_bytes: int = 16384,
    fast_mode: bool = True,
    providers: Optional[str] = "CUDAExecutionProvider,CPUExecutionProvider",
    trt_cache_dir: Optional[str] = None,
    keep_audio: bool = True,
) -> str:
    """Extract talking head (head-and-shoulders crop) from a video clip.

    When the layout is game‑full/face‑small, `video_path` will typically be the **person‑crop** saved in the `masked` folder by `bounding_box_extraction`; in that case, this function will refine that crop to a centered face clip under `face/...`.

    Uses the ghs_crop tool to create a square crop focused on the person's head and shoulders. Creates a 'face' folder structure similar to the clips folder. Updates the CSV with the face folder path.

    This function can take either a regular clip or a masked clip produced by bounding_box_extraction. When a masked clip is provided, the CSV update will attempt to route the face_out_dir to the row corresponding to the original unmasked clip when possible.
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
    providers = providers or "CUDAExecutionProvider,CPUExecutionProvider"
    print(f"🔧 Using execution providers: {providers}")
    if providers and "TensorrtExecutionProvider" in providers:
        providers = providers.replace("TensorrtExecutionProvider,", "").replace(",TensorrtExecutionProvider", "")
    if "TensorrtExecutionProvider" not in providers:
        print("ℹ️ TensorRT EP disabled (not installed or mismatched); using CUDA/CPU.")
    print(f"🧭 CUDA available: {torch.cuda.is_available()} | device_count={torch.cuda.device_count()} | current_device={(torch.cuda.current_device() if torch.cuda.is_available() else 'CPU')}")
    # Enforce GPU-only execution
    if "CUDAExecutionProvider" in providers or "TensorrtExecutionProvider" in providers:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU-only mode requested but no CUDA-capable GPU is available. Aborting.")

    def _is_ok(p: str) -> bool:
        try:
            return os.path.exists(p) and os.path.getsize(p) >= max(1, int(min_ok_bytes))
        except Exception:
            return False

    # Local helper to attach original audio to the face clip if missing
    def _maybe_add_audio():
        try:
            if keep_audio:
                _mux_original_audio_into_face(face_out_path, video_path, min_ok_bytes=min_ok_bytes)
        except Exception:
            pass

    # Skip if output already exists and is valid
    if skip_if_exists and _is_ok(face_out_path):
        elapsed = time.time() - start_time
        print(f"⏩ Face extraction already exists: {face_out_path} (checked in {elapsed:.2f}s)")
        # Still update CSV if needed
        _update_csv_face_path(df_path, video_path, face_out_dir)
        return face_out_path

    # Check if ghs_crop tool is available
    try:
        # Import the talking_head ghs_crop module
        _ensure_talking_head_deps()
        talking_head_root = _resolve_talking_head_root()
        if talking_head_root not in sys.path:
            sys.path.insert(0, talking_head_root)

        # Prepare debug log paths
        logs_dir = os.path.join(face_out_dir, ".logs")
        stdout_log = os.path.join(logs_dir, f"{os.path.basename(face_out_path)}.stdout.log")
        stderr_log = os.path.join(logs_dir, f"{os.path.basename(face_out_path)}.stderr.log")
        cmd_log    = os.path.join(logs_dir, f"{os.path.basename(face_out_path)}.cmd.txt")

        # Set environment variables to optimize performance
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "4"  # Limit CPU threads to avoid oversubscription
        env["MKL_NUM_THREADS"] = "4"
        env["PYTHONUNBUFFERED"] = "1"
        env["NVIDIA_TF32_OVERRIDE"] = "0"
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

        # Enable very verbose ORT/TensorRT logs for failures
        env_fail = env.copy()
        env_fail["ORT_LOG_SEVERITY_LEVEL"] = "0"          # 0=VERBOSE
        env_fail["CUDA_LAUNCH_BLOCKING"] = "1"            # sync CUDA for better stacktraces

        # Persist the exact command and key env for repro (will fill in actual cmd in helper)
        _write_text(cmd_log, "\n".join([
            f"CWD={os.getcwd()}",
            f"PYTHON={sys.executable}",
            f"PROVIDERS={providers}",
            f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES','')}",
            f"ORT_TENSORRT_ENGINE_CACHE_PATH={env.get('ORT_TENSORRT_ENGINE_CACHE_PATH','')}",
            # "CMD=" will be written by the helper
        ]))

        orig_cwd = os.getcwd()
        try:
            os.chdir(talking_head_root)

            def _run_ghs_with(prov: str, step: str, conf_thres: str, det_size: str, safety_scale: str, face_area_percentile: str, use_env_fail: bool = False) -> bool:
                cmd_local = [
                    sys.executable, "-m", "ghs_crop",
                    "--input", video_path,
                    "--output", face_out_path,
                    "--step", step,
                    "--conf-thres", conf_thres,
                    "--det-size", det_size,
                    "--headroom-bias", "0.05",
                    "--safety-scale", safety_scale,
                    "--face-area-percentile", face_area_percentile,
                    "--providers", prov,
                ]
                try:
                    env_base = env_fail if use_env_fail else env
                    # Allow CPU provider when explicitly requested.
                    if "CPUExecutionProvider" in (prov or "") and env_base.get("ORT_DISABLE_CPU") == "1":
                        env_local = env_base.copy()
                        env_local["ORT_DISABLE_CPU"] = "0"
                    else:
                        env_local = env_base
                    run = subprocess.run(
                        cmd_local,
                        check=True,
                        capture_output=True,
                        text=True,
                        env=env_local,
                        timeout=600,
                    )
                    _write_text(stdout_log, run.stdout)
                    _write_text(stderr_log, run.stderr)
                    print("📝 ghs_crop stdout tail:\n" + _tail(run.stdout))
                    print("📝 ghs_crop stderr tail:\n" + _tail(run.stderr))
                    return True
                except subprocess.CalledProcessError as e:
                    _write_text(stdout_log, e.stdout)
                    _write_text(stderr_log, e.stderr)
                    print(f"❗ ghs_crop failed (providers={prov}) rc={e.returncode}")
                    print("── stdout tail ──\n" + _tail(e.stdout))
                    print("── stderr tail ──\n" + _tail(e.stderr))
                    return False
                except subprocess.TimeoutExpired as e:
                    _write_text(stdout_log, (e.stdout or ""))
                    _write_text(stderr_log, (e.stderr or ""))
                    print(f"⏱️ ghs_crop timed out (providers={prov})")
                    return False

            # Stage A: fast settings
            if fast_mode:
                step = "10"
                conf_thres = "0.6"
                det_size = "384"
                safety_scale = "1.5"
                face_pct = "60"
            else:
                step = "3"
                conf_thres = "0.7"
                det_size = "640"
                safety_scale = "1.3"
                face_pct = "60"
            stage_a_success = False
            for prov in _providers_fallback_list(providers):
                print(f"🔁 ghs_crop fast attempt with providers: {prov}")
                if _run_ghs_with(prov, step, conf_thres, det_size, safety_scale, face_pct, use_env_fail=False):
                    stage_a_success = True
                    break
            # If Stage A succeeded, update CSV and print
            if stage_a_success:
                ghs_elapsed = time.time() - start_time
                total_elapsed = time.time() - start_time
                print(f"✅ Face extraction completed: {face_out_path} (ghs_crop: {ghs_elapsed:.2f}s, total: {total_elapsed:.2f}s)")
            else:
                # Stage B: robust settings
                print("🔁 No valid face detections at fast settings; retrying with robust settings (det=640, step=3, conf=0.30).")
                step = "3"
                conf_thres = "0.30"
                det_size = "640"
                safety_scale = "1.5"
                face_pct = "55"
                stage_b_success = False
                for prov in _providers_fallback_list(providers):
                    print(f"🔁 ghs_crop robust attempt with providers: {prov}")
                    if _run_ghs_with(prov, step, conf_thres, det_size, safety_scale, face_pct, use_env_fail=True):
                        stage_b_success = True
                        break
                if stage_b_success:
                    ghs_elapsed = time.time() - start_time
                    total_elapsed = time.time() - start_time
                    print(f"✅ Face extraction completed (robust): {face_out_path} (ghs_crop: {ghs_elapsed:.2f}s, total: {total_elapsed:.2f}s)")
                else:
                    # Stage C: ultra-robust settings
                    print("🔁 Ultra-robust attempt (det=960, step=1, conf=0.15, face_pct=40).")
                    step = "1"
                    conf_thres = "0.15"
                    det_size = "960"
                    safety_scale = "1.1"
                    face_pct = "40"
                    stage_c_success = False
                    for prov in _providers_fallback_list(providers):
                        print(f"🔁 ghs_crop ultra attempt with providers: {prov}")
                        if _run_ghs_with(prov, step, conf_thres, det_size, safety_scale, face_pct, use_env_fail=True):
                            ghs_elapsed = time.time() - start_time
                            total_elapsed = time.time() - start_time
                            print(f"✅ Face extraction completed (ultra): {face_out_path} (ghs_crop: {ghs_elapsed:.2f}s, total: {total_elapsed:.2f}s)")
                            stage_c_success = True
                            break
                    if stage_c_success:
                        # Ensure audio on face clip before returning
                        _maybe_add_audio()
                        _update_csv_face_path(df_path, video_path, os.path.dirname(face_out_path))
                        return face_out_path
                    # Stage D: masked→original fallback
                    masked_parts = video_path.split(os.sep)
                    try:
                        idx = masked_parts.index("masked")
                        if idx + 1 < len(masked_parts):
                            masked_dir = masked_parts[idx + 1]
                            if masked_dir.endswith("_masked"):
                                orig_dir = masked_dir[:-7] + "_clips"
                                orig_clip = os.sep.join(masked_parts[:idx] + ["clips", orig_dir] + masked_parts[idx + 2 :])
                            else:
                                orig_clip = ""
                        else:
                            orig_clip = ""
                    except Exception:
                        orig_clip = ""
                    if orig_clip and os.path.exists(orig_clip):
                        print(f"🧪 Retrying on unmasked original: {orig_clip}")
                        orig_video_path = orig_clip
                        # Try robust settings again
                        for prov in _providers_fallback_list(providers):
                            print(f"🔁 ghs_crop robust attempt (orig) with providers: {prov}")
                            if _run_ghs_with(prov, step, conf_thres, det_size, safety_scale, face_pct, use_env_fail=True):
                                ghs_elapsed = time.time() - start_time
                                total_elapsed = time.time() - start_time
                                print(f"✅ Face extraction completed (orig robust): {face_out_path} (ghs_crop: {ghs_elapsed:.2f}s, total: {total_elapsed:.2f}s)")
                                break
                        else:
                            # Ultra-robust on original
                            print("🔁 ghs_crop ultra attempt (orig) det=960, step=1, conf=0.15, face_pct=40")
                            step = "1"; conf_thres = "0.15"; det_size = "960"; safety_scale = "1.1"; face_pct = "40"
                            for prov in _providers_fallback_list(providers):
                                print(f"🔁 ghs_crop ultra attempt (orig) with providers: {prov}")
                                if _run_ghs_with(prov, step, conf_thres, det_size, safety_scale, face_pct, use_env_fail=True):
                                    face_out_dir = os.path.dirname(face_out_path)
                                    _update_csv_face_path(df_path, orig_video_path, face_out_dir)
                                    print(f"✅ Face extraction completed (orig ultra): {face_out_path}")
                                    _maybe_add_audio()
                                    return face_out_path
                            raise RuntimeError(
                                f"ghs_crop failed after all retries (robust, ultra, orig fallback). See logs: {stdout_log} / {stderr_log} / {cmd_log}"
                            )
                    else:
                        raise RuntimeError(
                            f"ghs_crop failed after all retries (fast, robust, ultra). See logs: {stdout_log} / {stderr_log} / {cmd_log}"
                        )
        finally:
            os.chdir(orig_cwd)

    except Exception as e:
        print(f"Warning: face extraction failed for {video_path}: {e}")
        print(f"⚠️ See logs under {os.path.join(os.path.dirname(face_out_path), '.logs')} for details.")
        # Create empty file to mark as attempted
        with open(face_out_path, 'w') as f:
            f.write("")

    # Update CSV with face folder path
    # Update CSV with face folder path
    _maybe_add_audio()
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
    providers: Optional[str] = "CUDAExecutionProvider,CPUExecutionProvider",
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
        else:
            videos_to_process.append((video_path, face_out_path))

    if not videos_to_process:
        print("All face extractions already exist, skipping batch processing")
        return output_paths

    batch_start = time.time()
    print(f"🚀 Batch processing {len(videos_to_process)} videos for face extraction at {time.strftime('%H:%M:%S')}")

    # Use the process_masked_videos_optimized.py approach for batch processing
    try:
        talking_head_root = _resolve_talking_head_root()
        if talking_head_root not in sys.path:
            sys.path.insert(0, talking_head_root)
        _ensure_talking_head_deps()

        # Use top-level imports for ThreadPoolExecutor, as_completed, threading

        lock = threading.Lock()
        processed_count = 0
        failed_count = 0

        def process_single_video_optimized(video_input_output):
            nonlocal processed_count, failed_count
            video_path, face_out_path = video_input_output

            single_start = time.time()
            print(f"🎬 Starting {os.path.basename(video_path)} at {time.strftime('%H:%M:%S')}")
            # Normalize providers, strip TensorRT if present
            local_providers = providers or "CUDAExecutionProvider,CPUExecutionProvider"
            if local_providers and "TensorrtExecutionProvider" in local_providers:
                local_providers = local_providers.replace("TensorrtExecutionProvider,", "").replace(",TensorrtExecutionProvider", "")
            if "TensorrtExecutionProvider" not in local_providers:
                print("ℹ️ TensorRT EP disabled (not installed or mismatched); using CUDA/CPU.")
            print(f"🔧 Using execution providers for {os.path.basename(video_path)}: {local_providers or 'default (CPU)'}")
            if ("CUDAExecutionProvider" in (local_providers or "")) or ("TensorrtExecutionProvider" in (local_providers or "")):
                if not torch.cuda.is_available():
                    raise RuntimeError("GPU-only mode requested but no CUDA-capable GPU is available. Aborting.")

            try:
                logs_dir = os.path.join(os.path.dirname(face_out_path), ".logs")
                stdout_log = os.path.join(logs_dir, f"{os.path.basename(face_out_path)}.stdout.log")
                stderr_log = os.path.join(logs_dir, f"{os.path.basename(face_out_path)}.stderr.log")
                cmd_log    = os.path.join(logs_dir, f"{os.path.basename(face_out_path)}.cmd.txt")

                # Set environment variables to optimize performance
                env = os.environ.copy()
                env["OMP_NUM_THREADS"] = "2"  # Fewer threads for parallel processing
                env["MKL_NUM_THREADS"] = "2"
                env["PYTHONUNBUFFERED"] = "1"
                env["NVIDIA_TF32_OVERRIDE"] = "0"
                _cache_dir = trt_cache_dir or os.path.join(os.path.dirname(face_out_path), ".trt_cache")
                os.makedirs(_cache_dir, exist_ok=True)
                env["ORT_TENSORRT_ENGINE_CACHE_ENABLE"] = "1"
                env["ORT_TENSORRT_ENGINE_CACHE_PATH"] = _cache_dir
                env["ORT_TENSORRT_FP16_ENABLE"] = "1"
                env["ORT_TENSORRT_MAX_WORKSPACE_SIZE"] = str(8 * 1024 * 1024 * 1024)
                env["CUDA_MODULE_LOADING"] = "LAZY"
                env["ORT_DISABLE_CPU"] = "1"
                env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

                env_fail = env.copy()
                env_fail["ORT_LOG_SEVERITY_LEVEL"] = "0"
                env_fail["CUDA_LAUNCH_BLOCKING"] = "1"

                orig_cwd = os.getcwd()
                try:
                    os.chdir(talking_head_root)

                    def _run_ghs_with(prov: str, step: str, conf_thres: str, det_size: str, safety_scale: str, face_area_percentile: str, use_env_fail: bool = False) -> bool:
                        cmd_local = [
                            sys.executable, "-m", "ghs_crop",
                            "--input", video_path,
                            "--output", face_out_path,
                            "--step", step,
                            "--conf-thres", conf_thres,
                            "--det-size", det_size,
                            "--headroom-bias", "0.05",
                            "--safety-scale", safety_scale,
                            "--face-area-percentile", face_area_percentile,
                            "--providers", prov,
                        ]
                        try:
                            run = subprocess.run(
                                cmd_local,
                                check=True,
                                capture_output=True,
                                text=True,
                                env=(env if not use_env_fail else env_fail),
                                timeout=600,
                            )
                            _write_text(stdout_log, run.stdout)
                            _write_text(stderr_log, run.stderr)
                            print("📝 stdout tail:\n" + _tail(run.stdout))
                            print("📝 stderr tail:\n" + _tail(run.stderr))
                            return True
                        except subprocess.CalledProcessError as e:
                            _write_text(stdout_log, e.stdout)
                            _write_text(stderr_log, e.stderr)
                            print(f"❗ [{os.path.basename(video_path)}] ghs_crop failed (providers={prov}) rc={e.returncode}")
                            print("── stdout tail ──\n" + _tail(e.stdout))
                            print("── stderr tail ──\n" + _tail(e.stderr))
                            return False
                        except subprocess.TimeoutExpired as e:
                            _write_text(stdout_log, (e.stdout or ""))
                            _write_text(stderr_log, (e.stderr or ""))
                            print(f"⏱️ [{os.path.basename(video_path)}] ghs_crop timed out (providers={prov})")
                            return False

                    # Stage A: fast settings
                    if fast_mode:
                        step = "10"
                        conf_thres = "0.6"
                        det_size = "384"
                        safety_scale = "1.5"
                        face_pct = "60"
                    else:
                        step = "3"
                        conf_thres = "0.7"
                        det_size = "640"
                        safety_scale = "1.3"
                        face_pct = "60"
                    stage_a_success = False
                    for prov in _providers_fallback_list(local_providers):
                        print(f"🔁 [{os.path.basename(video_path)}] ghs_crop fast attempt with providers: {prov}")
                        if _run_ghs_with(prov, step, conf_thres, det_size, safety_scale, face_pct, use_env_fail=False):
                            stage_a_success = True
                            break
                    if stage_a_success:
                        face_out_dir = os.path.dirname(face_out_path)
                        _update_csv_face_path(df_path, video_path, face_out_dir)
                        csv_elapsed = time.time() - single_start
                        single_elapsed = time.time() - single_start
                        with lock:
                            processed_count += 1
                            print(f"✅ Face extraction completed ({processed_count}/{len(videos_to_process)}): {os.path.basename(video_path)} "
                                  f"(csv: {csv_elapsed:.2f}s, total: {single_elapsed:.2f}s)")
                        return True, video_path, "success"
                    # Stage B: robust settings
                    print("🔁 No valid face detections at fast settings; retrying with robust settings (det=640, step=3, conf=0.30).")
                    step = "3"
                    conf_thres = "0.30"
                    det_size = "640"
                    safety_scale = "1.5"
                    face_pct = "55"
                    stage_b_success = False
                    for prov in _providers_fallback_list(local_providers):
                        print(f"🔁 [{os.path.basename(video_path)}] ghs_crop robust attempt with providers: {prov}")
                        if _run_ghs_with(prov, step, conf_thres, det_size, safety_scale, face_pct, use_env_fail=True):
                            stage_b_success = True
                            break
                    if stage_b_success:
                        face_out_dir = os.path.dirname(face_out_path)
                        _update_csv_face_path(df_path, video_path, face_out_dir)
                        csv_elapsed = time.time() - single_start
                        single_elapsed = time.time() - single_start
                        with lock:
                            processed_count += 1
                            print(f"✅ Face extraction completed (robust) ({processed_count}/{len(videos_to_process)}): {os.path.basename(video_path)} "
                                  f"(csv: {csv_elapsed:.2f}s, total: {single_elapsed:.2f}s)")
                        return True, video_path, "success"
                    # Stage C: ultra-robust settings
                    print("🔁 Ultra-robust attempt (det=960, step=1, conf=0.15, face_pct=40).")
                    step = "1"; conf_thres = "0.15"; det_size = "960"; safety_scale = "1.1"; face_pct = "40"
                    stage_c_success = False
                    for prov in _providers_fallback_list(local_providers):
                        print(f"🔁 [{os.path.basename(video_path)}] ghs_crop ultra attempt with providers: {prov}")
                        if _run_ghs_with(prov, step, conf_thres, det_size, safety_scale, face_pct, use_env_fail=True):
                            face_out_dir = os.path.dirname(face_out_path)
                            _update_csv_face_path(df_path, video_path, face_out_dir)
                            csv_elapsed = time.time() - single_start
                            single_elapsed = time.time() - single_start
                            with lock:
                                processed_count += 1
                                print(f"✅ Face extraction completed (ultra) ({processed_count}/{len(videos_to_process)}): {os.path.basename(video_path)} (csv: {csv_elapsed:.2f}s, total: {single_elapsed:.2f}s)")
                            return True, video_path, "success"
                    # Stage D: masked→original fallback
                    masked_parts = video_path.split(os.sep)
                    try:
                        idx = masked_parts.index("masked")
                        if idx + 1 < len(masked_parts):
                            masked_dir = masked_parts[idx + 1]
                            if masked_dir.endswith("_masked"):
                                orig_dir = masked_dir[:-7] + "_clips"
                                orig_clip = os.sep.join(masked_parts[:idx] + ["clips", orig_dir] + masked_parts[idx + 2 :])
                            else:
                                orig_clip = ""
                        else:
                            orig_clip = ""
                    except Exception:
                        orig_clip = ""
                    if orig_clip and os.path.exists(orig_clip):
                        print(f"🧪 Retrying on unmasked original: {orig_clip}")
                        orig_video_path = orig_clip
                        for prov in _providers_fallback_list(local_providers):
                            print(f"🔁 [{os.path.basename(orig_video_path)}] ghs_crop robust attempt (orig) with providers: {prov}")
                            if _run_ghs_with(prov, step, conf_thres, det_size, safety_scale, face_pct, use_env_fail=True):
                                face_out_dir = os.path.dirname(face_out_path)
                                _update_csv_face_path(df_path, orig_video_path, face_out_dir)
                                csv_elapsed = time.time() - single_start
                                single_elapsed = time.time() - single_start
                                with lock:
                                    processed_count += 1
                                    print(f"✅ Face extraction completed (orig robust) ({processed_count}/{len(videos_to_process)}): {os.path.basename(orig_video_path)} "
                                          f"(csv: {csv_elapsed:.2f}s, total: {single_elapsed:.2f}s)")
                                return True, orig_video_path, "success"
                        # Ultra-robust on original
                        print("🔁 ghs_crop ultra attempt (orig) det=960, step=1, conf=0.15, face_pct=40")
                        step = "1"; conf_thres = "0.15"; det_size = "960"; safety_scale = "1.1"; face_pct = "40"
                        for prov in _providers_fallback_list(local_providers):
                            print(f"🔁 [{os.path.basename(orig_video_path)}] ghs_crop ultra attempt (orig) with providers: {prov}")
                            if _run_ghs_with(prov, step, conf_thres, det_size, safety_scale, face_pct, use_env_fail=True):
                                face_out_dir = os.path.dirname(face_out_path)
                                _update_csv_face_path(df_path, orig_video_path, face_out_dir)
                                # No csv_elapsed/single_elapsed here as single_start is defined above
                                with lock:
                                    processed_count += 1
                                    print(f"✅ Face extraction completed (orig ultra) ({processed_count}/{len(videos_to_process)}): {os.path.basename(orig_video_path)}")
                                return True, orig_video_path, "success"
                        raise RuntimeError(f"ghs_crop failed after all retries (robust, ultra, orig fallback). See logs in {logs_dir}")
                    else:
                        raise RuntimeError(f"ghs_crop failed after all retries (fast, robust, ultra). See logs in {logs_dir}")
                finally:
                    os.chdir(orig_cwd)
            except Exception as e:
                single_elapsed = time.time() - single_start
                with lock:
                    failed_count += 1
                    print(f"❌ Face extraction failed ({failed_count}): {os.path.basename(video_path)} - {e} (took {single_elapsed:.2f}s)")
                    print(f"⚠️ See logs under {os.path.join(os.path.dirname(face_out_path), '.logs')} for details.")
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


# cut_fixed_chunks_from_video defined above



def cut_reaction_clips_from_csv(root_dir: str, video_path: str, csv_path: str, *, overwrite: bool = False) -> tuple[str, int]:
    """NOTE: This pipeline has been updated to prefer fixed 10-minute chunking; see `cut_fixed_chunks_from_video`.

    Cut reaction clips from a CSV for a given video.

    Reads `csv_path` and uses ffmpeg to cut clips from `input_video` into an output directory named
    `<stem>_clips` under `<root>/clips/<label>/`.

    It will cut **all rows present** in the CSV (filtered only by valid `start_time`/`end_time`).
    Adds/updates a `clip_path` column in the CSV with the output clip file paths.

    Returns (out_dir, number_of_new_clips).
    """
    # Robust stem from the actual basename
    stem = os.path.splitext(os.path.basename(video_path))[0]
    input_video = os.path.join(root_dir, video_path)
    # Infer dataset root/label from the real input path
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

    # Select all rows that have valid non-empty start/end times
    if "start_time" not in df.columns or "end_time" not in df.columns:
        print(f"Skipping {csv_path}: required columns missing (start_time/end_time)")
        return out_dir, 0

    # Preserve original indices so we can write back to the right rows
    selected_mask = (df["start_time"].astype(str).str.strip() != "") & (df["end_time"].astype(str).str.strip() != "")
    selected = df[selected_mask]

    if os.path.isdir(out_dir):
        existing_clips = [f for f in os.listdir(out_dir) if f.lower().endswith(".mp4")]
        if not overwrite and len(existing_clips) == len(selected):
            # All expected clips already exist. Do NOT early-return without updating CSV,
            # otherwise downstream stages (stimuli/masked/bbox) will be skipped due to empty clip_path.
            print(f"Skipping cutting for {video_path}: {len(existing_clips)} clips already present (will map to CSV)")
            try:
                existing_sorted = sorted(existing_clips)
                # Map in order: selected rows -> sorted existing files
                for i, sel_idx in enumerate(selected.index):
                    if i < len(existing_sorted):
                        out_path = os.path.join(out_dir, existing_sorted[i])
                        df.loc[sel_idx, "clip_path"] = out_path
                try:
                    df.to_csv(csv_path, index=False)
                except Exception as e:
                    print(f"Warning: failed to persist CSV while mapping existing clips: {e}")
            except Exception as e:
                print(f"Warning: failed to map existing clips to CSV: {e}")
            return out_dir, 0
        if overwrite and existing_clips:
            # Remove existing outputs to fully re-generate
            try:
                for f in existing_clips:
                    p = os.path.join(out_dir, f)
                    try:
                        os.remove(p)
                    except Exception:
                        pass
                print(f"Overwrite enabled: cleared {len(existing_clips)} existing clip(s) in {out_dir}")
            except Exception as e:
                print(f"Warning: failed to purge existing clips under {out_dir}: {e}")

    os.makedirs(out_dir, exist_ok=True)
    print(f"Cutting clips for {csv_path} -> {out_dir}")

    written = 0
    first_vcodec_printed = False
    for i, row_index in enumerate(selected.index):
        start = str(df.at[row_index, "start_time"]).strip()
        end = str(df.at[row_index, "end_time"]).strip()
        out_path = os.path.join(out_dir, f"{i:04d}.mp4")
        if os.path.exists(out_path):
            if overwrite:
                try:
                    os.remove(out_path)
                except Exception:
                    pass
            else:
                print(f"  Skipping existing clip {out_path}")
                df.loc[row_index, "clip_path"] = out_path
                # persist CSV immediately after updating clip_path
                try:
                    df.to_csv(csv_path, index=False)
                except Exception as e:
                    print(f"  Warning: failed to persist CSV for existing clip {out_path}: {e}")
                continue

        def _hms_to_seconds(t: str) -> float:
            parts = [int(p) for p in str(t).strip().split(":")]
            if len(parts) == 2:
                h, m, s = 0, parts[0], parts[1]
            elif len(parts) == 3:
                h, m, s = parts
            else:
                raise ValueError(f"Invalid time format: {t}")
            return int(h) * 3600 + int(m) * 60 + int(s)

        try:
            dur_s = max(0.0, _hms_to_seconds(end) - _hms_to_seconds(start))
        except Exception:
            dur_s = 0.0

        copy_ok = False
        if dur_s > 0:
            cmd_copy = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
                "-ss", start, "-i", input_video,
                "-t", f"{dur_s:.3f}",
                "-map", "0:v:0", "-map", "0:a:0?",
                "-c", "copy",
                "-movflags", "+faststart",
                "-y", out_path,
            ]
            try:
                subprocess.run(cmd_copy, check=True)
                # Treat very tiny files as failure (e.g., cut landed on non‑keyframe)
                if os.path.exists(out_path) and os.path.getsize(out_path) > 64 * 1024:
                    copy_ok = True
                    print(f"  Wrote (copy) {out_path}")
            except subprocess.CalledProcessError:
                copy_ok = False

        if not copy_ok:
            vcodec_args = _ffmpeg_video_codec_args()
            if not first_vcodec_printed:
                print(f"  Using vcodec args: {' '.join(vcodec_args)}")
                first_vcodec_printed = True
            # Use AAC 128k for compact audio; add faststart for streaming
            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
                "-ss", start, "-i", input_video,
                *( ["-t", f"{dur_s:.3f}"] if dur_s > 0 else [] ),
                *vcodec_args,
                "-c:a", "aac", "-b:a", "128k",
                "-movflags", "+faststart",
                "-map", "0:v:0", "-map", "0:a:0?",
                "-y", out_path,
            ]
            subprocess.run(cmd, check=True)
            print(f"  Wrote (re‑encode) {out_path}")
        written += 1
        df.loc[row_index, "clip_path"] = out_path
        # persist CSV immediately after updating clip_path
        try:
            df.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"  Warning: failed to persist CSV after writing {out_path}: {e}")

    # Save updated DataFrame with clip_path column back to CSV
    df.to_csv(csv_path, index=False)

    return out_dir, written


def cut_fixed_chunks_from_video(root_dir: str, video_path: str, chunk_seconds: int = 600) -> tuple[str, int]:
    """
    Cut the **entire input video** into fixed-length chunks (default: 10 minutes = 600s),
    numbered sequentially (0000.mp4, 0001.mp4, ...), regardless of any CSV.
    Returns (out_dir, number_of_new_clips).
    """
    # Robust stem from the actual basename
    stem = os.path.splitext(os.path.basename(video_path))[0]
    input_video = os.path.join(root_dir, video_path)

    # Infer dataset root/label from the real input path
    root_root, label_folder = _infer_root_and_label(input_video)
    label_folder = _sanitize_filename(label_folder)
    out_dir = os.path.join(root_root, 'clips', label_folder, f"{stem}_clips")

    # Probe duration
    duration = _ffprobe_video_duration_seconds(input_video)
    if duration <= 0:
        print(f"Skipping: could not determine duration for {input_video}")
        return out_dir, 0

    import math
    total_chunks = max(1, int(math.ceil(duration / float(max(1, int(chunk_seconds))))))

    # If clips already exist and match expected count, skip
    if os.path.isdir(out_dir):
        existing = sorted([f for f in os.listdir(out_dir) if f.lower().endswith('.mp4')])
        if len(existing) == total_chunks:
            print(f"Skipping cutting for {video_path}: {len(existing)} fixed {chunk_seconds}s clips already present")
            return out_dir, 0

    os.makedirs(out_dir, exist_ok=True)
    print(f"Cutting fixed {chunk_seconds}s clips for {input_video} -> {out_dir} (expected {total_chunks})")

    written = 0
    first_vcodec_printed = False

    for idx in range(total_chunks):
        start_s = idx * chunk_seconds
        # last chunk runs to the end
        end_s = duration if idx == total_chunks - 1 else min(duration, (idx + 1) * chunk_seconds)
        seg_len = max(0.01, end_s - start_s)

        out_path = os.path.join(out_dir, f"{idx:04d}.mp4")
        if os.path.exists(out_path) and os.path.getsize(out_path) > 64 * 1024:
            print(f"  Skipping existing clip {out_path}")
            continue

        # Try stream copy first (keyframe-aligned segments will succeed)
        cmd_copy = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-ss", f"{start_s:.3f}", "-i", input_video,
            "-t", f"{seg_len:.3f}",
            "-map", "0:v:0", "-map", "0:a:0?",
            "-c", "copy",
            "-movflags", "+faststart",
            "-y", out_path,
        ]
        copy_ok = False
        try:
            subprocess.run(cmd_copy, check=True)
            if os.path.exists(out_path) and os.path.getsize(out_path) > 64 * 1024:
                print(f"  Wrote (copy) {out_path}")
                written += 1
                copy_ok = True
        except subprocess.CalledProcessError:
            copy_ok = False

        if not copy_ok:
            # Fallback to re-encode to guarantee success and keep size small
            vcodec_args = _ffmpeg_video_codec_args()
            if not first_vcodec_printed:
                print(f"  Using vcodec args: {' '.join(vcodec_args)}")
                first_vcodec_printed = True
            cmd_encode = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
                "-ss", f"{start_s:.3f}", "-i", input_video,
                "-t", f"{seg_len:.3f}",
                *vcodec_args,
                "-c:a", "aac", "-b:a", "128k",
                "-movflags", "+faststart",
                "-map", "0:v:0", "-map", "0:a:0?",
                "-y", out_path,
            ]
            subprocess.run(cmd_encode, check=True)
            print(f"  Wrote (re-encode) {out_path}")
            written += 1

    return out_dir, written


def process_from_manifest(
    manifest_csv: str,
    *,
    use_batch_face_extraction: bool = True,
    mode: str = "auto",
    face_only: bool = False,
) -> None:
    """
    Process game reaction videos from a manifest CSV end-to-end.

    Expected columns:
    - selected        : bool    → only rows with True are processed
    - label           : str     → must equal 'game' (case-insensitive) for processing
    - local_path_low  : str     → optional low-res video (if present; for colocating CSVs)
    - local_path      : str     → the hi-res local video file (used for cutting)
    - react_start     : str     → HH:MM:SS reaction window start (auto mode)
    - react_end       : str     → HH:MM:SS reaction window end (auto mode)

    Flow per selected row:
    1) Ensure scene CSV exists (from react_start/react_end; or interactive if mode='manual').
    2) Cut fixed 10-minute clips (ignore CSV start/end).
    3) Layout/bbox + masked/stimuli generation for each cut clip.
    4) Face extraction on masked-or-original clips (batch or single).

    Side effects:
    - Writes scene CSV next to local_path_low if present, else next to local_path, as '<stem>_scene_transitions.csv'.
    - Writes cut clips under 'clips/<label>/<stem>_clips/'.
    - Writes masked/stimuli under 'masked'/'stimuli' mirrors and updates CSV with bbox/layout paths.
    - Writes face crops under 'face/<label>/<stem>_face/' and updates CSV with 'face_out_dir'.
    - Updates manifest with 'df_path' and 'out_dir'.
    """
    if not os.path.exists(manifest_csv):
        print(f"Error: Manifest CSV does not exist: {manifest_csv}")
        sys.exit(1)

    dfm = pd.read_csv(manifest_csv)

    # --- Validate presence of 'selected' column ---
    if "selected" not in dfm.columns:
        print("Error: Manifest must include a 'selected' column (True/False).")
        sys.exit(1)

    # Enforce: only process label == 'game'
    if "label" not in dfm.columns:
        print("Error: Manifest must include a 'label' column equal to 'game'.")
        return

    # Ensure output/progress columns exist (created for whole manifest, but only filled for selected rows)
    for col in ["df_path", "out_dir"]:
        if col not in dfm.columns:
            dfm[col] = ""

    # Helper: selection parsing (supports True/false and the special 'force')
    def _sel_mode(v):
        if isinstance(v, bool):
            return "on" if v else "off"
        if pd.isna(v):
            return "off"
        s = str(v).strip().lower()
        if s == "force":
            return "force"
        if s in {"true", "1", "yes", "y", "t"}:
            return "on"
        return "off"

    # Filter to rows explicitly selected == True AND label == 'game'
    sel_modes = dfm["selected"].map(_sel_mode)
    sel_mask = sel_modes.isin(["on", "force"])
    label_mask = dfm["label"].apply(_is_game_label)
    df_sel = dfm[sel_mask & label_mask].copy()
    if df_sel.empty:
        print("No rows marked selected=True and label='game'; nothing to process.")
        return

    print(f"Processing {len(df_sel)} selected rows (out of {len(dfm)} total).")

    # Cache env configs
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION")
    model_id = os.environ.get("GOOGLE_CLOUD_MODEL_ID")

    # Iterate only over selected rows; leave unselected rows untouched
    for idx, row in df_sel.iterrows():
        # Guard: local_path should exist for selected rows; local_path_low is optional
        low_path = str(row.get("local_path_low", "")).strip()
        hi_path = str(row.get("local_path", "")).strip()

        if not hi_path or not os.path.exists(hi_path):
            print(f"[{idx}] Skipping: selected row missing/invalid local_path: {hi_path}")
            continue

        hi_has_audio = True  # audio-clean removed; skip audio checks

        # Derive paths
        hi_dir = os.path.dirname(hi_path)
        hi_stem = os.path.splitext(os.path.basename(hi_path))[0]
        if low_path and os.path.exists(low_path):
            base_dir = os.path.dirname(low_path)
            base_stem = os.path.splitext(os.path.basename(low_path))[0]
        else:
            base_dir = hi_dir
            base_stem = hi_stem

        df_path = os.path.join(base_dir, f"{base_stem}_scene_transitions.csv")
        # Save df_path into manifest now so that interrupted runs retain progress
        dfm.at[idx, "df_path"] = df_path
        try:
            dfm.to_csv(manifest_csv, index=False)
        except Exception as e:
            print(f"[{idx}] Warning: failed to persist manifest (df_path): {e}")

        # Face-only mode: skip splitting/cutting/bbox; run only face on existing masked or clips
        if face_only:
            # Collect clips from CSV if present, else from expected out_dir
            clips_for_face: list[str] = []
            try:
                if os.path.exists(df_path):
                    dff = pd.read_csv(df_path)
                    if "clip_path" in dff.columns:
                        for _, r in dff.iterrows():
                            cp = str(r.get("clip_path", "")).strip()
                            if not cp:
                                continue
                            mod = cp
                            try:
                                masked_dir = str(r.get("masked_out_dir", "")).strip()
                                if masked_dir:
                                    cand = os.path.join(masked_dir, os.path.basename(cp))
                                    if os.path.exists(cand):
                                        mod = cand
                            except Exception:
                                pass
                            clips_for_face.append(mod)
            except Exception as e:
                print(f"[{idx}] face-only: failed to read clips from CSV: {e}")
            # If CSV not available, derive expected clips dir
            if not clips_for_face:
                root_root, label_folder = _infer_root_and_label(hi_dir)
                label_folder = _sanitize_filename(label_folder)
                out_dir_guess = os.path.join(root_root, 'clips', label_folder, f"{hi_stem}_clips")
                try:
                    for f in sorted(os.listdir(out_dir_guess)):
                        if f.lower().endswith('.mp4'):
                            clips_for_face.append(os.path.join(out_dir_guess, f))
                except Exception:
                    pass
            if not clips_for_face:
                print(f"[{idx}] face-only: no clips found; skipping row.")
                continue
            try:
                talking_head_extraction_batch(
                    clips_for_face,
                    df_path=df_path,
                    fast_mode=True,
                )
            except Exception as e:
                print(f"[{idx}] WARNING: face-only batch failed: {e}")
            continue

        # Determine if this row is forced regeneration
        mode_sel = _sel_mode(row.get("selected", ""))
        is_force = (mode_sel == "force")

        # 1) Build minimal CSV from start/end with <=10min segments
        if mode == "manual":
            print(f"[{idx}] Manual mode selected. Please input start and end time for {hi_path}")
            manual_start = input("Enter start time (HH:MM:SS): ").strip()
            manual_end   = input("Enter end time (HH:MM:SS): ").strip()
            if not (manual_start and manual_end):
                print(f"[{idx}] ERROR: manual mode requires both start and end times. Skipping.")
                continue
            try:
                manual_start = Scene._to_hms(manual_start)
                manual_end   = Scene._to_hms(manual_end)
            except Exception as e:
                print(f"[{idx}] ERROR: invalid time format; expected HH:MM:SS (or MM:SS). Detail: {e}. Skipping.")
                continue
            segments = _split_window_to_segments(manual_start, manual_end, max_len_s=180)
        else:
            s0 = str(row.get("react_start", "")).strip()
            s1 = str(row.get("react_end", "")).strip()
            if not (s0 and s1):
                print(f"[{idx}] ERROR: missing start/end in manifest (need react_start & react_end). Skipping.")
                continue
            try:
                s0 = Scene._to_hms(s0)
                s1 = Scene._to_hms(s1)
            except Exception as e:
                print(f"[{idx}] ERROR: invalid time format in manifest; expected HH:MM:SS (or MM:SS). Detail: {e}. Skipping.")
                continue
            segments = _split_window_to_segments(s0, s1, max_len_s=180)

        minimal_rows = []
        for i, (a, b) in enumerate(segments, start=1):
            minimal_rows.append({
                "scene_number": i,
                "start_time": a,
                "end_time": b,
                "clip_path": "",
                "stimuli_out_dir": "",
                "masked_out_dir": "",
                "bbox_x1": "",
                "bbox_y1": "",
                "bbox_x2": "",
                "bbox_y2": "",
            })
        df_min = pd.DataFrame(minimal_rows, columns=[
            "scene_number","start_time","end_time","clip_path","stimuli_out_dir","masked_out_dir","bbox_x1","bbox_y1","bbox_x2","bbox_y2"
        ])
        try:
            df_min.to_csv(df_path, index=False)
        except Exception as e:
            print(f"[{idx}] ERROR writing minimal CSV {df_path}: {e}")
            continue

        # 2) Cut clips according to the minimal CSV
        try:
            out_dir, written = cut_reaction_clips_from_csv(
                root_dir=hi_dir,
                video_path=os.path.basename(hi_path),
                csv_path=df_path,
                overwrite=is_force,
            )
            dfm.loc[idx, "out_dir"] = out_dir
            try:
                dfm.to_csv(manifest_csv, index=False)
            except Exception:
                pass
        except Exception as e:
            print(f"[{idx}] ERROR during cutting clips: {e}")
            continue

        # 3) Detect layout and compute one global bbox via Gemini for the hi-res input
        detected_layout = _detect_layout_for_video(hi_path, project=project, location=location or "us-central1", model_id=model_id or "gemini-2.0-flash")
        bbox = _compute_global_bbox_or_fallback(
            hi_path,
            project=project,
            location=location or "us-central1",
            model_id=model_id or "gemini-2.0-flash",
            layout=detected_layout,
        )
        x1, y1, x2, y2 = bbox

        # 4) Apply bbox to each produced clip to write stimuli/masked and update CSV columns
        try:
            dff = pd.read_csv(df_path)
        except Exception as e:
            print(f"[{idx}] ERROR: failed to read back CSV {df_path}: {e}")
            continue

        # Ensure required columns exist before writing back bbox/layout
        for col in [
            "stimuli_out_dir",
            "masked_out_dir",
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
            "layout",
        ]:
            if col not in dff.columns:
                dff[col] = ""
        for ridx, r in dff.iterrows():
            cp = str(r.get("clip_path", "")).strip()
            if not cp or not os.path.exists(cp):
                continue
            stimuli_path, masked_path = _apply_bbox_to_clip(
                cp,
                bbox,
                layout=detected_layout,
                skip_if_exists=(not is_force),
                min_ok_bytes=16384,
            )
            dff.at[ridx, "stimuli_out_dir"] = os.path.dirname(stimuli_path)
            dff.at[ridx, "masked_out_dir"]  = os.path.dirname(masked_path)
            dff.at[ridx, "bbox_x1"] = x1
            dff.at[ridx, "bbox_y1"] = y1
            dff.at[ridx, "bbox_x2"] = x2
            dff.at[ridx, "bbox_y2"] = y2
            # Persist detected layout for transparency
            dff.at[ridx, "layout"] = detected_layout
        try:
            dff.to_csv(df_path, index=False)
        except Exception as e:
            print(f"[{idx}] WARNING: failed to update CSV with stimuli/masked/bbox: {e}")

        # 5) Prepare clips for face extraction (prefer masked if available)
        clips_for_face: list[str] = []
        for ridx, r in dff.iterrows():
            cp = str(r.get("clip_path", "")).strip()
            if not cp or not os.path.exists(cp):
                continue
            clip_file = os.path.basename(cp)
            masked_dir = str(r.get("masked_out_dir", "")).strip()
            masked_clip = os.path.join(masked_dir, clip_file) if masked_dir else ""
            if masked_clip and os.path.exists(masked_clip):
                clips_for_face.append(masked_clip)
            else:
                clips_for_face.append(cp)

        # 6) Face extraction (batch or single)
        try:
            if use_batch_face_extraction:
                talking_head_extraction_batch(
                    clips_for_face,
                    df_path=df_path,
                    skip_if_exists=(not is_force),
                    fast_mode=True,
                )
            else:
                for vp in clips_for_face:
                    talking_head_extraction(
                        vp,
                        df_path=df_path,
                        skip_if_exists=(not is_force),
                        fast_mode=True,
                    )
        except Exception as e:
            print(f"[{idx}] WARNING: face extraction stage encountered errors: {e}")

def main() -> None:
    """CLI entry point for processing game reaction manifests.

    Usage examples:
      python -m scripts.pipeline_game /path/to/manifest.csv
      python -m scripts.pipeline_game /path/to/manifest.csv --mode manual --no-batch-face

    Notes:
      - Enforces label == 'game'.
      - Reads segment times from columns 'react_start' and 'react_end' in HH:MM:SS.
    """
    parser = argparse.ArgumentParser(
        prog="pipeline_game",
        description="Process game reaction videos listed in a CSV manifest. Expects label=='game' and times in react_start/react_end (HH:MM:SS).",
    )
    parser.add_argument(
        "manifest",
        help="Path to the manifest CSV.",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "manual"],
        default="auto",
        help="auto = use react_start/react_end from CSV; manual = interactively input times",
    )
    # Batch face extraction toggle (default True); provide --no-batch-face to disable
    try:
        # Python 3.9+ supports BooleanOptionalAction
        from argparse import BooleanOptionalAction  # type: ignore
        parser.add_argument(
            "--batch-face",
            action=BooleanOptionalAction,
            default=True,
            help="Use batch face extraction (default: True). Use --no-batch-face to disable.",
        )
        parser.add_argument(
            "--face-only",
            action=BooleanOptionalAction,
            default=False,
            help="Only run face extraction on existing clips/masked; skip split/cut/bbox stages.",
        )
        args = parser.parse_args()
        batch_face = bool(args.batch_face)
        face_only = bool(args.face_only)
    except Exception:
        # Fallback for older argparse: provide paired flags
        parser.add_argument("--batch-face", dest="batch_face", action="store_true", help="Enable batch face extraction (default)")
        parser.add_argument("--no-batch-face", dest="batch_face", action="store_false", help="Disable batch face extraction")
        parser.add_argument("--face-only", dest="face_only", action="store_true", help="Only run face extraction on existing clips/masked")
        parser.set_defaults(face_only=False)
        parser.set_defaults(batch_face=True)
        args = parser.parse_args()
        batch_face = bool(args.batch_face)
        face_only = bool(args.face_only)

    process_from_manifest(
        args.manifest,
        use_batch_face_extraction=batch_face,
        mode=args.mode,
        face_only=face_only,
    )


if __name__ == "__main__":
    main()
