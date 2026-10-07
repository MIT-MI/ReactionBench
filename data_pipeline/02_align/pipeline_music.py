from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "common"))
import rvb_config as cfg  # noqa: E402  (loads .env, puts every stage dir on sys.path)

import argparse
import hashlib
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

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
os.environ.setdefault("LIBROSA_NO_PYTHON_SOXR", "1")
from google import genai
from google.genai.types import GenerateContentConfig, Part
from pydantic import BaseModel, Field, field_validator

# Try to import AudioSep pipeline from the local repository
# We use absolute paths to ensure reliability regardless of CWD. AudioSep resolves
# some relative checkpoint paths at import time, so chdir into the repo for import.
_AUDIOSEP_ROOT = os.environ.get("AUDIOSEP_ROOT", "AudioSep")  # clone of github.com/Audio-AGI/AudioSep
if _AUDIOSEP_ROOT not in sys.path:
    sys.path.insert(0, _AUDIOSEP_ROOT)
build_audiosep = None  # type: ignore
separate_audio = None  # type: ignore
try:
    _orig = os.getcwd()
    try:
        os.chdir(_AUDIOSEP_ROOT)
        from pipeline import build_audiosep, separate_audio  # type: ignore
    finally:
        os.chdir(_orig)
except Exception as _e:
    build_audiosep = None  # type: ignore
    separate_audio = None  # type: ignore
    print(f"Warning: failed to import AudioSep pipeline from {_AUDIOSEP_ROOT}: {_e}")

# Lazy globals for AudioSep
_AUDIOSEP_MODEL = None
_AUDIOSEP_DEVICE = None


def _get_audiosep_model():
    global _AUDIOSEP_MODEL, _AUDIOSEP_DEVICE
    if _AUDIOSEP_MODEL is not None and _AUDIOSEP_DEVICE is not None:
        return _AUDIOSEP_MODEL, _AUDIOSEP_DEVICE

    if build_audiosep is None or separate_audio is None:
        raise RuntimeError(
            f"AudioSep pipeline is unavailable. Ensure {_AUDIOSEP_ROOT} contains pipeline.py and dependencies."
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_dir = os.environ.get("AUDIOSEP_CHECKPOINT_DIR", os.path.join(_AUDIOSEP_ROOT, "checkpoint"))
    checkpoint_path = os.path.join(checkpoint_dir, "audiosep_base_4M_steps.ckpt")
    config_yaml = os.path.join(_AUDIOSEP_ROOT, "config", "audiosep_base.yaml")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            "Missing AudioSep separation checkpoint: "
            f"{checkpoint_path}. Download `audiosep_base_4M_steps.ckpt` from "
            "https://huggingface.co/spaces/Audio-AGI/AudioSep/tree/main/checkpoint "
            "and place it in that directory."
        )
    if not os.path.exists(config_yaml):
        raise FileNotFoundError(f"Missing config yaml: {config_yaml}")

    # CLAP_Encoder inside AudioSep uses relative checkpoint paths; ensure CWD is repo root
    _orig = os.getcwd()
    try:
        os.chdir(_AUDIOSEP_ROOT)
        model = build_audiosep(
            config_yaml=config_yaml,
            checkpoint_path=str(checkpoint_path),
            device=device,
        )
    finally:
        os.chdir(_orig)

    _AUDIOSEP_MODEL = model
    _AUDIOSEP_DEVICE = device
    return _AUDIOSEP_MODEL, _AUDIOSEP_DEVICE


def cleanup_music_reaction_segments(
    input_video_path: str,
    output_video_like_path: str,
    *,
    skip_if_exists: bool = True,
    min_ok_bytes: int = 16384,
) -> tuple[str, str]:
    """
    Separate the input clip's audio into two stems (music vs commenter voice) using AudioSep.

    Saves two WAV files next to `output_video_like_path`, with suffixes `_music.wav` and `_commenter.wav`.

    Returns (music_wav_path, commenter_wav_path).

    Parameters
    ----------
    input_video_path : str
        Path to the source video to separate.
    output_video_like_path : str
        A path used to derive output filenames (same dir + base name).
    skip_if_exists : bool
        If True (default), re-use already separated stems when non-empty files exist.
        If False, always recompute/overwrite.
    min_ok_bytes : int
        Minimum file size to consider a previously generated WAV file "good".
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found. Please install it, e.g., `sudo apt-get install ffmpeg`.")

    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"Input video not found: {input_video_path}")

    base, _ = os.path.splitext(output_video_like_path)
    music_out = f"{base}_music.wav"
    commenter_out = f"{base}_commenter.wav"

    def _is_ok(p: str) -> bool:
        try:
            return os.path.exists(p) and os.path.getsize(p) >= max(1, int(min_ok_bytes))
        except Exception:
            return False

    # If both stems already exist and look good, short-circuit
    if skip_if_exists and _is_ok(music_out) and _is_ok(commenter_out):
        print(f"AudioSep: stems already exist, skipping: {music_out}, {commenter_out}")
        return music_out, commenter_out

    # Extract raw audio from the input clip (we may still need this if one stem is missing/bad)
    tmp_dir = tempfile.mkdtemp(prefix="audiosep_")
    extracted_wav = os.path.join(tmp_dir, "extracted.wav")

    cmd = [
        "ffmpeg", "-y", "-i", input_video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        extracted_wav,
    ]
    subprocess.run(cmd, check=True)

    # Lazily load the AudioSep model **after** we know we are not skipping,
    # to avoid paying model load cost on already-processed clips.
    model, device = _get_audiosep_model()

    # Run separation for music and/or commenter voice, only if needed
    try:
        if not (skip_if_exists and _is_ok(music_out)):
            separate_audio(
                model,
                extracted_wav,
                text="music, instrumental, songs with human vocals",
                output_file=music_out,
                device=device,
            )
        else:
            print(f"AudioSep: reusing existing music stem: {music_out}")

        if not (skip_if_exists and _is_ok(commenter_out)):
            separate_audio(
                model,
                extracted_wav,
                text="commenter speaking voice, human speech, and vocals",
                output_file=commenter_out,
                device=device,
            )
        else:
            print(f"AudioSep: reusing existing commenter stem: {commenter_out}")
    finally:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

    print(f"Wrote separated audio: {music_out} and {commenter_out}")
    return music_out, commenter_out


class Scene(BaseModel):
    scene_number: int = Field(..., description="1,2,3,... sequential id")
    start_time: str = Field(..., description="HH:MM:SS")
    end_time: str = Field(..., description="HH:MM:SS")
    description: str = Field(..., description="short scene summary")
    is_react: bool = Field(False, description="true if it's a real-time viewer currently continuingly reacting to music, false otherwise")
    is_playing: bool = Field(..., description="true if music is playing; false if paused/muted")

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


class AuditResult(BaseModel):
    is_music_reaction: bool = Field(..., description="true if the video segment shows a person reacting to music")
    is_music_playing_entire_time: bool = Field(..., description="true if music is audibly playing for the entire duration of the segment")
    reason: str = Field("", description="brief justification")


_BASE_INSTRUCTIONS = """You are a **multimodal Music Reaction Segment Detector**. Analyze the video and **output only the real-time music reaction segments**. Non-reaction parts are not important—**do not output them**.

**When to cut:**
- Boundaries only at **(1) reaction start/end** and **(2) music play/pause toggles**.
- Ignore jump cuts/B-roll unless they coincide with the two boundaries above.

**Cues (use both audio & visuals):**
- Audible track, on-screen player/cover/seek bar, play/pause icons, wearing headphones, beat-synced gestures/expressions, immediate commentary tied to the music.
- Host monologue/sponsor bits without active listening → not reaction.

**Timestamp rules:**
- **Be precise** at reaction boundaries and **exact** at play↔pause switch points.
- Timestamps must be within video duration.
- If play/pause changes inside a reaction, **split at the switch**.

**Output format (JSON array; only reaction segments):**
Each object must include:
- `scene_number` (int; 1,2,3…)
- `start_time` (HH:MM:SS)
- `end_time` (HH:MM:SS)
- `description` (short summary of the reaction in this span)
- `is_react` (boolean; **always `true`**)
- `is_playing` (boolean; **true = music playing, false = paused/muted**)
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
    Use Google GenAI (Gemini) to segment a local video into scenes and return a pandas.DataFrame.

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

    # Assemble input parts
    parts: List[Part] = [
        Part.from_text(text=_BASE_INSTRUCTIONS),
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


def evaluate_music_reaction_segments(
    video_path: str,
    *,
    project: Optional[str] = None,
    location: str = "us-central1",
    model_id: str = "gemini-2.0-flash",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_output_tokens: int = 1024,
) -> bool:
    """Return True iff the clip is a music reaction AND music plays the entire time.

    Uses Google GenAI (Gemini) to audit the clip with both audio & visual cues.
    """
    if not os.path.exists(video_path):
        print(f"evaluate_music_reaction_segments: file not found: {video_path}")
        return False

    project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        print("evaluate_music_reaction_segments: missing GCP project (set GOOGLE_CLOUD_PROJECT or pass project=)")
        return False

    client = genai.Client(vertexai=True, project=project, location=location)

    instructions = (
        "You are a strict multimedia auditor.\n"
        "Task: Decide if this entire video clip is (1) a person reacting to music, and (2) the music is playing for the entire duration.\n\n"
        "Definitions:\n"
        "- 'Person reacting to music': on-screen human is clearly listening to a song and giving immediate, real-time reactions (facial expressions, gestures, commentary tied to the track).\n"
        "- 'Music playing entire time': the song is audibly present for 90% of the clip duration (allowing for very brief hard cuts under ~0.2s, or a very short comment).\n\n"
        "Signals to use (both audio & visuals): audible music, headphones/earbuds, visible player/cover/seek bar, beat-synced gestures, commentary referencing what is being heard.\n"
        "Counter-signals: sponsor read, unrelated monologue, silence, muted/ducked music sections, non-music content.\n\n"
        "Output *only* JSON with fields: is_music_reaction (bool), is_music_playing_entire_time (bool), reason (short string)."
    )

    parts: List[Part] = [
        Part.from_text(text=instructions),
        _part_from_local_file(video_path, mime_fallback="video/mp4"),
    ]

    config = GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
        response_mime_type="application/json",
        response_schema=AuditResult,
    )

    try:
        resp = client.models.generate_content(
            model=model_id,
            contents=parts,
            config=config,
        )
    except Exception as e:
        print(f"evaluate_music_reaction_segments: model call failed: {e}, {model_id}, {parts}, {config}")
        return False

    result: Optional[AuditResult] = None
    try:
        result = resp.parsed  # SDK structured parsing
    except Exception:
        result = None

    if result is None:
        # Fallback to parsing raw JSON
        try:
            data = json.loads(getattr(resp, "text", "{}"))
            result = AuditResult(**data)
        except Exception as e:
            print(f"evaluate_music_reaction_segments: parse failed: {e}")
            return False

    # Optional: print reason for debugging
    try:
        print(
            f"Audit: reaction={result.is_music_reaction}, playing_entire_time={result.is_music_playing_entire_time}; "
            f"reason={result.reason}"
        )
    except Exception:
        pass

    return bool(result.is_music_reaction and result.is_music_playing_entire_time)


def _sanitize_filename(filename: str) -> str:
    """Mirror the manifest's sanitize: remove illegal chars, replace spaces, cap length."""
    sanitized = re.sub(r'[\\/*?:"<>|]', "", filename)
    sanitized = sanitized.replace(" ", "_")
    return sanitized[:100]

def _label_key(label: str) -> str:
    """Create a safe, short key from a label for column names and isolation.
    Example: "Music reactions" -> "music_reactions".
    """
    key = re.sub(r"[^a-z0-9]+", "_", label.strip().lower())
    key = key.strip("_")
    return key or "default"

def _infer_root_and_label(path: str) -> tuple[str, str]:
    """
    From any path under the dataset, infer:
      - root directory (the parent of 'raw'/'clips'/'audio_clean_clips'/'stimuli'/'masked'/'face')
      - label subfolder name (immediately under the anchor like 'raw' or 'clips').
    Example:
      /data/human_reaction_data/raw/General_pop-culture_React/0000.mp4
      -> ("/data/human_reaction_data", "General_pop-culture_React")
    """
    parts = os.path.abspath(path).split(os.sep)
    anchors = ("raw", "clips", "audio_clean_clips", "stimuli", "masked", "face")
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


def _resolve_providers(provider_str: Optional[str]) -> tuple[str, bool]:
    """
    Normalize an ONNX Runtime provider string and decide whether GPU execution will be used.

    Returns
    -------
    tuple[str, bool]
        (normalized_provider_string, use_gpu_flag)
    """
    default = "TensorrtExecutionProvider,CUDAExecutionProvider"
    raw = provider_str or default
    tokens = [p.strip() for p in raw.split(",") if p.strip()]
    gpu_tokens = {"CUDAExecutionProvider", "TensorrtExecutionProvider"}

    requested_gpu = any(p in gpu_tokens for p in tokens)
    cuda_available = torch.cuda.is_available()
    use_gpu = cuda_available and requested_gpu

    if requested_gpu and not use_gpu:
        print("⚠️ No CUDA-capable GPU detected; falling back to CPUExecutionProvider.")
    if not use_gpu:
        tokens = [p for p in tokens if p not in gpu_tokens]
        if not tokens or "CPUExecutionProvider" not in tokens:
            tokens.append("CPUExecutionProvider")

    seen: set[str] = set()
    normalized_tokens: list[str] = []
    for token in tokens or ["CPUExecutionProvider"]:
        if token not in seen:
            normalized_tokens.append(token)
            seen.add(token)

    return ",".join(normalized_tokens), use_gpu


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


# === Bounding box extraction for MV panel region ===

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
        "and another region shows the MUSIC VIDEO (MV) player window. "
        "Return the bounding box for the entire MV player frame (include its visible border if any) "
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
            "Return ONLY the bounding box for the MV panel as [ymin, xmin, ymax, xmax] in 0-1000 integers."
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
    """Extract the bounding box of the *music video (MV) player* region from a reaction clip.

    Workflow:
    1) Grab a representative frame (middle of the video) as a JPEG.
    2) Ask Gemini to return a single bbox as [ymin, xmin, ymax, xmax] in 0–1000 coords.
    3) Convert that normalized box to **pixel coordinates** relative to the frame and return `(xmin, ymin, xmax, ymax, masked_out_path)`.

    Returns
    -------
    (xmin, ymin, xmax, ymax, masked_out_path) : tuple
        - xmin, ymin, xmax, ymax: integer pixel coordinates of the detected MV region.
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
        # Write masked version (black out the MV region)
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

    Uses the ghs_crop tool to create a square 1:1 crop focused on the person's head and shoulders.
    Creates a 'face' folder structure similar to audio_clean_clips and clips folders.
    Updates the CSV with the face folder path.

    This function can take either a regular clip or a masked clip produced by bounding_box_extraction.
    Under a masked_face-only policy, callers should pass masked clips exclusively.
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
        Execution providers string for ONNX Runtime (default: "TensorrtExecutionProvider,CUDAExecutionProvider"; falls back to CPU automatically if CUDA is unavailable).
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

    # Masked-face only policy
    if clip_dir_base.endswith("_clips"):
        print(f"🛑 Skipping unmasked clip for face extraction (masked_face only): {video_path}")
        return ""
    elif clip_dir_base.endswith("_masked"):
        face_stem = clip_dir_base + "_face"   # => *_masked_face
    else:
        print(f"🛑 Unknown clip directory suffix (expect *_masked). Skipping: {clip_dir_base}")
        return ""
    face_out_dir = os.path.join(root_root, 'face', label_folder, face_stem)
    # NOTE: Do not create face_out_dir yet; only create it after a successful crop
    face_out_path = os.path.join(face_out_dir, os.path.basename(video_path))

    # Prefer TensorRT on large NVIDIA GPUs when available; fall back to CPU if CUDA is unavailable
    providers_resolved, use_gpu = _resolve_providers(providers)
    print(f"🔧 Using execution providers: {providers_resolved}")

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

        # Write to a temporary file first; only move into place after success
        import tempfile
        _tmp_dir = tempfile.mkdtemp(prefix="ghs_out_")
        _tmp_out = os.path.join(_tmp_dir, os.path.basename(face_out_path))
        cmd = [
            sys.executable, "-m", "ghs_crop",
            "--input", video_path,
            "--output", _tmp_out,
            "--step", step,
            "--conf-thres", conf_thres,
            "--det-size", det_size,
            "--headroom-bias", "0.05",
            "--safety-scale", safety_scale,
            "--face-area-percentile", "60",
            "--providers", providers_resolved,
        ]

        # Change to the talking_head directory to ensure proper imports
        orig_cwd = os.getcwd()
        try:
            os.chdir(talking_head_root)
            # Set environment variables to optimize performance
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "4"  # Limit CPU threads to avoid oversubscription
            env["MKL_NUM_THREADS"] = "4"
            if use_gpu:
                # Enable ONNX Runtime TensorRT EP optimizations when available
                cache_dir = trt_cache_dir or os.path.join(os.path.dirname(video_path), ".trt_cache_ghs")
                os.makedirs(cache_dir, exist_ok=True)
                env["ORT_TENSORRT_ENGINE_CACHE_ENABLE"] = "1"
                env["ORT_TENSORRT_ENGINE_CACHE_PATH"] = cache_dir
                env["ORT_TENSORRT_FP16_ENABLE"] = "1"           # allow FP16 kernels
                env["ORT_TENSORRT_MAX_WORKSPACE_SIZE"] = str(8 * 1024 * 1024 * 1024)  # 8GB
                # Reduce host mem fragmentation on very parallel runs
                env["CUDA_MODULE_LOADING"] = "LAZY"
                # Disallow CPU fallback within ORT by omitting CPU EP and hinting GPU usage
                env["ORT_DISABLE_CPU"] = "1"
                env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
            else:
                env.pop("ORT_TENSORRT_ENGINE_CACHE_ENABLE", None)
                env.pop("ORT_TENSORRT_ENGINE_CACHE_PATH", None)
                env.pop("ORT_TENSORRT_FP16_ENABLE", None)
                env.pop("ORT_TENSORRT_MAX_WORKSPACE_SIZE", None)
                env.pop("CUDA_MODULE_LOADING", None)
                env.pop("ORT_DISABLE_CPU", None)
                env.pop("CUDA_VISIBLE_DEVICES", None)

            ghs_start = time.time()
            print(f"🔄 Running ghs_crop command for {os.path.basename(video_path)}...")

            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                env=env,
                timeout=600,  # 5 minute timeout per video
            )

            ghs_elapsed = time.time() - ghs_start
            total_elapsed = time.time() - start_time
            # Commit result: create final dir only on success
            os.makedirs(face_out_dir, exist_ok=True)
            import shutil
            try:
                shutil.move(_tmp_out, face_out_path)
            finally:
                try:
                    shutil.rmtree(_tmp_dir, ignore_errors=True)
                except Exception:
                    pass
            print(f"✅ Face extraction completed: {face_out_path} (ghs_crop: {ghs_elapsed:.2f}s, total: {total_elapsed:.2f}s)")
        finally:
            os.chdir(orig_cwd)

    except subprocess.TimeoutExpired:
        print(f"Warning: face extraction timed out for {video_path}")
        import shutil
        try:
            shutil.rmtree(_tmp_dir, ignore_errors=True)
        except Exception:
            pass
    except subprocess.CalledProcessError as e:
        print(f"Warning: face extraction failed for {video_path}: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        import shutil
        try:
            shutil.rmtree(_tmp_dir, ignore_errors=True)
        except Exception:
            pass
    except Exception as e:
        print(f"Warning: face extraction failed for {video_path}: {e}")
        import shutil
        try:
            shutil.rmtree(_tmp_dir, ignore_errors=True)
        except Exception:
            pass

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
        Execution providers string for ONNX Runtime (default: "TensorrtExecutionProvider,CUDAExecutionProvider"; falls back to CPU automatically if CUDA is unavailable).
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

        # Masked-face only policy
        if clip_dir_base.endswith("_clips"):
            print(f"🛑 Skipping unmasked clip for face extraction (masked_face only): {video_path}")
            output_paths.append("")
            continue
        elif clip_dir_base.endswith("_masked"):
            face_stem = clip_dir_base + "_face"   # => *_masked_face
        else:
            print(f"🛑 Unknown clip directory suffix (expect *_masked). Skipping: {clip_dir_base}")
            output_paths.append("")
            continue

        face_out_dir = os.path.join(root_root, 'face', label_folder, face_stem)
        # NOTE: Do not create face_out_dir yet; only on successful crop
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

    providers_resolved, use_gpu = _resolve_providers(providers)
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
            print(f"🔧 Using execution providers for {os.path.basename(video_path)}: {providers_resolved}")

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

                import tempfile
                _tmp_dir = tempfile.mkdtemp(prefix="ghs_out_")
                _tmp_out = os.path.join(_tmp_dir, os.path.basename(face_out_path))
                cmd = [
                    sys.executable, "-m", "ghs_crop",
                    "--input", video_path,
                    "--output", _tmp_out,
                    "--step", step,
                    "--conf-thres", conf_thres,
                    "--det-size", det_size,
                    "--headroom-bias", "0.05",
                    "--safety-scale", safety_scale,
                    "--face-area-percentile", "60",
                    "--providers", providers_resolved,
                ]

                # Set environment variables to optimize performance
                env = os.environ.copy()
                env["OMP_NUM_THREADS"] = "2"  # Fewer threads for parallel processing
                env["MKL_NUM_THREADS"] = "2"
                if use_gpu:
                    # Enable ORT TensorRT EP engine caching/fp16 for speed
                    _cache_dir = trt_cache_dir or os.path.join(os.path.dirname(video_path), ".trt_cache_ghs")
                    os.makedirs(_cache_dir, exist_ok=True)
                    env["ORT_TENSORRT_ENGINE_CACHE_ENABLE"] = "1"
                    env["ORT_TENSORRT_ENGINE_CACHE_PATH"] = _cache_dir
                    env["ORT_TENSORRT_FP16_ENABLE"] = "1"
                    env["ORT_TENSORRT_MAX_WORKSPACE_SIZE"] = str(8 * 1024 * 1024 * 1024)
                    env["CUDA_MODULE_LOADING"] = "LAZY"
                    env["ORT_DISABLE_CPU"] = "1"
                    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
                else:
                    env.pop("ORT_TENSORRT_ENGINE_CACHE_ENABLE", None)
                    env.pop("ORT_TENSORRT_ENGINE_CACHE_PATH", None)
                    env.pop("ORT_TENSORRT_FP16_ENABLE", None)
                    env.pop("ORT_TENSORRT_MAX_WORKSPACE_SIZE", None)
                    env.pop("CUDA_MODULE_LOADING", None)
                    env.pop("ORT_DISABLE_CPU", None)
                    env.pop("CUDA_VISIBLE_DEVICES", None)

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
                        timeout=600,
                    )
                    ghs_elapsed = time.time() - ghs_start

                    # Create final dir only after success and move the file
                    import shutil
                    os.makedirs(os.path.dirname(face_out_path), exist_ok=True)
                    try:
                        shutil.move(_tmp_out, face_out_path)
                    finally:
                        try:
                            shutil.rmtree(_tmp_dir, ignore_errors=True)
                        except Exception:
                            pass

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
                import shutil
                try:
                    shutil.rmtree(_tmp_dir, ignore_errors=True)
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


def cut_reaction_clips_from_csv(root_dir: str, video_path: str, csv_path: str) -> tuple[str, int]:
    """Cut reaction clips from a CSV for a given video.

    Reads `csv_path`, filters rows where both `is_react` and `is_playing` are true,
    and uses ffmpeg to cut clips from `input_video` into an output directory named
    `<video_stem>_clips` under `root_dir`.

    Adds a `clip_path` column to the CSV with the output clip file paths.

    Returns (out_dir, number_of_new_clips).
    """
    video_stem = video_path.replace('.mp4', '')
    input_video = os.path.join(root_dir, video_path)
    root_root, label_folder = _infer_root_and_label(root_dir)
    label_folder = _sanitize_filename(label_folder)
    out_dir = os.path.join(root_root, 'clips', label_folder, f"{video_stem}_clips")

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
        out_path = os.path.join(out_dir, f"{idx:04d}.mp4")
        if os.path.exists(out_path):
            print(f"  Skipping existing clip {out_path}")
            df.loc[selected.index[idx], "clip_path"] = out_path
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
            df.loc[selected.index[idx], "clip_path"] = out_path
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
    label: str = "Music reactions",
    fast_skip_local: bool = False,
    face_only: bool = False,
) -> None:
    """
    Process videos listed in a CSV manifest instead of scanning a directory.

    Expected columns in the CSV:
    - selected       : bool -> only rows with True are processed; others are ignored
    - local_path_low : str  -> (present/valid only for selected=True rows) used for detect_scene_transitions_to_df()
    - local_path     : str  -> (present/valid only for selected=True rows) used for cut_reaction_clips_from_csv()
    - subtitle_path  : str  -> (optional; present/valid only for selected=True rows)

    Side effects (on selected=True rows only):
    - Writes the scene detection CSV (df_path) next to local_path_low's directory.
    - Cuts clips and writes them under a sibling 'clips' directory (same behavior as folder mode).
    - Cleans audio for valid clips under a sibling 'audio_clean_clips' directory.
    - Updates the manifest with 'df_path', 'out_dir', and 'clean_out_dir' for each selected row.
    When `fast_skip_local=True`, skip decisions are made *purely from local filesystem artifacts* and the manifest file is not modified.
    When writing progress to the manifest (default), columns are namespaced per-label using the label key (e.g., `music_reactions_df_path`).
    This avoids interference when multiple pipelines share one manifest.
    """
    if not os.path.exists(manifest_csv):
        print(f"Error: Manifest CSV does not exist: {manifest_csv}")
        sys.exit(1)

    dfm = pd.read_csv(manifest_csv)

    label_key = _label_key(label)
    col_df_path = f"{label_key}_df_path"
    col_out_dir = f"{label_key}_out_dir"
    col_clean_out = f"{label_key}_clean_out_dir"

    def _maybe_persist_manifest():
        if not fast_skip_local:
            try:
                dfm.to_csv(manifest_csv, index=False)
            except Exception as e:
                print(f"Warning: failed to persist manifest: {e}")

    # --- Validate presence of 'selected' column ---
    if "selected" not in dfm.columns:
        print("Error: Manifest must include a 'selected' column (True/False).")
        sys.exit(1)

    # Ensure output/progress columns exist (created for whole manifest, but only filled for selected rows)
    if not fast_skip_local:
        for col in [col_df_path, col_out_dir, col_clean_out]:
            if col not in dfm.columns:
                dfm[col] = ""

    # Helper: robust boolean cast (mirrors helper used elsewhere)
    def _to_bool(v):
        if isinstance(v, bool):
            return v
        if pd.isna(v):
            return False
        return str(v).strip().lower() in {"true", "1", "yes", "y", "t"}

    # Filter to rows explicitly selected == True
    df_sel = dfm[dfm["selected"].map(_to_bool)].copy()
    # Further restrict to only the requested label
    if "label" in df_sel.columns and label:
        df_sel = df_sel[df_sel["label"] == label]
    if df_sel.empty:
        print(f"No rows marked selected=True and label=='{label}'; nothing to process.")
        return

    # === Face-only mode: skip detection/cutting/eval/bbox; run only face extraction ===
    if face_only:
        print("\n[Face-only] Running only the face extraction step; skipping scene detection, cutting, evaluation, and bbox.\n")
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION")
        model_id = os.environ.get("GOOGLE_CLOUD_MODEL_ID")

        # Collect candidate clips for face extraction per row
        face_jobs: list[tuple[str, Optional[str]]] = []  # (clip_path, df_path)

        def _expected_paths(low_path: str, hi_path: str) -> tuple[str, str, str, str]:
            """Return (df_path, out_dir, masked_dir, hi_stem)."""
            low_dir = os.path.dirname(low_path)
            low_stem = os.path.splitext(os.path.basename(low_path))[0]
            df_path = os.path.join(low_dir, f"{low_stem}_scene_transitions.csv")

            hi_dir = os.path.dirname(hi_path)
            hi_stem = os.path.splitext(os.path.basename(hi_path))[0]
            root_root, label_folder = _infer_root_and_label(hi_dir)
            label_folder = _sanitize_filename(label_folder)
            out_dir = os.path.join(root_root, 'clips', label_folder, f"{hi_stem}_clips")
            masked_dir = os.path.join(root_root, 'masked', label_folder, f"{hi_stem}_masked")
            return df_path, out_dir, masked_dir, hi_stem

        for idx, row in df_sel.iterrows():
            low_path = str(row.get("local_path_low", "")).strip()
            hi_path = str(row.get("local_path", "")).strip()
            if not hi_path or not os.path.exists(hi_path):
                print(f"[{idx}] [Face-only] Skipping: invalid/missing local_path: {hi_path}")
                continue
            # df_path is optional in face-only mode; used only if present for CSV updates
            df_path, out_dir, masked_dir, hi_stem = _expected_paths(low_path, hi_path) if low_path else ("", "", "", "")

            # masked_face-only policy: only use masked clips; skip unmasked clips entirely
            if not (masked_dir and os.path.isdir(masked_dir)):
                print(f"[{idx}] [Face-only] No masked clip directory: {masked_dir}. Skipping this video per masked_face-only policy.")
                continue

            try:
                for f in sorted(os.listdir(masked_dir)):
                    if not f.lower().endswith('.mp4'):
                        continue
                    clip_path = os.path.join(masked_dir, f)
                    face_jobs.append((clip_path, df_path if df_path and os.path.exists(df_path) else None))
            except Exception as e:
                print(f"[{idx}] [Face-only] Failed to list masked dir {masked_dir}: {e}")
                continue

        if not face_jobs:
            print("[Face-only] No clips found to process. Nothing to do.")
            return

        # Group clips by df_path (so CSV updates, if any, touch the right file)
        clips_by_csv: dict[Optional[str], list[str]] = {}
        for clip_path, df_path in face_jobs:
            clips_by_csv.setdefault(df_path, []).append(clip_path)

        # Run batch face extraction per CSV group
        all_outputs: list[tuple[str, str]] = []  # (input_clip, face_out_path)
        for df_path, clips in clips_by_csv.items():
            print(f"[Face-only] Batch processing {len(clips)} clips (df={df_path or 'None'}) for talking head extraction")
            # We can predict the face_out_path for report building using the same logic as talking_head_extraction
            predicted_outputs: list[str] = []
            for clip in clips:
                root_root, label_folder = _infer_root_and_label(clip)
                label_folder = _sanitize_filename(label_folder)
                clip_dir = os.path.dirname(clip)
                clip_dir_base = os.path.basename(clip_dir)
                if clip_dir_base.endswith("_masked") or os.path.basename(os.path.dirname(clip)).endswith("_masked"):
                    # when clip resides in .../<stem>_masked/
                    masked_dir = os.path.basename(os.path.dirname(clip))
                    face_stem = masked_dir + "_face"   # => *_masked_face
                else:
                    # Do not predict/unify any *_face for non-masked inputs
                    continue

                face_out_dir = os.path.join(root_root, 'face', label_folder, face_stem)
                # NOTE: Do not create face_out_dir here; it's only created on successful crops
                face_out_path = os.path.join(face_out_dir, os.path.basename(clip))
                predicted_outputs.append(face_out_path)
                all_outputs.append((clip, face_out_path))

            # Execute the actual batch
            try:
                talking_head_extraction_batch(
                    clips,
                    df_path=df_path,
                    fast_mode=True,
                    providers="TensorrtExecutionProvider,CUDAExecutionProvider",
                    trt_cache_dir=os.path.join(os.path.dirname(df_path), ".trt_cache") if df_path else None,
                    max_workers=6,
                )
            except Exception as e:
                print(f"[Face-only] Batch face extraction failed for group df={df_path}: {e}. Falling back to individual.")
                for clip in clips:
                    try:
                        talking_head_extraction(clip, df_path=df_path, fast_mode=True)
                    except Exception as e2:
                        print(f"[Face-only] Individual fallback failed for {clip}: {e2}")

        # === Build report by checking file existence/size ===
        def _is_ok(p: str, min_bytes: int = 2048) -> bool:
            if not os.path.exists(p):
                return False
            try:
                # Quick size sanity (very small files are likely corrupt)
                if os.path.getsize(p) < min_bytes:
                    return False
                # Verify that there is at least one video stream
                probe = subprocess.run(
                    [
                        "ffprobe", "-v", "error",
                        "-select_streams", "v:0",
                        "-show_entries", "stream=codec_name",
                        "-of", "default=noprint_wrappers=1:nokey=1",
                        p,
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                return bool(probe)
            except Exception:
                return False

        report_rows = []
        total, success, failed = 0, 0, 0
        for clip_path, face_out_path in all_outputs:
            total += 1
            ok = _is_ok(face_out_path)
            status = "success" if ok else "failed"
            if ok:
                success += 1
            else:
                failed += 1
            report_rows.append({
                "clip_path": clip_path,
                "face_out_path": face_out_path,
                "status": status,
                "size_bytes": os.path.getsize(face_out_path) if os.path.exists(face_out_path) else 0,
            })

        # Write a CSV report next to the manifest
        try:
            report_csv = os.path.join(os.path.dirname(manifest_csv), os.path.splitext(os.path.basename(manifest_csv))[0] + "_face_report.csv")
            pd.DataFrame(report_rows).to_csv(report_csv, index=False)
            print(f"\n[Face-only] Face extraction report written to: {report_csv}")
        except Exception as e:
            print(f"[Face-only] Failed to write report CSV: {e}")

        print(f"\n[Face-only] Summary: total={total}, success={success}, failed={failed}")
        return

    if fast_skip_local:
        print(f"Fast-skip mode enabled (local-only). Manifest will not be modified. Label filter: '{label}' [{_label_key(label)}]")
    else:
        print(f"Manifest progress columns will be namespaced under '{_label_key(label)}' to avoid cross-label interference.")
    print(f"Processing {len(df_sel)} selected rows (out of {len(dfm)} total).")

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

        # Compute expected artifact locations without touching manifest (used for fast local skip)
        low_dir = os.path.dirname(low_path)
        low_stem = os.path.splitext(os.path.basename(low_path))[0]
        hi_dir = os.path.dirname(hi_path)
        hi_stem = os.path.splitext(os.path.basename(hi_path))[0]

        expected_df_path = os.path.join(low_dir, f"{low_stem}_scene_transitions.csv")
        root_root, label_folder = _infer_root_and_label(hi_dir)
        label_folder = _sanitize_filename(label_folder)
        expected_out_dir = os.path.join(root_root, 'clips', label_folder, f"{hi_stem}_clips")
        expected_clean_out_dir = os.path.join(root_root, 'audio_clean_clips', label_folder, f"{hi_stem}_clean")

        if fast_skip_local:
            # Purely-local, O(1) checks: if detection CSV exists AND clips already exist AND clean dir exists -> skip
            has_df = os.path.exists(expected_df_path)
            has_clips = os.path.isdir(expected_out_dir) and any(
                f.lower().endswith('.mp4') for f in os.listdir(expected_out_dir)
            )
            has_clean = os.path.isdir(expected_clean_out_dir)
            if has_df and has_clips and has_clean:
                print(f"[{idx}] ⏩ Fast-skip (local artifacts present): {hi_stem}")
                continue

        # Derive paths (already computed above)
        # low_dir = os.path.dirname(low_path)
        # low_stem = os.path.splitext(os.path.basename(low_path))[0]
        # hi_dir = os.path.dirname(hi_path)
        # hi_stem = os.path.splitext(os.path.basename(hi_path))[0]

        df_path = os.path.join(low_dir, f"{low_stem}_scene_transitions.csv")
        if not fast_skip_local:
            dfm.at[idx, col_df_path] = df_path
            _maybe_persist_manifest()

        # 1) Scene detection on the low-res file (skip if CSV already exists)
        if os.path.exists(df_path):
            print(f"[{idx}] Skipping scene detection; CSV exists: {df_path}")
        else:
            print(f"[{idx}] Detecting scenes for {low_path}")
            try:
                df_detect = detect_scene_transitions_to_df(
                    video_path=low_path,
                    project=project,
                    location=location,
                    model_id=model_id,
                )
            except Exception as e:
                print(f"[{idx}] Scene detection failed for {low_path}: {e}")
                continue
            df_detect.to_csv(df_path, index=False)

        # 2) Cut clips from the high-res file using the detection CSV
        out_dir, written = cut_reaction_clips_from_csv(
            root_dir=hi_dir,
            video_path=os.path.basename(hi_path),
            csv_path=df_path,
        )
        if not fast_skip_local:
            dfm.at[idx, col_out_dir] = out_dir
            _maybe_persist_manifest()
        print(f"[{idx}] Cut {written} clips to {out_dir}")

        # 3) Evaluate & clean audio for passing clips
        root_root, label_folder = _infer_root_and_label(hi_dir)
        label_folder = _sanitize_filename(label_folder)
        clean_out_dir = os.path.join(root_root, 'audio_clean_clips', label_folder, f"{hi_stem}_clean")
        os.makedirs(clean_out_dir, exist_ok=True)
        if not fast_skip_local:
            dfm.at[idx, col_clean_out] = clean_out_dir
            _maybe_persist_manifest()

        # Load scene detection CSV once and ensure eval_pass column exists
        try:
            df_eval = pd.read_csv(df_path)
        except Exception:
            df_eval = pd.DataFrame()

        if not df_eval.empty and "eval_pass" not in df_eval.columns:
            df_eval["eval_pass"] = ""

        # If no clips were produced (e.g., no reaction segments), continue to next row
        if not os.path.isdir(out_dir):
            print(f"[{idx}] No out_dir found (no clips?): {out_dir}")
            continue

        # Collect all clips for this video first
        passing_clips = []
        for clip_file in os.listdir(out_dir):
            if not clip_file.endswith(".mp4"):
                continue
            full_clip = os.path.join(out_dir, clip_file)

            # Determine evaluation result: reuse existing if present, otherwise run evaluation
            passed = None
            if not df_eval.empty and "clip_path" in df_eval.columns and "eval_pass" in df_eval.columns:
                existing = df_eval.loc[df_eval["clip_path"] == full_clip, "eval_pass"]
                if not existing.empty:
                    val = existing.iloc[0]
                    if pd.notna(val) and str(val).strip() != "":
                        passed = str(val).strip().lower() in {"true", "1", "yes", "y", "t"}

            if passed is None:
                passed = evaluate_music_reaction_segments(full_clip)
                if not df_eval.empty and "clip_path" in df_eval.columns:
                    df_eval.loc[df_eval["clip_path"] == full_clip, "eval_pass"] = bool(passed)
                    df_eval.to_csv(df_path, index=False)

            if not passed:
                print(f"[{idx}] Skipping non-passing clip {clip_file}")
                continue

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
                max_workers=3,
            )
        else:
            bbox_map = {}

        # Now clean audio and stage masked clip (if available) for face extraction
        for full_clip in passing_clips:
            clip_file = os.path.basename(full_clip)
            print(f"[{idx}] Cleaning audio for {clip_file}")
            cleanup_music_reaction_segments(full_clip, os.path.join(clean_out_dir, clip_file))

            x1, y1, x2, y2, masked_clip = bbox_map.get(full_clip, (0, 0, 0, 0, ""))
            print(f"Bounding box extraction: {x1}, {y1}, {x2}, {y2}; masked={masked_clip}")
            # masked_face-only policy: require masked clip; skip unmasked
            if masked_clip:
                prepared_clips_for_face.append(masked_clip)
            else:
                print(f"[{idx}] No masked clip available for {clip_file}; skipping face extraction per masked_face-only policy.")

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
                    max_workers=3,
                )

        if not fast_skip_local:
            _maybe_persist_manifest()

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
    parser = argparse.ArgumentParser(description="Process music reaction videos")
    parser.add_argument(
        "--manifest",
        help="CSV manifest containing video paths (uses local_path_low for detection and local_path for cutting)",
        default=None,
    )
    parser.add_argument(
        "--label",
        help="Only process rows whose label exactly matches this string (prevents cross-label interference)",
        default="Music reactions",
    )
    parser.add_argument(
        "--fast-skip-local",
        action="store_true",
        help="Make skip decisions purely from local artifacts and do not write to the manifest",
    )
    parser.add_argument(
        "--face-only",
        action="store_true",
        help="Run only the face extraction step (skip detection, cutting, evaluation, and bbox); outputs a failure report.",
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
            label=args.label,
            fast_skip_local=args.fast_skip_local,
            face_only=args.face_only,
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
            df_detect.to_csv(df_path, index=False)

        # Cut clips from CSVs using helper
        out_dir, written = cut_reaction_clips_from_csv(root_dir=root_dir, video_path=video_path, csv_path=df_path)
        print(f"Cut {written} clips from {video_path} to {out_dir}")

        clean_out_dir = os.path.join(os.path.dirname(root_dir), 'audio_clean_clips', f"{video_path.replace('.mp4', '')}_clean")
        os.makedirs(clean_out_dir, exist_ok=True)

        # Load scene detection CSV once and ensure eval_pass column exists
        try:
            df_eval = pd.read_csv(df_path)
        except Exception:
            df_eval = pd.DataFrame()

        if not df_eval.empty and "eval_pass" not in df_eval.columns:
            df_eval["eval_pass"] = ""

        # Collect passing clips to run face extraction in batch at the end
        _passing_clips_for_bbox = []
        for clip_path in os.listdir(out_dir):
            if not clip_path.endswith(".mp4"):
                continue
            print(f"Processing {clip_path}")
            full_clip = os.path.join(out_dir, clip_path)

            # Determine evaluation result: reuse existing if present, otherwise run evaluation
            passed = None
            if not df_eval.empty and "clip_path" in df_eval.columns and "eval_pass" in df_eval.columns:
                existing = df_eval.loc[df_eval["clip_path"] == full_clip, "eval_pass"]
                if not existing.empty:
                    val = existing.iloc[0]
                    if pd.notna(val) and str(val).strip() != "":
                        passed = str(val).strip().lower() in {"true", "1", "yes", "y", "t"}

            if passed is None:
                passed = evaluate_music_reaction_segments(full_clip)
                # Update the scene detection CSV with the new evaluation result
                if not df_eval.empty:
                    if "clip_path" in df_eval.columns:
                        df_eval.loc[df_eval["clip_path"] == full_clip, "eval_pass"] = bool(passed)
                    # Persist the update
                    df_eval.to_csv(df_path, index=False)

            if not passed:
                print(f"Skipping {clip_path} because it's not a valid music reaction segment")
                continue

            print(f"Evaluating {clip_path}")
            # Defer cleanup and bbox; collect to list first
            _passing_clips_for_bbox = _passing_clips_for_bbox if '_passing_clips_for_bbox' in locals() else []
            _passing_clips_for_bbox.append(full_clip)

        # After the loop, batch bbox and cleanup
        if '_passing_clips_for_bbox' in locals() and _passing_clips_for_bbox:
            print(f"Running parallel bounding-box extraction for {len(_passing_clips_for_bbox)} clips")
            bbox_map = bounding_box_extraction_batch(
                _passing_clips_for_bbox,
                df_path=df_path,
                project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
                location=os.environ.get("GOOGLE_CLOUD_LOCATION") or "us-central1",
                model_id=os.environ.get("GOOGLE_CLOUD_MODEL_ID") or "gemini-2.0-flash",
                max_workers=6,
            )

            # Now run audio cleanup and stage clips for face extraction without re-running bbox
            _clips_for_batch_face = []
            for full_clip in _passing_clips_for_bbox:
                clip_file = os.path.basename(full_clip)
                cleanup_music_reaction_segments(full_clip, os.path.join(clean_out_dir, clip_file))
                x1, y1, x2, y2, masked_clip = bbox_map.get(full_clip, (0, 0, 0, 0, ""))
                print(f"Bounding box extraction: {x1}, {y1}, {x2}, {y2}; masked={masked_clip}")
                _clips_for_batch_face.append(masked_clip if masked_clip else full_clip)
        else:
            _clips_for_batch_face = []

        # Run talking head extraction in batch for all passing clips of this video
        if _clips_for_batch_face:
            print(f"Batch processing {len(_clips_for_batch_face)} clips for talking head extraction")
            try:
                talking_head_extraction_batch(
                    _clips_for_batch_face,
                    df_path=df_path,
                    fast_mode=True,
                    providers="TensorrtExecutionProvider,CUDAExecutionProvider",
                    trt_cache_dir=os.path.join(os.path.dirname(df_path), ".trt_cache"),
                    max_workers=6,
                )
            except Exception as e:
                print(f"Warning: batch talking head extraction failed, falling back to individual processing: {e}")
                for _clip in _clips_for_batch_face:
                    try:
                        talking_head_extraction(_clip, df_path=df_path, fast_mode=True)
                    except Exception as e2:
                        print(f"Individual fallback also failed for {_clip}: {e2}")


if __name__ == "__main__":
    main()
