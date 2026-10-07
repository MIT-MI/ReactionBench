#!/usr/bin/env python3
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "common"))
import rvb_config as cfg  # noqa: E402  (loads .env, puts every stage dir on sys.path)

import argparse
import csv
import itertools
import json
import logging
import mimetypes
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from google import genai
from google.genai.types import GenerateContentConfig, Part
from pydantic import BaseModel, Field
from tqdm.auto import tqdm


class PlaybackAudit(BaseModel):
    is_playing: bool = Field(
        ...,
        description="True only if the video keeps moving/playing for effectively the entire duration.",
    )
    reason: str = Field(default="", description="Concise justification for the decision.")


class FaceAudit(BaseModel):
    is_face_available: bool = Field(
        ...,
        description="True only if a human face is visible in every frame.",
    )
    reason: str = Field(default="", description="Concise justification for the decision.")


_PLAYBACK_PROMPT = """You are a meticulous video quality inspector.
Goal: decide if the provided clip keeps playing/moving the entire time (no extended freezes, blank frames, or static imagery).
Instructions:
- Watch the full video.
- Consider both visual motion and audio continuity if present.
- Treat the clip as NOT playing if the screen freezes, goes black, displays stills for longer than a blink, or the player obviously stalls.
- Short hard cuts or transitions under ~0.2s are acceptable.
- If unsure, answer false.
Respond ONLY with JSON matching the schema: {"is_playing": bool, "reason": string}.
"""

_FACE_PROMPT = """You are verifying a cropped face-track video.
Goal: confirm that a recognizable human face is visible in every frame without disappearing, being fully occluded, or drifting out of frame.
Instructions:
- Watch the full clip carefully.
- Mark false if any frame is blank, shows a different subject, or the face leaves the frame.
- Allow short motion blur but not full occlusion.
- If unsure, answer false.
Respond ONLY with JSON matching the schema: {"is_face_available": bool, "reason": string}.
"""


class _IncrementalCsvWriter:
    _FIELDNAMES = ["id", "is_playing", "is_face_available", "percent_frames_without_face", "percent_freeze_duration"]

    def __init__(self, output_path: str):
        self._output_path = output_path
        self._initialized = False

    def initialize(self) -> None:
        if self._initialized:
            return
        directory = os.path.dirname(self._output_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        needs_header = True
        if os.path.isfile(self._output_path) and os.path.getsize(self._output_path) > 0:
            needs_header = False
        if needs_header:
            with open(self._output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._FIELDNAMES)
                writer.writeheader()
        self._initialized = True

    def append_rows(self, rows: Iterable[Dict[str, object]]) -> None:
        buffered = list(rows)
        if not buffered:
            return
        if not self._initialized:
            self.initialize()
        with open(self._output_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._FIELDNAMES)
            writer.writerows(buffered)



def _load_existing_results(
    output_path: str, fieldnames: List[str]
) -> Tuple[Set[str], Set[str], Set[str], Set[str], Set[str]]:
    existing_ids: Set[str] = set()
    missing_playback_ids: Set[str] = set()
    missing_face_ids: Set[str] = set()
    missing_yunet_ids: Set[str] = set()
    missing_freeze_ids: Set[str] = set()
    if not os.path.isfile(output_path) or os.path.getsize(output_path) <= 0:
        return (
            existing_ids,
            missing_playback_ids,
            missing_face_ids,
            missing_yunet_ids,
            missing_freeze_ids,
        )
    try:
        with open(output_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            header = reader.fieldnames or []
    except Exception as exc:
        logging.warning("Failed to read existing output %s: %s", output_path, exc)
        return (
            existing_ids,
            missing_playback_ids,
            missing_face_ids,
            missing_yunet_ids,
            missing_freeze_ids,
        )

    for row in rows:
        if not row:
            continue
        sample_id = row.get("id")
        if not sample_id:
            continue
        sample_id_str = str(sample_id)
        existing_ids.add(sample_id_str)
        playback_value = (row.get("is_playing") or "").strip()
        face_value = (row.get("is_face_available") or "").strip()
        percent_face_value = (row.get("percent_frames_without_face") or "").strip()
        freeze_value = (row.get("percent_freeze_duration") or "").strip()
        if playback_value == "":
            missing_playback_ids.add(sample_id_str)
        if face_value == "":
            missing_face_ids.add(sample_id_str)
        if percent_face_value == "":
            missing_yunet_ids.add(sample_id_str)
        if freeze_value == "":
            missing_freeze_ids.add(sample_id_str)

    if header != fieldnames:
        logging.info(
            "Updating output CSV header for %s; rewriting existing rows to include new fields.",
            output_path,
        )
        for row in rows:
            for field in fieldnames:
                row.setdefault(field, "")
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    return (
        existing_ids,
        missing_playback_ids,
        missing_face_ids,
        missing_yunet_ids,
        missing_freeze_ids,
    )







def _compute_no_face_percentage(video_path: str, sample_step: int, model_path: Optional[str]) -> Optional[float]:
    sample_step = max(1, int(sample_step))
    try:
        import cv2  # type: ignore
    except ImportError:
        logging.warning("OpenCV is not installed; skipping YuNet evaluation for %s", video_path)
        return None

    if not hasattr(cv2, "FaceDetectorYN"):
        logging.error(
            "OpenCV build does not provide FaceDetectorYN; skipping YuNet evaluation for %s",
            video_path,
        )
        return None

    resolved_model: Optional[str]
    if model_path:
        if os.path.isfile(model_path):
            resolved_model = model_path
        else:
            logging.error(
                "YuNet model not found at %s; skipping YuNet evaluation for %s",
                model_path,
                video_path,
            )
            return None
    else:
        try:
            resolved_model = cv2.data.getFilePath("face_detection_yunet_2023mar.onnx")
        except Exception:
            resolved_model = None
        if not resolved_model or not os.path.isfile(resolved_model):
            logging.error(
                "Unable to locate default YuNet model; supply --yunet-model-path to enable YuNet evaluation.",
            )
            return None

    try:
        detector = cv2.FaceDetectorYN.create(resolved_model, "", (320, 320), 0.3, 0.3, 5000)
    except Exception as exc:
        logging.error("Failed to initialize YuNet detector for %s: %s", video_path, exc)
        return None

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        logging.error("Failed to open video %s for YuNet evaluation", video_path)
        return None

    input_size_set = False
    initial_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    initial_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if initial_width > 0 and initial_height > 0:
        try:
            detector.setInputSize((initial_width, initial_height))
            input_size_set = True
        except Exception as exc:
            logging.error("Failed to set YuNet input size for %s using capture metadata: %s", video_path, exc)
    attempted_input_size = input_size_set

    sampled_frames = 0
    frames_without_face = 0
    frame_index = 0

    try:
        while True:
            success, frame = capture.read()
            if not success or frame is None:
                break

            if frame_index % sample_step == 0:
                sampled_frames += 1
                if not attempted_input_size:
                    height, width = frame.shape[:2]
                    try:
                        detector.setInputSize((width, height))
                        input_size_set = True
                    except Exception as exc:
                        logging.error("Failed to set YuNet input size for %s from first frame: %s", video_path, exc)
                        return None
                    attempted_input_size = True

                try:
                    _, faces = detector.detect(frame)
                except Exception as exc:
                    logging.error(
                        "YuNet detection failed for %s at frame %d: %s",
                        video_path,
                        frame_index,
                        exc,
                    )
                    faces = None

                if faces is None or len(faces) == 0:
                    frames_without_face += 1

            frame_index += 1

        if sampled_frames == 0:
            logging.warning("No frames sampled when evaluating %s with YuNet", video_path)
            return None

        return (frames_without_face / sampled_frames) * 100.0
    finally:
        capture.release()


def _parse_insightface_ctx(device: Optional[str]) -> int:
    if device is None:
        return 0
    token = str(device).strip()
    if not token:
        return 0
    if token.isdigit():
        return int(token)
    lowered = token.lower()
    for prefix in ("cuda:", "gpu:", "device:", "dev:"):
        if lowered.startswith(prefix):
            remainder = token[len(prefix) :]
            if remainder.isdigit():
                return int(remainder)
            break
    logging.debug("Unable to parse InsightFace device '%s'; defaulting to ctx_id=0", token)
    return 0


def _compute_no_face_percentage_insightface(
    video_path: str,
    sample_fps: float,
    det_size: int,
    device: Optional[str],
) -> Optional[float]:
    sample_fps = float(sample_fps)
    if sample_fps <= 0:
        sample_fps = 4.0
    det_size = max(1, int(det_size))

    try:
        import cv2  # type: ignore
    except ImportError:
        logging.warning("OpenCV is not installed; skipping InsightFace evaluation for %s", video_path)
        return None

    try:
        from insightface.app import FaceAnalysis  # type: ignore
    except ImportError:
        logging.warning("InsightFace is not installed; skipping InsightFace evaluation for %s", video_path)
        return None

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    ctx_id = _parse_insightface_ctx(device)
    detector_app = None

    try:
        detector_app = FaceAnalysis(
            name="buffalo_l",
            providers=providers,
            allowed_modules=["detection"],
        )
        detector_app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))
        logging.debug(
            "Initialized InsightFace detector for %s on device %s (ctx_id=%d, det_size=%d)",
            video_path,
            device if device is not None else "auto",
            ctx_id,
            det_size,
        )
    except Exception as exc:
        if device is not None:
            logging.error(
                "Failed to initialize InsightFace detector for %s on device %s: %s",
                video_path,
                device,
                exc,
            )
            return None
        logging.warning(
            "InsightFace GPU initialization failed for %s (%s); falling back to CPU-only.",
            video_path,
            exc,
        )
        try:
            detector_app = FaceAnalysis(
                name="buffalo_l",
                providers=["CPUExecutionProvider"],
                allowed_modules=["detection"],
            )
            detector_app.prepare(ctx_id=0, det_size=(det_size, det_size))
        except Exception as cpu_exc:
            logging.error("Failed to initialize InsightFace detector for %s: %s", video_path, cpu_exc)
            return None

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        logging.error("Failed to open video %s for InsightFace evaluation", video_path)
        return None

    try:
        ms_per_sample = 1000.0 / max(sample_fps, 1e-6)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        ms_per_frame = 1000.0 / fps if fps > 0 else None

        sampled_frames = 0
        frames_without_face = 0
        next_ms = 0.0
        frame_index = 0

        while True:
            success, frame = capture.read()
            if not success or frame is None:
                break

            frame_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            if frame_ms <= 0.0 and ms_per_frame is not None:
                frame_ms = frame_index * ms_per_frame
            elif frame_ms <= 0.0:
                frame_ms = float(frame_index)

            if frame_ms + 1e-6 < next_ms:
                frame_index += 1
                continue

            sampled_frames += 1
            try:
                faces = detector_app.get(frame)  # type: ignore[call-arg]
            except Exception as exc:
                logging.error(
                    "InsightFace detection failed for %s at frame %d: %s",
                    video_path,
                    frame_index,
                    exc,
                )
                faces = None

            if not faces:
                frames_without_face += 1

            next_ms += ms_per_sample
            frame_index += 1

        if sampled_frames == 0:
            logging.warning("No frames sampled when evaluating %s with InsightFace", video_path)
            return None

        return (frames_without_face / sampled_frames) * 100.0
    finally:
        capture.release()

def _compute_freeze_percentage(
    video_path: str,
    noise_level: float,
    freeze_duration: float,
    hwaccel: Optional[str],
    hwaccel_device: Optional[str],
) -> Optional[float]:
    try:
        noise = float(noise_level)
        duration_threshold = float(freeze_duration)
    except Exception:
        logging.error("Invalid freeze detection parameters; skipping freeze analysis for %s", video_path)
        return None

    import subprocess
    import tempfile

    try:
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            text=True,
        )
    except FileNotFoundError as exc:
        logging.error("Required tool missing for freeze detection on %s: %s", video_path, exc)
        return None
    except Exception as exc:
        logging.error("ffprobe invocation failed for %s: %s", video_path, exc)
        return None

    if probe.returncode != 0:
        logging.warning(
            "Skipping freeze detection for %s; ffprobe could not read the video stream.",
            video_path,
        )
        return None

    raw_duration = (probe.stdout or "").strip()
    try:
        video_duration = float(raw_duration)
    except Exception:
        logging.warning(
            "Skipping freeze detection for %s; ffprobe did not return a valid duration (%s).",
            video_path,
            raw_duration,
        )
        return None

    if video_duration <= 0:
        logging.warning("Video duration not positive for %s; skipping freeze analysis", video_path)
        return None

    log_file = None
    try:
        log_file_obj = tempfile.NamedTemporaryFile(delete=False)
        log_file = log_file_obj.name
        log_file_obj.close()

        ffmpeg_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
        ]
        if hwaccel:
            ffmpeg_cmd.extend(["-hwaccel", str(hwaccel)])
            if hwaccel_device:
                ffmpeg_cmd.extend(["-hwaccel_device", str(hwaccel_device)])
        ffmpeg_cmd.extend(
            [
                "-i",
                video_path,
                "-map",
                "0:v:0",
                "-vf",
                f"scale=320:-1,format=gray,boxblur=2:1,freezedetect=n={noise}:d={duration_threshold},metadata=print",
                "-an",
                "-f",
                "null",
                "-",
            ]
        )

        with open(log_file, "w") as log_handle:
            result = subprocess.run(
                ffmpeg_cmd,
                stdout=subprocess.DEVNULL,
                stderr=log_handle,
                check=False,
            )
        if result.returncode != 0:
            logging.error(
                "ffmpeg freezedetect failed for %s with code %s", video_path, result.returncode
            )
            return None

        sum_freeze = 0.0
        last_start = None
        freeze_starts = 0
        freeze_ends = 0

        with open(log_file, "r") as log_handle:
            for line in log_handle:
                line = line.strip()
                if "freeze_duration" in line:
                    try:
                        if ":" in line:
                            token = line.split(":", 1)[1]
                        elif "=" in line:
                            token = line.split("=", 1)[1]
                        else:
                            continue
                        token = token.strip()
                        sum_freeze += float(token)
                        freeze_ends += 1
                    except Exception:
                        continue
                elif "freeze_start" in line:
                    if ":" in line:
                        token = line.split(":", 1)[1]
                    elif "=" in line:
                        token = line.split("=", 1)[1]
                    else:
                        continue
                    token = token.strip()
                    freeze_starts += 1
                    try:
                        last_start = float(token)
                    except Exception:
                        last_start = None

        if freeze_starts > 0 and freeze_ends < freeze_starts and last_start is not None:
            tail = max(0.0, video_duration - last_start) if video_duration - last_start >= duration_threshold else 0.0
            sum_freeze += tail

        percentage = (sum_freeze / video_duration) * 100.0
        return max(0.0, min(percentage, 100.0))
    except FileNotFoundError as exc:
        logging.error("Required tool missing for freeze detection on %s: %s", video_path, exc)
        return None
    except Exception as exc:
        logging.error("Unexpected error during freeze detection for %s: %s", video_path, exc)
        return None
    finally:
        if log_file:
            try:
                os.remove(log_file)
            except OSError:
                pass


def _format_percent(value: float) -> str:
    formatted = f"{value:.6f}".rstrip("0").rstrip(".")
    return formatted or "0"


def _apply_csv_updates(
    output_path: str,
    fieldnames: List[str],
    column_updates: Dict[str, Dict[str, str]],
) -> int:
    if not column_updates:
        return 0

    filtered_updates: Dict[str, Dict[str, str]] = {
        column: {sample_id: value for sample_id, value in updates.items() if value != ""}
        for column, updates in column_updates.items()
        if column in fieldnames and updates
    }

    if not filtered_updates:
        return 0

    if not os.path.isfile(output_path):
        logging.warning("Cannot apply updates; output file missing: %s", output_path)
        return 0

    try:
        with open(output_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception as exc:
        logging.warning("Failed to read %s when applying updates: %s", output_path, exc)
        return 0

    if not rows:
        return 0

    updated = 0
    for row in rows:
        sample_id = row.get("id")
        if not sample_id:
            continue
        key = str(sample_id)
        changed = False
        for column, updates in filtered_updates.items():
            if key not in updates:
                continue
            new_value = updates[key]
            current_value = (row.get(column) or "").strip()
            if current_value == new_value:
                continue
            row[column] = new_value
            changed = True
        if changed:
            updated += 1

    if not updated:
        return 0

    for row in rows:
        for field in fieldnames:
            row.setdefault(field, "")

    try:
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    except Exception as exc:
        logging.error("Failed to write updates back to %s: %s", output_path, exc)
        return 0

    return updated




def _part_from_local_file(path: str, *, mime_fallback: str = "application/octet-stream") -> Part:
    mime, _ = mimetypes.guess_type(path)
    if not mime:
        mime = mime_fallback
    with open(path, "rb") as f:
        data = f.read()
    return Part.from_bytes(data=data, mime_type=mime)


def _load_client(project: Optional[str], location: str) -> genai.Client:
    project_id = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        raise ValueError("Missing project ID. Supply --project or set GOOGLE_CLOUD_PROJECT.")
    return genai.Client(vertexai=True, project=project_id, location=location)


def _call_gemini(
    *,
    client: genai.Client,
    model_id: str,
    prompt: str,
    video_path: str,
    schema_type: type[BaseModel],
    temperature: float,
    top_p: float,
    max_output_tokens: int,
) -> Optional[BaseModel]:
    parts: List[Part] = [
        Part.from_text(text=prompt),
        _part_from_local_file(video_path, mime_fallback="video/mp4"),
    ]
    config = GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
        response_mime_type="application/json",
        response_schema=schema_type,
    )
    try:
        response = client.models.generate_content(
            model=model_id,
            contents=parts,
            config=config,
        )
    except Exception as exc:
        logging.error("Gemini request failed for %s: %s", video_path, exc)
        return None

    parsed: Optional[BaseModel] = None
    try:
        parsed = response.parsed
    except Exception:
        parsed = None

    if parsed is not None:
        return parsed

    raw = getattr(response, "text", "") or ""
    if not raw.strip():
        logging.error("Gemini returned empty response for %s", video_path)
        return None

    try:
        data = json.loads(raw)
        return schema_type(**data)
    except Exception as exc:
        logging.error("Failed to parse Gemini JSON for %s: %s (raw=%s)", video_path, exc, raw)
        return None


def _collect_video_map(directory: str) -> Dict[str, str]:
    entries: Dict[str, str] = {}
    for name in os.listdir(directory):
        full_path = os.path.join(directory, name)
        if not os.path.isfile(full_path):
            continue
        base, ext = os.path.splitext(name)
        if ext.lower() not in {".mp4", ".mov", ".mkv"}:
            continue
        entries[base] = full_path
    return entries


def _evaluate_single_sample(
    sample_id: str,
    *,
    client: genai.Client,
    model_id: str,
    stimuli_path: Optional[str],
    face_path: Optional[str],
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    face_sample_step: int,
    yunet_model_path: Optional[str],
    face_process_mode: str,
    face_device: Optional[str],
    face_insight_sample_fps: float,
    face_insight_det_size: int,
    freeze_noise: float,
    freeze_duration: float,
    freeze_hwaccel: Optional[str],
    freeze_hwaccel_device: Optional[str],
    run_playback: bool,
    run_face: bool,
    run_yunet: bool,
    run_freeze: bool,
) -> Dict[str, object]:
    logging.debug("Evaluating sample %s", sample_id)

    is_playing: object = ""
    if run_playback:
        if stimuli_path:
            playback = _call_gemini(
                client=client,
                model_id=model_id,
                prompt=_PLAYBACK_PROMPT,
                video_path=stimuli_path,
                schema_type=PlaybackAudit,
                temperature=temperature,
                top_p=top_p,
                max_output_tokens=max_output_tokens,
            )
            if isinstance(playback, PlaybackAudit):
                is_playing = bool(playback.is_playing)
            else:
                logging.warning("Playback audit unavailable for %s; defaulting to False", sample_id)
                is_playing = False
        else:
            logging.warning("Stimuli video missing for %s; defaulting is_playing to False", sample_id)
            is_playing = False

    is_face_available: object = ""
    if run_face:
        if face_path:
            face_audit = _call_gemini(
                client=client,
                model_id=model_id,
                prompt=_FACE_PROMPT,
                video_path=face_path,
                schema_type=FaceAudit,
                temperature=temperature,
                top_p=top_p,
                max_output_tokens=max_output_tokens,
            )
            if isinstance(face_audit, FaceAudit):
                is_face_available = bool(face_audit.is_face_available)
            else:
                logging.warning("Face audit unavailable for %s; defaulting to False", sample_id)
                is_face_available = False
        else:
            logging.warning("Face video missing for %s; defaulting is_face_available to False", sample_id)
            is_face_available = False

    percent_without_face: Optional[float] = None
    if run_yunet:
        if face_path:
            mode_label = "InsightFace" if face_process_mode == "insightface" else "YuNet"
            if face_process_mode == "insightface":
                percent_without_face = _compute_no_face_percentage_insightface(
                    face_path,
                    face_insight_sample_fps,
                    face_insight_det_size,
                    face_device,
                )
            else:
                percent_without_face = _compute_no_face_percentage(
                    face_path,
                    face_sample_step,
                    yunet_model_path,
                )
            if percent_without_face is None:
                logging.warning(
                    "%s evaluation unavailable for %s; leaving percent_frames_without_face empty",
                    mode_label,
                    sample_id,
                )
        else:
            logging.warning("Face video missing for %s; cannot compute face coverage percentage", sample_id)

    percent_freeze: Optional[float] = None
    if run_freeze:
        if stimuli_path:
            percent_freeze = _compute_freeze_percentage(
                stimuli_path,
                freeze_noise,
                freeze_duration,
                freeze_hwaccel,
                freeze_hwaccel_device,
            )
            if percent_freeze is None:
                logging.warning(
                    "Freeze detection unavailable for %s; leaving percent_freeze_duration empty",
                    sample_id,
                )
        else:
            logging.warning("Stimuli video missing for %s; cannot compute freeze percentage", sample_id)

    return {
        "id": sample_id,
        "is_playing": is_playing,
        "is_face_available": is_face_available,
        "percent_frames_without_face": (
            _format_percent(percent_without_face)
            if percent_without_face is not None
            else ""
        ),
        "percent_freeze_duration": (
            _format_percent(percent_freeze)
            if percent_freeze is not None
            else ""
        ),
    }


def _evaluate_dataset(
    *,
    client: genai.Client,
    model_id: str,
    stimuli_map: Dict[str, str],
    face_map: Dict[str, str],
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    face_sample_step: int,
    yunet_model_path: Optional[str],
    face_process_mode: str,
    face_insight_sample_fps: float,
    face_insight_det_size: int,
    freeze_noise: float,
    freeze_duration: float,
    freeze_hwaccel: Optional[str],
    freeze_hwaccel_devices: Optional[Sequence[str]],
    existing_ids: Set[str],
    missing_playback_ids: Set[str],
    missing_face_ids: Set[str],
    missing_yunet_ids: Set[str],
    missing_freeze_ids: Set[str],
    min_id: Optional[int],
    output_path: str,
    flush_num: int,
    flush_seconds: float,
    num_workers: int,
    yunet_num_workers: Optional[int],
    freeze_num_workers: Optional[int],
    run_playback: bool,
    run_face: bool,
    run_yunet: bool,
    run_freeze: bool,
) -> Tuple[int, int]:
    if not (run_playback or run_face or run_yunet or run_freeze):
        logging.info("No evaluation modes enabled; nothing to do.")
        return 0, 0

    def _should_skip(sample_id: str) -> bool:
        if min_id is None:
            return False
        try:
            return int(sample_id) < min_id
        except ValueError:
            logging.debug("Sample id %s is non-numeric; ignoring min_id filter", sample_id)
            return False

    all_ids = sorted(sample_id for sample_id in (set(stimuli_map) | set(face_map)) if not _should_skip(sample_id))
    writer = _IncrementalCsvWriter(output_path)
    writer.initialize()

    existing_ids = {sample_id for sample_id in existing_ids if not _should_skip(sample_id)}
    missing_playback_ids = {sample_id for sample_id in missing_playback_ids if not _should_skip(sample_id)}
    missing_face_ids = {sample_id for sample_id in missing_face_ids if not _should_skip(sample_id)}
    missing_yunet_ids = {sample_id for sample_id in missing_yunet_ids if not _should_skip(sample_id)}
    missing_freeze_ids = {sample_id for sample_id in missing_freeze_ids if not _should_skip(sample_id)}

    new_ids = [sample_id for sample_id in all_ids if sample_id not in existing_ids]

    if (run_playback or run_face) and existing_ids:
        logging.info(
            "Skipping Gemini evaluation for %d samples already recorded in %s",
            len(existing_ids),
            output_path,
        )

    use_insightface = face_process_mode == "insightface"
    coverage_label = "InsightFace" if use_insightface else "YuNet"

    yunet_only_pending: List[str] = []
    if run_yunet:
        for sample_id in sorted(missing_yunet_ids):
            if sample_id in existing_ids and sample_id not in new_ids:
                face_video = face_map.get(sample_id)
                if face_video:
                    yunet_only_pending.append(sample_id)
                else:
                    logging.warning(
                        "Face video missing for %s; cannot compute %s percentage on existing entry.",
                        sample_id,
                        coverage_label,
                    )
        if yunet_only_pending:
            logging.info(
                "Recomputing %s coverage for %d existing samples lacking percentages.",
                coverage_label,
                len(yunet_only_pending),
            )

    freeze_only_pending: List[str] = []
    if run_freeze:
        for sample_id in sorted(missing_freeze_ids):
            if sample_id in existing_ids and sample_id not in new_ids:
                stimuli_video = stimuli_map.get(sample_id)
                if stimuli_video:
                    freeze_only_pending.append(sample_id)
                else:
                    logging.warning(
                        "Stimuli video missing for %s; cannot compute freeze percentage on existing entry.",
                        sample_id,
                    )
        if freeze_only_pending:
            logging.info(
                "Recomputing freeze coverage for %d existing samples lacking percentages.",
                len(freeze_only_pending),
            )

    total_tasks = len(new_ids)
    if run_yunet:
        total_tasks += len(yunet_only_pending)
    if run_freeze:
        total_tasks += len(freeze_only_pending)

    if total_tasks == 0:
        if all_ids:
            logging.info(
                "No new samples or pending updates; existing results cover all %d entries.",
                len(all_ids),
            )
        else:
            logging.info("No samples to evaluate; wrote header to %s", output_path)
        return 0, 0

    flush_num = max(0, int(flush_num))
    flush_seconds = max(0.0, float(flush_seconds))
    new_workers = max(1, int(num_workers))
    effective_yunet_workers = yunet_num_workers if yunet_num_workers is not None else new_workers
    yunet_workers = max(1, int(effective_yunet_workers))
    effective_freeze_workers = freeze_num_workers if freeze_num_workers is not None else new_workers
    freeze_workers = max(1, int(effective_freeze_workers))

    pending_rows: List[Dict[str, object]] = []
    pending_updates: Dict[str, Dict[str, str]] = {}
    total_written = 0
    total_updated = 0
    last_flush = time.monotonic()

    gpu_devices = [device for device in (freeze_hwaccel_devices or []) if device]
    if (run_freeze or (run_yunet and use_insightface)) and gpu_devices:
        logging.info(
            "Rotating GPU devices for freeze/face tasks: %s",
            ", ".join(gpu_devices),
        )

    freeze_device_cycle = itertools.cycle(gpu_devices if gpu_devices else [None]) if run_freeze else None
    face_device_cycle = (
        itertools.cycle(gpu_devices if gpu_devices else [None]) if run_yunet and use_insightface else None
    )

    def _pick_freeze_device() -> Optional[str]:
        if not run_freeze or freeze_device_cycle is None:
            return None
        return next(freeze_device_cycle)

    def _pick_face_device() -> Optional[str]:
        if not run_yunet or not use_insightface or face_device_cycle is None:
            return None
        return next(face_device_cycle)

    def _queue_update(column: str, sample_id: str, value: Optional[float]) -> None:
        if value is None:
            return
        formatted = _format_percent(value)
        updates = pending_updates.setdefault(column, {})
        updates[sample_id] = formatted

    def _flush_pending(now: Optional[float] = None) -> None:
        nonlocal pending_rows, pending_updates, total_written, total_updated, last_flush
        flushed_anything = False

        if pending_rows:
            flushed_count = len(pending_rows)
            writer.append_rows(pending_rows)
            total_written += flushed_count
            pending_rows.clear()
            flushed_anything = True
            logging.info("Flushed %d rows (total %d) to %s", flushed_count, total_written, output_path)

        if pending_updates:
            updates_copy = {column: dict(values) for column, values in pending_updates.items() if values}
            pending_updates.clear()
            if updates_copy:
                updated_count = _apply_csv_updates(
                    output_path,
                    _IncrementalCsvWriter._FIELDNAMES,
                    updates_copy,
                )
                if updated_count > 0:
                    total_updated += updated_count
                    flushed_anything = True
                    logging.info(
                        "Applied column updates for %d rows (total %d) in %s",
                        updated_count,
                        total_updated,
                        output_path,
                    )

        if flushed_anything:
            last_flush = now if now is not None else time.monotonic()

    with ThreadPoolExecutor(max_workers=new_workers) as new_executor, ThreadPoolExecutor(max_workers=yunet_workers) as yunet_executor, ThreadPoolExecutor(max_workers=freeze_workers) as freeze_executor:
        futures: Dict[Future, Tuple[str, str]] = {}

        for sample_id in new_ids:
            futures[
                new_executor.submit(
                    _evaluate_single_sample,
                    sample_id,
                    client=client,
                    model_id=model_id,
                    stimuli_path=stimuli_map.get(sample_id),
                    face_path=face_map.get(sample_id),
                    temperature=temperature,
                    top_p=top_p,
                    max_output_tokens=max_output_tokens,
                    face_sample_step=face_sample_step,
                    yunet_model_path=yunet_model_path,
                    face_process_mode=face_process_mode,
                    face_device=_pick_face_device(),
                    face_insight_sample_fps=face_insight_sample_fps,
                    face_insight_det_size=face_insight_det_size,
                    freeze_noise=freeze_noise,
                    freeze_duration=freeze_duration,
                    freeze_hwaccel=freeze_hwaccel,
                    freeze_hwaccel_device=_pick_freeze_device(),
                    run_playback=run_playback,
                    run_face=run_face,
                    run_yunet=run_yunet,
                    run_freeze=run_freeze,
                )
            ] = ("new", sample_id)

        if run_yunet:
            for sample_id in yunet_only_pending:
                face_video = face_map.get(sample_id)
                if not face_video:
                    continue
                if use_insightface:
                    futures[
                        yunet_executor.submit(
                            _compute_no_face_percentage_insightface,
                            face_video,
                            face_insight_sample_fps,
                            face_insight_det_size,
                            _pick_face_device(),
                        )
                    ] = ("yunet", sample_id)
                else:
                    futures[
                        yunet_executor.submit(
                            _compute_no_face_percentage,
                            face_video,
                            face_sample_step,
                            yunet_model_path,
                        )
                    ] = ("yunet", sample_id)

        if run_freeze:
            for sample_id in freeze_only_pending:
                stimuli_video = stimuli_map.get(sample_id)
                if not stimuli_video:
                    continue
                futures[
                    freeze_executor.submit(
                        _compute_freeze_percentage,
                        stimuli_video,
                        freeze_noise,
                        freeze_duration,
                        freeze_hwaccel,
                        _pick_freeze_device(),
                    )
                ] = ("freeze", sample_id)

        with tqdm(total=total_tasks, desc="Quality checks", unit="video", disable=total_tasks == 0) as progress:
            for future in as_completed(futures):
                task_kind, sample_id = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    logging.exception("Unhandled error evaluating %s: %s", sample_id, exc)
                    if task_kind == "new":
                        row = {
                            "id": sample_id,
                            "is_playing": False if run_playback else "",
                            "is_face_available": False if run_face else "",
                            "percent_frames_without_face": "",
                            "percent_freeze_duration": "",
                        }
                        pending_rows.append(row)
                    progress.update(1)
                    continue

                if task_kind == "new":
                    row = result if isinstance(result, dict) else None
                    if not row:
                        row = {
                            "id": sample_id,
                            "is_playing": False if run_playback else "",
                            "is_face_available": False if run_face else "",
                            "percent_frames_without_face": "",
                            "percent_freeze_duration": "",
                        }
                    else:
                        row.setdefault("is_playing", "" if not run_playback else False)
                        row.setdefault("is_face_available", "" if not run_face else False)
                        row.setdefault("percent_frames_without_face", "" if not run_yunet else "")
                        row.setdefault("percent_freeze_duration", "" if not run_freeze else "")
                    pending_rows.append(row)
                elif task_kind == "yunet":
                    percent = result
                    if percent is None:
                        logging.warning(
                            "%s evaluation unavailable for %s; leaving percent_frames_without_face empty",
                            coverage_label,
                            sample_id,
                        )
                    else:
                        _queue_update("percent_frames_without_face", sample_id, percent)
                else:  # freeze
                    percent = result
                    if percent is None:
                        logging.warning(
                            "Freeze detection unavailable for %s; leaving percent_freeze_duration empty",
                            sample_id,
                        )
                    else:
                        _queue_update("percent_freeze_duration", sample_id, percent)

                progress.update(1)

                now = time.monotonic()
                pending_total = len(pending_rows) + sum(len(v) for v in pending_updates.values())
                flush_due = False
                if flush_num > 0 and pending_total >= flush_num:
                    flush_due = True
                if (
                    not flush_due
                    and flush_seconds > 0
                    and (now - last_flush) >= flush_seconds
                    and pending_total > 0
                ):
                    flush_due = True

                if flush_due:
                    _flush_pending(now=now)

    _flush_pending()
    return total_written, total_updated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quality-check reaction dataset videos with Gemini.")
    parser.add_argument("--stimuli-dir", required=True, help="Directory containing source stimuli videos.")
    parser.add_argument("--face-dir", required=True, help="Directory containing face track videos.")
    parser.add_argument("--output", required=True, help="Path to write the CSV report.")
    parser.add_argument("--project", default=None, help="Google Cloud project ID (defaults to GOOGLE_CLOUD_PROJECT env var).")
    parser.add_argument("--location", default="us-central1", help="Vertex AI region.")
    parser.add_argument("--model", default="gemini-2.0-flash", help="Gemini model ID.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p nucleus sampling parameter.")
    parser.add_argument("--max-output-tokens", type=int, default=1024, help="Maximum tokens for the JSON response.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    parser.add_argument("--flush-num", type=int, default=25, help="Flush accumulated rows to CSV after this many results (0 disables count-based flushing).")
    parser.add_argument("--flush-seconds", type=float, default=120.0, help="Flush accumulated rows to CSV after this many seconds since the last flush (0 disables time-based flushing).")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of worker threads for parallel Gemini calls.")
    parser.add_argument(
        "--min-id",
        type=int,
        default=None,
        help="Skip evaluating samples whose numeric id is lower than this value.",
    )
    parser.add_argument(
        "--face-sample-step",
        type=int,
        default=5,
        help="Sample every N frames (>=1) when estimating face coverage with YuNet.",
    )
    parser.add_argument(
        "--face-process-mode",
        choices=["yunet", "insightface"],
        default="yunet",
        help="Backend used for face coverage percentage: yunet (CPU) or insightface (GPU-capable).",
    )
    parser.add_argument(
        "--face-insight-fps",
        type=float,
        default=4.0,
        help="Target samples per second when using InsightFace for face coverage.",
    )
    parser.add_argument(
        "--face-insight-det-size",
        type=int,
        default=320,
        help="Detector input size (square) when using InsightFace for face coverage.",
    )
    parser.add_argument(
        "--yunet-model-path",
        default=None,
        help="Optional path to a YuNet ONNX model; defaults to the OpenCV sample model if available.",
    )
    parser.add_argument(
        "--yunet-num-workers",
        type=int,
        default=None,
        help="Override the number of worker threads used for YuNet-only jobs (defaults to --num-workers).",
    )
    parser.add_argument(
        "--freeze-num-workers",
        type=int,
        default=None,
        help="Override worker threads used for freeze-only jobs (defaults to --num-workers).",
    )
    parser.add_argument(
        "--freeze-noise",
        type=float,
        default=0.03,
        help="Noise level threshold passed to ffmpeg freezedetect (n parameter).",
    )
    parser.add_argument(
        "--freeze-duration",
        type=float,
        default=1.0,
        help="Minimum freeze duration in seconds passed to ffmpeg freezedetect (d parameter).",
    )
    parser.add_argument(
        "--freeze-hwaccel",
        default=None,
        help="Optional ffmpeg hardware acceleration flag (e.g., cuda, qsv) for freeze detection.",
    )
    parser.add_argument(
        "--freeze-hwaccel-device",
        default=None,
        help="Optional hardware acceleration device specifier passed to ffmpeg (comma-separated for multiple devices).",
    )
    parser.add_argument(
        "--only",
        choices=["playback", "face", "yunet", "freeze", "freeze-gpu", "non-gemini"],
        default=None,
        help="Limit evaluation scope. Options: playback, face, yunet (face coverage), freeze, freeze-gpu, or non-gemini (run yunet+freeze only).",
    )
    parser.add_argument(
        "--no-gemini",
        action="store_true",
        help="Disable Gemini evaluations (playback and face audit) regardless of --only."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    stimuli_dir = os.path.abspath(args.stimuli_dir)
    face_dir = os.path.abspath(args.face_dir)

    if not os.path.isdir(stimuli_dir):
        raise FileNotFoundError(f"Stimuli directory not found: {stimuli_dir}")
    if not os.path.isdir(face_dir):
        raise FileNotFoundError(f"Face directory not found: {face_dir}")

    client = _load_client(args.project, args.location)

    stimuli_map = _collect_video_map(stimuli_dir)
    face_map = _collect_video_map(face_dir)

    if not stimuli_map:
        logging.warning("No stimuli videos found in %s", stimuli_dir)
    if not face_map:
        logging.warning("No face videos found in %s", face_dir)

    if args.min_id is not None:
        logging.info("Skipping evaluation for sample ids < %d", args.min_id)

    output_path = os.path.abspath(args.output)
    (
        existing_ids,
        missing_playback_ids,
        missing_face_ids,
        missing_yunet_ids,
        missing_freeze_ids,
    ) = _load_existing_results(output_path, _IncrementalCsvWriter._FIELDNAMES)

    freeze_hwaccel = args.freeze_hwaccel
    face_process_mode = args.face_process_mode
    face_insight_sample_fps = args.face_insight_fps if args.face_insight_fps > 0 else 4.0
    if face_process_mode == "insightface" and args.face_insight_fps <= 0:
        logging.warning(
            "Invalid --face-insight-fps (%s); defaulting to 4.0 fps for InsightFace sampling.",
            args.face_insight_fps,
        )
    face_insight_det_size = max(1, int(args.face_insight_det_size))
    freeze_hwaccel_devices: Optional[List[str]] = None
    raw_freeze_devices = args.freeze_hwaccel_device
    if raw_freeze_devices is not None:
        if isinstance(raw_freeze_devices, (list, tuple)):
            parsed_devices: List[str] = []
            for item in raw_freeze_devices:
                parsed_devices.extend(part.strip() for part in str(item).split(",") if part.strip())
            freeze_hwaccel_devices = parsed_devices or None
        else:
            freeze_hwaccel_devices = [part.strip() for part in str(raw_freeze_devices).split(",") if part.strip()] or None

    if args.only is None:
        run_playback = True
        run_face = True
        run_yunet = True
        run_freeze = True
    elif args.only == "playback":
        run_playback = True
        run_face = False
        run_yunet = False
        run_freeze = False
    elif args.only == "face":
        run_playback = False
        run_face = True
        run_yunet = False
        run_freeze = False
    elif args.only == "yunet":
        run_playback = False
        run_face = False
        run_yunet = True
        run_freeze = False
    elif args.only == "freeze":
        run_playback = False
        run_face = False
        run_yunet = False
        run_freeze = True
    elif args.only == "non-gemini":
        # Run only the non-Gemini paths: face coverage (YuNet/InsightFace) and freeze detection
        run_playback = False
        run_face = False
        run_yunet = True
        run_freeze = True
    else:  # args.only == "freeze-gpu"
        run_playback = False
        run_face = False
        run_yunet = False
        run_freeze = True
        if not freeze_hwaccel:
            freeze_hwaccel = "cuda"

    # Final override: if --no-gemini is set, always disable Gemini-based evaluations
    if getattr(args, "no_gemini", False):
        run_playback = False
        run_face = False

    new_rows_written, updated_rows = _evaluate_dataset(
        client=client,
        model_id=args.model,
        stimuli_map=stimuli_map,
        face_map=face_map,
        temperature=args.temperature,
        top_p=args.top_p,
        max_output_tokens=args.max_output_tokens,
        face_sample_step=args.face_sample_step,
        yunet_model_path=args.yunet_model_path,
        face_process_mode=face_process_mode,
        face_insight_sample_fps=face_insight_sample_fps,
        face_insight_det_size=face_insight_det_size,
        freeze_noise=args.freeze_noise,
        freeze_duration=args.freeze_duration,
        freeze_hwaccel=freeze_hwaccel,
        freeze_hwaccel_devices=freeze_hwaccel_devices,
        existing_ids=existing_ids,
        missing_playback_ids=missing_playback_ids,
        missing_face_ids=missing_face_ids,
        missing_yunet_ids=missing_yunet_ids,
        missing_freeze_ids=missing_freeze_ids,
        min_id=args.min_id,
        output_path=output_path,
        flush_num=args.flush_num,
        flush_seconds=args.flush_seconds,
        num_workers=args.num_workers,
        yunet_num_workers=args.yunet_num_workers,
        freeze_num_workers=args.freeze_num_workers,
        run_playback=run_playback,
        run_face=run_face,
        run_yunet=run_yunet,
        run_freeze=run_freeze,
    )

    logging.info(
        "Wrote %d new rows and updated %d existing rows in %s",
        new_rows_written,
        updated_rows,
        output_path,
    )


if __name__ == "__main__":
    main()
