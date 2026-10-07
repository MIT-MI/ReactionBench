#!/usr/bin/env python3
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "common"))
import rvb_config as cfg  # noqa: E402  (loads .env, puts every stage dir on sys.path)

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    # Hint PyTorch / CUDA to be less fragmented if downstream libs import torch
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
except Exception:
    pass

# --- Project paths -----------------------------------------------------------------
TALKING_HEAD_ROOT = Path(__file__).resolve().parent / "talking_head"
if str(TALKING_HEAD_ROOT) not in sys.path:
    sys.path.insert(0, str(TALKING_HEAD_ROOT))

from ghs_crop.roi import (  # type: ignore  # noqa: E402
    adjust_roi_for_head_ratio,
    compute_global_roi,
    convert_face_to_head_shoulders_square,
)
from google import genai  # noqa: E402
from google.genai.types import GenerateContentConfig, Part  # noqa: E402


# --- Dataclasses -------------------------------------------------------------------
@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @staticmethod
    def from_pixels(x1: int, y1: int, x2: int, y2: int) -> "BoundingBox":
        return BoundingBox(int(x1), int(y1), int(x2), int(y2))

    def as_crop_expr(self) -> str:
        width = max(1, self.xmax - self.xmin)
        height = max(1, self.ymax - self.ymin)
        return f"crop={width}:{height}:{self.xmin}:{self.ymin}"

    def clamp(self, frame_w: int, frame_h: int) -> "BoundingBox":
        x1 = max(0, min(self.xmin, frame_w - 1))
        y1 = max(0, min(self.ymin, frame_h - 1))
        x2 = max(x1 + 1, min(self.xmax, frame_w))
        y2 = max(y1 + 1, min(self.ymax, frame_h))
        return BoundingBox(x1, y1, x2, y2)


@dataclass
class FaceCluster:
    boxes_square: List[Tuple[int, int, int, int]]
    face_areas: List[float]
    centers: List[Tuple[float, float]]
    total_score: float

    @property
    def samples(self) -> int:
        return len(self.boxes_square)

    @property
    def mean_center(self) -> Tuple[float, float]:
        if not self.centers:
            return (0.0, 0.0)
        xs, ys = zip(*self.centers)
        return (float(sum(xs)) / len(xs), float(sum(ys)) / len(ys))

    def add_sample(
        self,
        box_square: Tuple[int, int, int, int],
        face_area: float,
        center: Tuple[float, float],
        score: float,
    ) -> None:
        self.boxes_square.append(tuple(int(v) for v in box_square))
        self.face_areas.append(float(face_area))
        self.centers.append((float(center[0]), float(center[1])))
        self.total_score += float(score)


@dataclass
class FaceCropSpec:
    x: int
    y: int
    side: int

    def as_crop_expr(self) -> str:
        return f"crop={self.side}:{self.side}:{self.x}:{self.y}"


# --- Helpers -----------------------------------------------------------------------
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_bool(value: object) -> bool:
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"", "0", "false", "no", "n"}:
            return False
        if s in {"1", "true", "yes", "y"}:
            return True
        try:
            return bool(int(s))
        except Exception:
            return bool(s)
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    return bool(value)


def safe_float(value: object, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    if isinstance(value, float):
        if math.isnan(value):
            return default
        return value
    try:
        s = str(value).strip()
        if s == "" or s.lower() in {"nan", "none"}:
            return default
        return float(s)
    except Exception:
        return default


def run_ffmpeg_crop(
    input_path: Path,
    output_path: Path,
    crop_expr: str,
    *,
    start_time: Optional[float] = None,
    duration_limit: Optional[float] = None,
) -> None:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
    ]
    if start_time is not None and start_time > 0:
        cmd.extend(["-ss", f"{start_time:.3f}"])
    cmd.extend(["-i", str(input_path)])
    if duration_limit and duration_limit > 0:
        cmd.extend(["-t", f"{duration_limit:.3f}"])
    cmd.extend(
        [
            "-filter:v",
            crop_expr,
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-c:a",
            "copy",
            str(output_path),
        ]
    )
    subprocess.run(cmd, check=True)


def _ffprobe_stream_size(video_path: Path) -> Tuple[int, int]:
    """Return (width, height) via ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=s=x:p=0",
        str(video_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    out = proc.stdout.strip()
    if not out or "x" not in out:
        raise RuntimeError(f"Failed to probe dimensions for {video_path}")
    w_str, h_str = out.split("x", 1)
    return int(w_str), int(h_str)


def _probe_video_meta(video_path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,duration,nb_frames",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError(f"ffprobe returned no video streams for {video_path}")
    stream = streams[0]
    fmt = data.get("format") or {}

    width = int(float(stream.get("width") or 0))
    height = int(float(stream.get("height") or 0))
    fps_str = str(stream.get("avg_frame_rate") or "0/1")
    fps = 0.0
    if "/" in fps_str:
        num, den = fps_str.split("/", 1)
        try:
            num_f = float(num)
            den_f = float(den)
            if den_f != 0:
                fps = num_f / den_f
        except Exception:
            fps = 0.0
    else:
        try:
            fps = float(fps_str)
        except Exception:
            fps = 0.0

    nb_frames = stream.get("nb_frames")
    frame_count = int(float(nb_frames)) if nb_frames not in (None, "", "N/A") else 0
    stream_dur = stream.get("duration")
    stream_duration = float(stream_dur) if stream_dur not in (None, "", "N/A") else 0.0
    format_dur = fmt.get("duration")
    format_duration = float(format_dur) if format_dur not in (None, "", "N/A") else 0.0
    duration = stream_duration or format_duration

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration": duration,
    }


def _part_from_local_file(path: Path, mime_fallback: str = "image/jpeg") -> Part:
    data = path.read_bytes()
    return Part.from_bytes(data=data, mime_type=mime_fallback)


def _call_gemini_bbox(
    image_path: Path,
    *,
    project: Optional[str],
    location: str,
    model_id: str,
) -> List[int]:
    client = genai.Client(vertexai=True, project=project, location=location)
    prompt = (
        "Task: Given a single frame from a REACTION VIDEO, find the rectangle that contains the SOURCE media (the movie/clip being watched).\n"
        "Context:\n"
        "- A reaction video typically shows one or more reactors' webcam feeds and the source media in a separate rectangular box.\n"
        "- The source box can be positioned in a corner or along one edge; it can be resized but remains a straight-edged rectangle.\n"
        "- Ignore reactor faces/webcams, chat overlays, logos, subtitles, progress bars, or decorative frames; select the minimal exact box that shows the external movie/clip/game/etc.\n"
        "Output format:\n"
        "- Return ONLY a JSON array of four integers [ymin, xmin, ymax, xmax] in a 0-1000 coordinate space (origin at top-left).\n"
        "- The values must be integers (no floats/strings). Do NOT return any words, explanations, or Markdown.\n"
        "- If multiple candidates exist, choose the single box that best covers the full source media region (not the reactors)."
    )
    parts: List[Part] = [Part.from_text(text=prompt), _part_from_local_file(image_path)]
    config = GenerateContentConfig(temperature=0.0, top_p=1.0, max_output_tokens=256)

    response = client.models.generate_content(
        model=model_id,
        contents=parts,
        config=config,
    )
    text = getattr(response, "text", "") or ""
    match = re.search(r"\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]", text)
    if not match:
        raise ValueError(f"Gemini response missing bbox JSON: {text}")
    box = json.loads(match.group(0))
    if not (isinstance(box, list) and len(box) == 4):
        raise ValueError(f"Unexpected bbox format: {box}")
    return [int(box[0]), int(box[1]), int(box[2]), int(box[3])]


def _extract_json_array_from_text(text: str) -> List:
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except Exception:
        pass

    start = None
    depth = 0
    for idx, ch in enumerate(text):
        if ch == "[":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "]":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    snippet = text[start : idx + 1]
                    try:
                        data = json.loads(snippet)
                        if isinstance(data, list):
                            return data
                    except Exception:
                        pass
    raise ValueError(f"Could not parse JSON array from Gemini response: {text}")


def _call_gemini_heads(
    image_path: Path,
    *,
    project: Optional[str],
    location: str,
    model_id: str,
) -> List[List[int]]:
    client = genai.Client(vertexai=True, project=project, location=location)
    prompt = (
        "Identify every visible streamer/reactor head in this masked reaction frame. "
        "Return a JSON array where each item has the form {\"bbox\": [ymin, xmin, ymax, xmax]} "
        "using integers scaled to a 0-1000 coordinate space. "
        "Do not include any prose or Markdown, only the JSON array."
    )
    parts: List[Part] = [Part.from_text(text=prompt), _part_from_local_file(image_path)]
    config = GenerateContentConfig(temperature=0.0, top_p=1.0, max_output_tokens=512)

    response = client.models.generate_content(
        model=model_id,
        contents=parts,
        config=config,
    )
    text = getattr(response, "text", "") or ""
    try:
        raw_list = _extract_json_array_from_text(text)
    except Exception:
        # Attempt a simpler fallback prompt once
        fallback_prompt = (
            "Return ONLY a JSON array of objects with key \"bbox\" -> [ymin, xmin, ymax, xmax] "
            "for every face/head region visible. Coordinates must be 0-1000 integers."
        )
        response = client.models.generate_content(
            model=model_id,
            contents=[Part.from_text(text=fallback_prompt), _part_from_local_file(image_path)],
            config=config,
        )
        text = getattr(response, "text", "") or ""
        raw_list = _extract_json_array_from_text(text)

    boxes: List[List[int]] = []
    for item in raw_list:
        raw_bbox = None
        if isinstance(item, dict):
            raw_bbox = item.get("bbox") or item.get("box") or item.get("bounds")
        elif isinstance(item, list):
            raw_bbox = item
        if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
            continue
        try:
            boxes.append([int(round(float(v))) for v in raw_bbox])
        except Exception:
            continue
    return boxes


def _load_annotated_bbox(
    full_video: Path,
    frame_w: int,
    frame_h: int,
    candidate_dirs: Sequence[Path],
) -> Optional[BoundingBox]:
    filenames = [
        f"{full_video.stem}.stimuli_bbox.json",
        f"{full_video.name}.stimuli_bbox.json",
    ]
    candidates: List[Path] = []
    seen: set[Path] = set()

    for directory in candidate_dirs:
        try:
            dir_path = directory.resolve()
        except Exception:
            dir_path = directory
        for fname in filenames:
            path = dir_path / fname
            if path not in seen:
                candidates.append(path)
                seen.add(path)

    # also check next to video
    local_path = full_video.with_name(f"{full_video.stem}.stimuli_bbox.json")
    if local_path not in seen:
        candidates.append(local_path)
        seen.add(local_path)

    for path in candidates:
        try:
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        box_pixels = data.get("box_pixels")
        if isinstance(box_pixels, (list, tuple)) and len(box_pixels) == 4:
            x1, y1, x2, y2 = [int(round(float(v))) for v in box_pixels]
            bbox = BoundingBox.from_pixels(x1, y1, x2, y2).clamp(frame_w, frame_h)
            return bbox

        box_norm = data.get("box_norm_1000")
        if isinstance(box_norm, (list, tuple)) and len(box_norm) == 4 and frame_w > 0 and frame_h > 0:
            ymin, xmin, ymax, xmax = [int(round(float(v))) for v in box_norm]
            x1 = int(round(xmin / 1000.0 * frame_w))
            y1 = int(round(ymin / 1000.0 * frame_h))
            x2 = int(round(xmax / 1000.0 * frame_w))
            y2 = int(round(ymax / 1000.0 * frame_h))
            bbox = BoundingBox.from_pixels(x1, y1, x2, y2).clamp(frame_w, frame_h)
            return bbox

    return None


def _normalized_to_pixel_box(
    box_norm: Sequence[int],
    frame_w: int,
    frame_h: int,
) -> BoundingBox:
    ymin, xmin, ymax, xmax = [int(v) for v in box_norm]
    x1 = int(round(xmin / 1000.0 * frame_w))
    y1 = int(round(ymin / 1000.0 * frame_h))
    x2 = int(round(xmax / 1000.0 * frame_w))
    y2 = int(round(ymax / 1000.0 * frame_h))
    return BoundingBox(x1, y1, x2, y2).clamp(frame_w, frame_h)


def _extract_frame(
    video_path: Path,
    timestamp: float,
    dst_path: Path,
) -> None:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{timestamp:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(dst_path),
    ]
    subprocess.run(cmd, check=True)


def sample_gemini_bounding_box(
    video_path: Path,
    *,
    samples: int,
    project: Optional[str],
    location: str,
    model_id: str,
) -> BoundingBox:
    width, height = _ffprobe_stream_size(video_path)
    meta = _probe_video_meta(video_path)
    fps = max(float(meta.get("fps") or 0.0), 0.0)
    frame_count = int(meta.get("frame_count") or 0)
    duration = float(meta.get("duration") or 0.0)
    if duration <= 0 and fps > 0 and frame_count > 0:
        duration = frame_count / fps

    picks: List[BoundingBox] = []
    with tempfile.TemporaryDirectory(prefix="patreon_bbox_") as tmp_dir:
        for idx in range(samples):
            if duration > 0:
                frac = (idx + 1) / (samples + 1)
                ts = max(0.0, min(duration * frac, max(0.0, duration - 0.001)))
            else:
                ts = float(idx)
            img_path = Path(tmp_dir) / f"frame_{idx:02d}.jpg"
            _extract_frame(video_path, ts, img_path)
            bbox_norm = _call_gemini_bbox(img_path, project=project, location=location, model_id=model_id)
            picks.append(_normalized_to_pixel_box(bbox_norm, width, height))

    if not picks:
        return BoundingBox(0, 0, width, height)

    coords = np.array([[b.xmin, b.ymin, b.xmax, b.ymax] for b in picks], dtype=np.float64)
    medians = np.median(coords, axis=0)
    bbox = BoundingBox(int(round(medians[0])), int(round(medians[1])), int(round(medians[2])), int(round(medians[3])))
    return bbox.clamp(width, height)


def _cluster_faces(
    frame_w: int,
    frame_h: int,
    detections: Iterable[Tuple[Tuple[int, int, int, int], float]],
    *,
    margin_x: float,
    margin_top: float,
    margin_bot: float,
    headroom_bias: float,
    mad_k: float,
    target_ratio: float,
    ratio_band: float,
    face_area_percentile: float,
    safety_scale: float,
) -> List[Tuple[FaceCropSpec, FaceCluster]]:
    clusters: List[FaceCluster] = []
    diag = math.hypot(frame_w, frame_h)
    threshold = max(40.0, diag * 0.07)

    for face_box, score in detections:
        sx1, sy1, sx2, sy2, face_area = convert_face_to_head_shoulders_square(
            face_box,
            frame_w,
            frame_h,
            margin_x=margin_x,
            margin_top=margin_top,
            margin_bot=margin_bot,
        )
        center = ((sx1 + sx2) / 2.0, (sy1 + sy2) / 2.0)

        assigned = False
        for cluster in clusters:
            cx, cy = cluster.mean_center
            dist = math.hypot(center[0] - cx, center[1] - cy)
            if dist <= threshold:
                cluster.add_sample((sx1, sy1, sx2, sy2), face_area, center, score)
                assigned = True
                break
        if not assigned:
            new_cluster = FaceCluster(
                boxes_square=[(sx1, sy1, sx2, sy2)],
                face_areas=[face_area],
                centers=[center],
                total_score=float(score),
            )
            clusters.append(new_cluster)

    crops: List[Tuple[FaceCropSpec, FaceCluster]] = []
    for cluster in clusters:
        roi = compute_global_roi(
            cluster.boxes_square,
            frame_w,
            frame_h,
            mad_k=mad_k,
            headroom_bias=headroom_bias,
        )
        roi = adjust_roi_for_head_ratio(
            roi,
            cluster.face_areas,
            cluster.boxes_square,
            target_ratio=target_ratio,
            ratio_band=ratio_band,
            frame_width=frame_w,
            frame_height=frame_h,
            face_area_percentile=face_area_percentile,
            safety_scale=safety_scale,
            headroom_bias=headroom_bias,
        )
        spec = FaceCropSpec(x=int(roi[0]), y=int(roi[1]), side=int(roi[2]))
        # Bias horizontal framing toward the image center adaptively to avoid clipping the outer cheek.
        cx, _cy = cluster.mean_center
        spec = _bias_crop_toward_center(spec, cx, frame_w, strength=None)
        crops.append((spec, cluster))

    return crops


def detect_face_rois(
    masked_video_path: Path,
    *,
    max_heads: Optional[int],
    sample_count: int,
    gemini_project: Optional[str],
    gemini_location: str,
    gemini_model: str,
    margin_x: float,
    margin_top: float,
    margin_bot: float,
    headroom_bias: float,
    mad_k: float,
    target_ratio: float,
    ratio_band: float,
    face_area_percentile: float,
    safety_scale: float,
    clip_start: float,
    clip_end: Optional[float],
) -> List[FaceCropSpec]:
    meta = _probe_video_meta(masked_video_path)
    frame_w = int(meta.get("width") or 0) or 1
    frame_h = int(meta.get("height") or 0) or 1
    duration = float(meta.get("duration") or 0.0)

    sample_count = max(1, sample_count)

    clip_start_sec = max(0.0, clip_start or 0.0)
    if clip_end is not None and clip_end > clip_start_sec:
        clip_end_sec = clip_end
    else:
        clip_end_sec = duration if duration > 0 else clip_start_sec
    if duration > 0:
        clip_end_sec = min(clip_end_sec, duration)
    window = max(0.0, clip_end_sec - clip_start_sec)

    if window > 1e-3:
        timestamps = []
        for idx in range(sample_count):
            rel = (idx + 1) / (sample_count + 1)
            ts = clip_start_sec + rel * window
            ts = min(clip_end_sec - 1e-3, max(clip_start_sec, ts))
            timestamps.append(ts)
        if not timestamps:
            timestamps = [clip_start_sec]
    elif duration > 0:
        timestamps = [
            min(duration * (idx + 1) / (sample_count + 1), max(0.0, duration - 0.001))
            for idx in range(sample_count)
        ]
    else:
        timestamps = [float(idx) for idx in range(sample_count)]

    detections: List[Tuple[Tuple[int, int, int, int], float]] = []
    with tempfile.TemporaryDirectory(prefix="patreon_faces_") as tmp_dir:
        for idx, ts in enumerate(timestamps):
            img_path = Path(tmp_dir) / f"frame_{idx:02d}.jpg"
            try:
                _extract_frame(masked_video_path, ts, img_path)
            except Exception:
                continue
            if not img_path.exists():
                continue
            try:
                boxes_norm = _call_gemini_heads(
                    img_path,
                    project=gemini_project,
                    location=gemini_location,
                    model_id=gemini_model,
                )
            except Exception as exc:
                print(f"  [WARN] Gemini head detection failed at ts={ts:.2f}s: {exc}")
                continue
            if not boxes_norm:
                continue
            for box_norm in boxes_norm:
                try:
                    bbox = _normalized_to_pixel_box(box_norm, frame_w, frame_h)
                except Exception:
                    continue
                face_box = (bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax)
                detections.append((face_box, 1.0))

    if not detections:
        return []

    crop_pairs = _cluster_faces(
        frame_w,
        frame_h,
        detections,
        margin_x=margin_x,
        margin_top=margin_top,
        margin_bot=margin_bot,
        headroom_bias=headroom_bias,
        mad_k=mad_k,
        target_ratio=target_ratio,
        ratio_band=ratio_band,
        face_area_percentile=face_area_percentile,
        safety_scale=safety_scale,
    )

    if not crop_pairs:
        return []

    # Deduplicate overlapping crops (large overlap indicates same cluster)
    filtered_pairs: List[Tuple[FaceCropSpec, FaceCluster]] = []
    for spec, cluster in crop_pairs:
        keep = True
        for other_spec, _ in filtered_pairs:
            iou = _square_iou(spec, other_spec)
            if iou >= 0.7:
                keep = False
                break
        if keep:
            filtered_pairs.append((spec, cluster))

    if not filtered_pairs:
        return []

    filtered_pairs.sort(key=lambda pair: (-pair[1].samples, -pair[1].total_score, -pair[0].side))

    selected_specs: List[FaceCropSpec] = []
    limit = max_heads if (max_heads is not None and max_heads > 0) else len(filtered_pairs)
    for spec, _cluster in filtered_pairs[:limit]:
        # Re-check duplicates after sort, in case multiple clusters are similar
        keep = True
        for other in selected_specs:
            iou = _square_iou(spec, other)
            if iou >= 0.7:
                keep = False
                break
        if keep:
            selected_specs.append(spec)

    return selected_specs


def _square_iou(a: FaceCropSpec, b: FaceCropSpec) -> float:
    xa1, ya1, xa2, ya2 = a.x, a.y, a.x + a.side, a.y + a.side
    xb1, yb1, xb2, yb2 = b.x, b.y, b.x + b.side, b.y + b.side
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = a.side * a.side
    area_b = b.side * b.side
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return float(inter_area) / float(union)

# --- Adaptive center bias helpers --------------------------------------------------
def _adaptive_center_bias_strength(face_center_x: float, frame_w: int) -> float:
    """
    Return a bias strength in [0.06, 0.24] proportional to how far the face center is
    from the horizontal midpoint of the frame. Near edges => stronger bias (larger shift).
    """
    if frame_w <= 0:
        return 0.12
    d = abs((face_center_x - (frame_w / 2.0)) / max(1.0, (frame_w / 2.0)))  # 0 at center, ~1 at edges
    return 0.06 + 0.18 * max(0.0, min(1.0, d))


# Bias crop toward center with adaptive strength
def _bias_crop_toward_center(
    spec: FaceCropSpec, face_center_x: float, frame_w: int, *, strength: float | None = None
) -> FaceCropSpec:
    """
    Shift the crop horizontally toward the frame center so edge heads get extra room on the
    outer side. If `strength` is None, use an adaptive value based on how far the face is
    from the image center (0.06..0.24 fraction of the crop side).
    """
    use_strength = _adaptive_center_bias_strength(face_center_x, frame_w) if strength is None else max(0.0, min(1.0, strength))
    shift = int(round(spec.side * use_strength))
    half_w = frame_w / 2.0
    if face_center_x < half_w:
        # Person is on the left; nudge crop to the RIGHT.
        new_x = spec.x + shift
    else:
        # Person is on the right; nudge crop to the LEFT.
        new_x = spec.x - shift
    # Clamp within [0, frame_w - side]
    new_x = max(0, min(new_x, max(0, frame_w - spec.side)))
    if new_x == spec.x:
        return spec
    return FaceCropSpec(x=new_x, y=spec.y, side=spec.side)


def _compute_duration_seconds(meta: dict) -> float:
    duration = float(meta.get("duration") or 0.0)
    if duration > 0:
        return duration
    fps = float(meta.get("fps") or 0.0)
    frame_count = int(meta.get("frame_count") or 0)
    if fps > 0 and frame_count > 0:
        return frame_count / fps
    return 0.0


def _generate_segments(total_duration: float, max_time: Optional[float]) -> List[Tuple[float, Optional[float]]]:
    segments: List[Tuple[float, Optional[float]]] = []
    if max_time is None or max_time <= 0:
        if total_duration > 0:
            segments.append((0.0, total_duration))
        else:
            segments.append((0.0, None))
        return segments

    if total_duration <= 0:
        segments.append((0.0, max_time))
        return segments

    start = 0.0
    tol = 1e-3
    while start < total_duration - tol:
        remaining = max(0.0, total_duration - start)
        seg_dur = min(max_time, remaining)
        if seg_dur <= tol:
            break
        segments.append((start, seg_dur))
        start += seg_dur

    if not segments:
        segments.append((0.0, min(max_time, total_duration)))

    return segments


def generate_blackout_masked(full_video: Path, bbox: BoundingBox, dst_path: Path) -> None:
    """Create a temporary masked video by blacking out the stimuli bbox area.
    The masked video is identical to `full_video` except the rectangular region
    defined by `bbox` is filled with solid black. Audio is copied.
    """
    x = int(bbox.xmin)
    y = int(bbox.ymin)
    w = max(1, int(bbox.xmax - bbox.xmin))
    h = max(1, int(bbox.ymax - bbox.ymin))
    drawbox = f"drawbox=x={x}:y={y}:w={w}:h={h}:color=black@1:t=fill"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(full_video),
        "-filter:v",
        drawbox,
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-c:a",
        "copy",
        str(dst_path),
    ]
    subprocess.run(cmd, check=True)

# --- Main processing ----------------------------------------------------------------
def process_patreon_video(
    full_video: Path,
    masked_video: Path,
    *,
    output_stimuli_dir: Path,
    output_face_dir: Path,
    max_heads: Optional[int],
    max_time: Optional[float],
    gemini_samples: int,
    gemini_project: Optional[str],
    gemini_location: str,
    gemini_model: str,
    overwrite: bool,
    face_samples: int,
    face_margin_x: float,
    face_margin_top: float,
    face_margin_bot: float,
    face_headroom_bias: float,
    face_mad_k: float,
    face_target_ratio: float,
    face_ratio_band: float,
    face_percentile: float,
    face_safety_scale: float,
    stimuli_roi_dirs: Sequence[Path],
    clip_start: Optional[float] = None,
    clip_end: Optional[float] = None,
    mode: str = "all",
    skip_existing: bool = False,
) -> Tuple[List[Path], List[Path], int]:
    ensure_dir(output_stimuli_dir)
    ensure_dir(output_face_dir)

    full_meta = _probe_video_meta(full_video)
    duration = _compute_duration_seconds(full_meta)
    clip_start_sec = max(0.0, clip_start) if clip_start is not None else 0.0
    if clip_end is not None and clip_end > clip_start_sec:
        clip_end_sec = clip_end
    else:
        clip_end_sec = duration if duration > 0 else None
    if clip_end_sec is not None and duration > 0:
        clip_end_sec = min(clip_end_sec, duration)
    if clip_end_sec is None and duration > 0:
        clip_end_sec = duration
    window_duration = 0.0
    if clip_end_sec is not None:
        window_duration = max(0.0, clip_end_sec - clip_start_sec)
    effective_duration = window_duration if window_duration > 0 else duration
    segments = _generate_segments(effective_duration, max_time)
    multi_segment = len(segments) > 1

    def _nonempty(path: Path) -> bool:
        try:
            return path.exists() and path.stat().st_size > 0
        except Exception:
            return False

    stimuli_outputs: List[Path] = []
    stimuli_plan: List[Tuple[Path, float, Optional[float]]] = []
    for seg_idx, (rel_start, seg_duration) in enumerate(segments, start=1):
        if multi_segment:
            name = f"{full_video.stem}_stimuli_{seg_idx:03d}.mp4"
        else:
            name = f"{full_video.stem}_stimuli.mp4"
        path = output_stimuli_dir / name
        stimuli_outputs.append(path)
        stimuli_plan.append((path, rel_start, seg_duration))

    # Helpers to check existing outputs quickly
    def _all_exist(paths: List[Path]) -> bool:
        return all(p.exists() and p.stat().st_size > 0 for p in paths)

    def _existing_face_files(stem: str, out_dir: Path) -> List[Path]:
        return sorted(out_dir.glob(f"{stem}_face*.mp4"))

    # Determine if we need to crop stimuli depending on mode and skip_existing
    if mode == "face":
        need_stimuli_crop = False
    else:
        # mode == "all" or "stimuli"
        if skip_existing and _all_exist([p for p, _, _ in stimuli_plan]):
            # Everything already exists; skip any bbox/Gemini work for stimuli
            need_stimuli_crop = False
        else:
            need_stimuli_crop = overwrite or any(not _nonempty(path) for path, _, _ in stimuli_plan)

    bbox: Optional[BoundingBox] = None
    if need_stimuli_crop:
        frame_w = int(full_meta.get("width") or 0)
        frame_h = int(full_meta.get("height") or 0)
        bbox = _load_annotated_bbox(full_video, frame_w, frame_h, stimuli_roi_dirs)
        if bbox is None:
            if frame_w <= 0 or frame_h <= 0:
                frame_w, frame_h = _ffprobe_stream_size(full_video)
            if gemini_samples > 0:
                try:
                    bbox = sample_gemini_bounding_box(
                        full_video,
                        samples=gemini_samples,
                        project=gemini_project,
                        location=gemini_location,
                        model_id=gemini_model,
                    )
                except Exception as exc:
                    print(f"  [WARN] Gemini bbox failed for {full_video.name}: {exc}. Using full frame.")
                    bbox = BoundingBox(0, 0, frame_w, frame_h)
            else:
                bbox = BoundingBox(0, 0, frame_w, frame_h)

    if bbox is not None:
        crop_expr = bbox.as_crop_expr()
        for path, rel_start, seg_duration in stimuli_plan:
            if not overwrite and _nonempty(path):
                continue
            abs_start = clip_start_sec + rel_start
            run_ffmpeg_crop(
                full_video,
                path,
                crop_expr,
                start_time=abs_start,
                duration_limit=seg_duration,
            )

    if mode == "stimuli":
        # Stimuli-only mode: do not run face detection/cropping.
        return stimuli_outputs, [], 0

    if max_heads is not None and max_heads <= 0:
        return stimuli_outputs, [], 0

    # In face or all mode, honor --skip-existing by skipping face work if outputs are already present
    # Determine the stem and any existing face files under output_face_dir
    video_stem = full_video.stem
    existing_face_files = _existing_face_files(video_stem, output_face_dir)
    if skip_existing and existing_face_files and not overwrite:
        # Nothing to do for faces; return early
        return stimuli_outputs, [], 0

    crops = detect_face_rois(
        masked_video,
        max_heads=max_heads,
        sample_count=face_samples,
        gemini_project=gemini_project,
        gemini_location=gemini_location,
        gemini_model=gemini_model,
        margin_x=face_margin_x,
        margin_top=face_margin_top,
        margin_bot=face_margin_bot,
        headroom_bias=face_headroom_bias,
        mad_k=face_mad_k,
        target_ratio=face_target_ratio,
        ratio_band=face_ratio_band,
        face_area_percentile=face_percentile,
        safety_scale=face_safety_scale,
        clip_start=clip_start_sec,
        clip_end=clip_end_sec,
    )

    if not crops:
        return stimuli_outputs, [], 0

    face_outputs: List[Path] = []
    for head_idx, crop in enumerate(crops, start=1):
        crop_expr = crop.as_crop_expr()
        for seg_idx, (rel_start, seg_duration) in enumerate(segments, start=1):
            if multi_segment:
                face_name = f"{full_video.stem}_face{head_idx:02d}_seg{seg_idx:03d}.mp4"
            else:
                face_name = f"{full_video.stem}_face{head_idx:02d}.mp4"
            face_out = output_face_dir / face_name
            face_outputs.append(face_out)
            if not overwrite and _nonempty(face_out):
                continue
            abs_start = clip_start_sec + rel_start
            run_ffmpeg_crop(
                masked_video,
                face_out,
                crop_expr,
                start_time=abs_start,
                duration_limit=seg_duration,
            )

    return stimuli_outputs, face_outputs, len(crops)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract Patreon stimuli and talking-head crops.")
    parser.add_argument("root_dir", help="Root directory containing 'full' and 'masked' subfolders.")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional output root. Defaults to ROOT/stimuli and ROOT/face under the given root.",
    )
    parser.add_argument(
        "--max-heads",
        type=int,
        default=0,
        help="Upper bound on talking head crops to emit (0 means use all detected heads).",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional manifest CSV (id, video_link, label, num_reactors, selected, react_start, react_end). "
             "Only rows with selected=True are processed; num_reactors is updated after processing.",
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=0.0,
        help="Maximum duration (seconds) for output clips. 0 means full length.",
    )
    parser.add_argument(
        "--gemini-samples",
        type=int,
        default=5,
        help="Number of frames to sample for Gemini bounding box consensus.",
    )
    parser.add_argument("--gemini-project", default=os.environ.get("GOOGLE_CLOUD_PROJECT"))
    parser.add_argument("--gemini-location", default=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"))
    parser.add_argument("--gemini-model", default=os.environ.get("GOOGLE_CLOUD_MODEL_ID", "gemini-2.0-flash"))
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument(
        "--stimuli-roi-dir",
        action="append",
        default=None,
        help="Directory containing <video>.stimuli_bbox.json annotations (repeatable). "
             "Defaults to searching root/stimuli_bbox and alongside each video.",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "stimuli", "face"],
        default="all",
        help="Processing mode: 'all' (default) runs stimuli crop and face crops; 'stimuli' only crops stimuli; 'face' only crops faces from masked video."
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="If set, skip expensive processing for a video when the corresponding expected outputs already exist."
    )

    # Face detector parameters
    parser.add_argument(
        "--face-samples",
        type=int,
        default=10,
        help="Number of masked-frame samples to query Gemini for head detection (per video).",
    )
    parser.add_argument("--face-margin-x", type=float, default=0.25)
    parser.add_argument("--face-margin-top", type=float, default=0.35)
    parser.add_argument("--face-margin-bot", type=float, default=0.5)
    parser.add_argument("--face-headroom-bias", type=float, default=0.05)
    parser.add_argument("--face-mad-k", type=float, default=3.0)
    parser.add_argument("--face-target-ratio", type=float, default=0.5)
    parser.add_argument("--face-ratio-band", type=float, default=0.15)
    parser.add_argument("--face-percentile", type=float, default=60.0)
    parser.add_argument("--face-safety-scale", type=float, default=1.1)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    root_dir = Path(args.root_dir).resolve()
    if not root_dir.exists() or not root_dir.is_dir():
        raise FileNotFoundError(f"Root directory not found: {root_dir}")

    full_dir = root_dir / "full"
    masked_dir = root_dir / "masked"
    if not full_dir.is_dir():
        raise FileNotFoundError(f"Missing 'full' directory: {full_dir}")
    if not masked_dir.is_dir():
        raise FileNotFoundError(f"Missing 'masked' directory: {masked_dir}")

    if args.output_root:
        output_root = Path(args.output_root).resolve()
        stimuli_dir = output_root / "stimuli"
        face_dir = output_root / "face"
    else:
        stimuli_dir = root_dir / "stimuli"
        face_dir = root_dir / "face"

    ensure_dir(stimuli_dir)
    ensure_dir(face_dir)

    stimuli_roi_dirs: List[Path] = []
    if args.stimuli_roi_dir:
        for entry in args.stimuli_roi_dir:
            try:
                candidate = Path(entry).expanduser().resolve()
            except Exception:
                candidate = Path(entry)
            if candidate.is_dir():
                stimuli_roi_dirs.append(candidate)
            else:
                print(f"[WARN] Stimuli ROI directory not found: {candidate}")
    default_roi = root_dir / "stimuli_bbox"
    if default_roi.is_dir():
        stimuli_roi_dirs.append(default_roi)
    # Deduplicate while preserving order
    seen_dirs: set[Path] = set()
    unique_roi_dirs: List[Path] = []
    for d in stimuli_roi_dirs:
        if d not in seen_dirs:
            unique_roi_dirs.append(d)
            seen_dirs.add(d)
    stimuli_roi_dirs = unique_roi_dirs

    masked_lookup = {p.name: p for p in masked_dir.glob("*.mp4")}
    files = sorted(full_dir.glob("*.mp4"))

    if args.manifest:
        manifest_path = Path(args.manifest).expanduser()
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        try:
            df = pd.read_csv(manifest_path, dtype={"id": str})
        except Exception:
            # Fall back to default inference if pandas rejects dtype hints
            df = pd.read_csv(manifest_path)
            if "id" in df.columns:
                df["id"] = df["id"].apply(lambda value: "" if pd.isna(value) else str(value))
        if df.empty:
            print(f"No rows found in manifest: {manifest_path}")
            return

        updated = False
        for idx, row in df.iterrows():
            if not parse_bool(row.get("selected", False)):
                continue
            video_id = str(row.get("id", "")).strip()
            if not video_id:
                print(f"[WARN] Manifest row {idx}: missing id; skipping.")
                continue
            video_name = video_id if video_id.lower().endswith(".mp4") else f"{video_id}.mp4"
            full_video = full_dir / video_name
            masked_video = masked_dir / video_name
            full_exists = full_video.exists()
            masked_exists = masked_video.exists()
            masked_temp_dir = None
            if not full_exists:
                print(f"[WARN] Manifest row {idx}: missing full video for {video_name}; skipping.")
                df.at[idx, "num_reactors"] = 0
                updated = True
                continue
            if args.mode in ("all", "face") and not masked_exists:
                # Try to synthesize a temporary masked video by blacking out the stimuli bbox on the full video
                full_meta = _probe_video_meta(full_video)
                frame_w = int(full_meta.get("width") or 0) or 1
                frame_h = int(full_meta.get("height") or 0) or 1
                bbox = _load_annotated_bbox(full_video, frame_w, frame_h, stimuli_roi_dirs)
                if bbox is not None:
                    import tempfile as _tempfile
                    masked_temp_dir = _tempfile.TemporaryDirectory(prefix="auto_masked_")
                    masked_video = Path(masked_temp_dir.name) / video_name
                    try:
                        generate_blackout_masked(full_video, bbox, masked_video)
                        masked_exists = True
                        print(f"[INFO] Manifest row {idx}: synthesized masked video for {video_name} using stimuli bbox.")
                    except Exception as _exc:
                        print(f"[WARN] Manifest row {idx}: failed to synthesize masked video: {_exc}")
                        masked_video = masked_dir / video_name  # reset to original path
                else:
                    print(f"[WARN] Manifest row {idx}: missing masked video and no stimuli bbox for {video_name}; skipping.")
                    df.at[idx, "num_reactors"] = 0
                    updated = True
                    continue
            if args.mode == "stimuli" and not masked_exists:
                print(f"[INFO] Manifest row {idx}: masked video missing for {video_name}; continuing (stimuli mode).")
            clip_start = safe_float(row.get("react_start"), 0.0) or 0.0
            clip_end = safe_float(row.get("react_end"), None)

            print(f"Processing manifest row {idx}: {video_name} (react window {clip_start}-{clip_end})")
            stimuli_out_dir = stimuli_dir
            face_out_dir = face_dir / Path(video_name).stem
            ensure_dir(face_out_dir)

            try:
                stimuli_outs, face_outs, face_count = process_patreon_video(
                    full_video,
                    masked_video,
                    output_stimuli_dir=stimuli_out_dir,
                    output_face_dir=face_out_dir,
                    max_heads=args.max_heads if args.max_heads > 0 else None,
                    max_time=args.max_time if args.max_time > 0 else None,
                    gemini_samples=max(0, args.gemini_samples),
                    gemini_project=args.gemini_project,
                    gemini_location=args.gemini_location,
                    gemini_model=args.gemini_model,
                    overwrite=args.overwrite,
                    face_samples=max(1, args.face_samples),
                    face_margin_x=args.face_margin_x,
                    face_margin_top=args.face_margin_top,
                    face_margin_bot=args.face_margin_bot,
                    face_headroom_bias=args.face_headroom_bias,
                    face_mad_k=args.face_mad_k,
                    face_target_ratio=args.face_target_ratio,
                    face_ratio_band=args.face_ratio_band,
                    face_percentile=args.face_percentile,
                    face_safety_scale=args.face_safety_scale,
                    stimuli_roi_dirs=stimuli_roi_dirs,
                    clip_start=clip_start,
                    clip_end=clip_end,
                    mode=args.mode,
                    skip_existing=args.skip_existing,
                )
                if args.mode in ("all", "stimuli"):
                    stim_list = ", ".join(str(p.relative_to(root_dir)) for p in stimuli_outs) if stimuli_outs else "none"
                    print(f"  Stimuli -> {stim_list}")
                if args.mode in ("all", "face"):
                    face_list = ", ".join(str(p.relative_to(root_dir)) for p in face_outs) if face_outs else "none"
                    print(f"  Faces   -> {face_list} (reactors={face_count})")
                if args.mode != "stimuli":
                    df.at[idx, "num_reactors"] = int(face_count)
                    updated = True
            finally:
                try:
                    if masked_temp_dir is not None:
                        masked_temp_dir.cleanup()
                except Exception:
                    pass

        if updated:
            df.to_csv(manifest_path, index=False)
            print(f"Manifest updated: {manifest_path}")
        return

    if not files:
        print(f"No mp4 files found in {full_dir}")
        return

    for full_video in files:
        masked_video = masked_lookup.get(full_video.name)
        masked_temp_dir = None
        if masked_video is None:
            # Try to synthesize a temporary masked video by blacking out the stimuli bbox on the full video
            full_meta = _probe_video_meta(full_video)
            frame_w = int(full_meta.get("width") or 0) or 1
            frame_h = int(full_meta.get("height") or 0) or 1
            bbox = _load_annotated_bbox(full_video, frame_w, frame_h, stimuli_roi_dirs)
            if bbox is not None:
                import tempfile as _tempfile
                masked_temp_dir = _tempfile.TemporaryDirectory(prefix="auto_masked_")
                masked_video = Path(masked_temp_dir.name) / full_video.name
                try:
                    generate_blackout_masked(full_video, bbox, masked_video)
                    print(f"[INFO] Synthesized masked video for {full_video.name} using stimuli bbox.")
                except Exception as _exc:
                    print(f"[WARN] Failed to synthesize masked video for {full_video.name}: {_exc}; skipping.")
                    continue
            else:
                print(f"[WARN] No masked counterpart and no stimuli bbox for {full_video.name}, skipping.")
                continue
        print(f"Processing {full_video.name}")
        stimuli_out_dir = stimuli_dir
        face_out_dir = face_dir / full_video.stem
        ensure_dir(face_out_dir)

        try:
            stimuli_outs, face_outs, face_count = process_patreon_video(
                full_video,
                masked_video,
                output_stimuli_dir=stimuli_out_dir,
                output_face_dir=face_out_dir,
                max_heads=args.max_heads if args.max_heads > 0 else None,
                max_time=args.max_time if args.max_time > 0 else None,
                gemini_samples=max(0, args.gemini_samples),
                gemini_project=args.gemini_project,
                gemini_location=args.gemini_location,
                gemini_model=args.gemini_model,
                overwrite=args.overwrite,
                face_samples=max(1, args.face_samples),
                face_margin_x=args.face_margin_x,
                face_margin_top=args.face_margin_top,
                face_margin_bot=args.face_margin_bot,
                face_headroom_bias=args.face_headroom_bias,
                face_mad_k=args.face_mad_k,
                face_target_ratio=args.face_target_ratio,
                face_ratio_band=args.face_ratio_band,
                face_percentile=args.face_percentile,
                face_safety_scale=args.face_safety_scale,
                stimuli_roi_dirs=stimuli_roi_dirs,
                clip_start=None,
                clip_end=None,
                mode=args.mode,
                skip_existing=args.skip_existing,
            )
            if args.mode in ("all", "stimuli"):
                stim_list = ", ".join(str(p.relative_to(root_dir)) for p in stimuli_outs) if stimuli_outs else "none"
                print(f"  Stimuli -> {stim_list}")
            if args.mode in ("all", "face"):
                face_list = ", ".join(str(p.relative_to(root_dir)) for p in face_outs) if face_outs else "none"
                print(f"  Faces   -> {face_list} (reactors={face_count})")
        except Exception as exc:
            print(f"[ERROR] Failed to process {full_video.name}: {exc}")
        finally:
            try:
                if masked_temp_dir is not None:
                    masked_temp_dir.cleanup()
            except Exception:
                pass


if __name__ == "__main__":
    main()
