import argparse
import json
import os
import subprocess
from typing import List, Optional, Tuple

import numpy as np

from .detector import FaceDetector
from .roi import (
	convert_face_to_head_shoulders_square,
	compute_global_roi,
	adjust_roi_for_head_ratio,
)
from .video import (
	iterate_frames,
	get_video_meta,
)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Global 1:1 head-and-shoulders crop for whole video",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	parser.add_argument("--input", required=True, help="Input video path")
	parser.add_argument("--output", required=True, help="Output video path")
	parser.add_argument("--step", type=int, default=3, help="Sample every N frames for ROI estimation")
	parser.add_argument("--conf-thres", type=float, default=0.7, help="Min face detection confidence")
	parser.add_argument("--det-size", type=int, default=640, help="Detector input size")
	parser.add_argument("--providers", type=str, default="CUDAExecutionProvider,CPUExecutionProvider", help="ONNX Runtime providers, comma-separated")
	parser.add_argument("--margin-x", type=float, default=0.25, help="Left/right margin multiplier of face width")
	parser.add_argument("--margin-top", type=float, default=0.35, help="Top margin multiplier of face height")
	parser.add_argument("--margin-bot", type=float, default=0.50, help="Bottom margin multiplier of face height")
	parser.add_argument("--mad-k", type=float, default=3.0, help="MAD threshold multiplier for outlier removal")
	parser.add_argument("--target-ratio", type=float, default=0.5, help="Target median face_area/roi_area ratio")
	parser.add_argument("--ratio-band", type=float, default=0.15, help="Allowed deviation around target ratio")
	parser.add_argument("--headroom-bias", type=float, default=0.0, help="Shift ROI upward by this fraction of side (e.g., 0.05)")
	parser.add_argument("--safety-scale", type=float, default=1.0, help="Multiply final ROI side by this factor for extra negative space (e.g., 1.05)")
	parser.add_argument("--face-area-percentile", type=float, default=50.0, help="Percentile of face area used for sizing (e.g., 60 for a slightly larger head)")
	parser.add_argument("--dry-run", action="store_true", help="Compute ROI only; do not encode")
	parser.add_argument("--print-debug", action="store_true", help="Print debug info for ROI steps")
	return parser.parse_args()


def run_ffmpeg_crop(input_path: str, output_path: str, x: int, y: int, side: int) -> None:
	filter_str = f"crop={side}:{side}:{x}:{y}"
	cmd = [
		"ffmpeg",
		"-y",
		"-i",
		input_path,
		"-filter:v",
		filter_str,
		"-c:a",
		"copy",
		output_path,
	]
	subprocess.run(cmd, check=True)


def main() -> None:
	args = parse_args()
	input_path = os.path.abspath(args.input)
	output_path = os.path.abspath(args.output)
	providers = [p.strip() for p in args.providers.split(",") if p.strip()]

	meta = get_video_meta(input_path)
	width, height, fps, frame_count = meta["width"], meta["height"], meta["fps"], meta["frame_count"]

	detector = FaceDetector(
		conf_threshold=args.conf_thres,
		input_size=args.det_size,
		providers=providers,
	)

	boxes_square: List[Tuple[int, int, int, int]] = []
	face_areas: List[float] = []

	for idx, (frame_bgr, frame_idx) in enumerate(
		iterate_frames(input_path, step=args.step)
	):
		box, score = detector.detect_one(frame_bgr)
		if box is None or score < args.conf_thres:
			continue
		# Expand to head-and-shoulders and force square
		sx1, sy1, sx2, sy2, face_area = convert_face_to_head_shoulders_square(
			box, width, height,
			margin_x=args.margin_x,
			margin_top=args.margin_top,
			margin_bot=args.margin_bot,
		)
		boxes_square.append((sx1, sy1, sx2, sy2))
		face_areas.append(face_area)

	if not boxes_square:
		raise RuntimeError("No valid face detections met the confidence threshold.")

	global_roi = compute_global_roi(
		boxes_square, width, height, mad_k=args.mad_k, headroom_bias=args.headroom_bias
	)

	# Sanity check: head size ratio
	global_roi = adjust_roi_for_head_ratio(
		global_roi,
		face_areas,
		boxes_square,
		target_ratio=args.target_ratio,
		ratio_band=args.ratio_band,
		frame_width=width,
		frame_height=height,
		face_area_percentile=args.face_area_percentile,
		safety_scale=args.safety_scale,
		headroom_bias=args.headroom_bias,
	)

	if args.print_debug:
		print(json.dumps({
			"meta": meta,
			"num_samples": len(boxes_square),
			"roi": {
				"x": int(global_roi[0]),
				"y": int(global_roi[1]),
				"side": int(global_roi[2]),
			}
		}, indent=2))

	if args.dry_run:
		return

	run_ffmpeg_crop(input_path, output_path, int(global_roi[0]), int(global_roi[1]), int(global_roi[2]))


if __name__ == "__main__":
	main()


