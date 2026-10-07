from typing import Dict, Generator, Tuple

import cv2


def get_video_meta(path: str) -> Dict[str, float]:
	capture = cv2.VideoCapture(path)
	if not capture.isOpened():
		raise RuntimeError(f"Failed to open video: {path}")
	width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
	frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
	capture.release()
	return {
		"width": width,
		"height": height,
		"fps": fps,
		"frame_count": frame_count,
	}


def iterate_frames(path: str, step: int = 1) -> Generator[Tuple["cv2.Mat", int], None, None]:
	capture = cv2.VideoCapture(path)
	if not capture.isOpened():
		raise RuntimeError(f"Failed to open video: {path}")
	idx = 0
	while True:
		ok, frame = capture.read()
		if not ok:
			break
		if idx % step == 0:
			yield frame, idx
		idx += 1
	capture.release()


