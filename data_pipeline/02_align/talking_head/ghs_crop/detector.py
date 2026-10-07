from typing import Optional, Tuple

import numpy as np
import cv2

from insightface.app import FaceAnalysis


class FaceDetector:
	def __init__(
		self,
		conf_threshold: float = 0.7,
		input_size: int = 640,
		providers: Optional[list] = None,
	) -> None:
		self.conf_threshold = conf_threshold
		self.input_size = input_size
		self.providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
		self._app = FaceAnalysis(name="buffalo_l", providers=self.providers)
		self._app.prepare(ctx_id=0, det_size=(self.input_size, self.input_size))

	def detect_one(self, frame_bgr: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
		# InsightFace expects BGR ndarray
		faces = self._app.get(frame_bgr)
		if not faces:
			return None, 0.0
		# Select highest score face
		best = max(faces, key=lambda f: float(getattr(f, "det_score", 0.0)))
		score = float(getattr(best, "det_score", 0.0))
		if score < self.conf_threshold:
			return None, score
		x1, y1, x2, y2 = [int(v) for v in best.bbox]
		return (x1, y1, x2, y2), score


