from typing import Iterable, List, Sequence, Tuple

import numpy as np


def clamp(value: int, low: int, high: int) -> int:
	return max(low, min(int(value), high))


def convert_face_to_head_shoulders_square(
	face_box: Tuple[int, int, int, int],
	frame_w: int,
	frame_h: int,
	margin_x: float = 0.25,
	margin_top: float = 0.35,
	margin_bot: float = 0.50,
) -> Tuple[int, int, int, int, float]:
	fx1, fy1, fx2, fy2 = face_box
	w = max(1, fx2 - fx1)
	h = max(1, fy2 - fy1)

	# Expand with margins
	x1 = fx1 - margin_x * w
	x2 = fx2 + margin_x * w
	y1 = fy1 - margin_top * h
	y2 = fy2 + margin_bot * h

	# Center and make square
	cx = (x1 + x2) / 2.0
	cy = (y1 + y2) / 2.0
	w_exp = x2 - x1
	h_exp = y2 - y1
	side = max(w_exp, h_exp)
	x1s = int(round(cx - side / 2.0))
	y1s = int(round(cy - side / 2.0))
	x2s = x1s + int(round(side))
	y2s = y1s + int(round(side))

	# Clamp to bounds
	x1s = clamp(x1s, 0, frame_w - 1)
	y1s = clamp(y1s, 0, frame_h - 1)
	x2s = clamp(x2s, x1s + 1, frame_w)
	y2s = clamp(y2s, y1s + 1, frame_h)

	face_area = float(w * h)
	return x1s, y1s, x2s, y2s, face_area


def median_abs_deviation(values: np.ndarray) -> float:
	med = np.median(values)
	return float(np.median(np.abs(values - med)))


def filter_outliers_mad(boxes: Sequence[Tuple[int, int, int, int]], k: float = 3.0) -> List[Tuple[int, int, int, int]]:
	if not boxes:
		return []
	coords = np.array(boxes, dtype=np.float64)
	keep_mask = np.ones(len(coords), dtype=bool)
	for j in range(4):
		col = coords[:, j]
		med = np.median(col)
		mad = median_abs_deviation(col)
		mad_eff = mad if mad > 1e-6 else 1.0
		z = np.abs(col - med) / mad_eff
		keep_mask &= (z <= k)
	return [tuple(map(int, coords[i])) for i in range(len(coords)) if keep_mask[i]]



def compute_global_roi(
	boxes_square: Sequence[Tuple[int, int, int, int]],
	frame_w: int,
	frame_h: int,
	mad_k: float = 3.0,
	percentiles: Tuple[float, float] = (10.0, 90.0),
	headroom_bias: float = 0.0,
) -> Tuple[int, int, int]:
	filtered = filter_outliers_mad(boxes_square, k=mad_k)
	if not filtered:
		filtered = list(boxes_square)

	coords = np.array(filtered, dtype=np.float64)
	p_lo, p_hi = percentiles
	x1 = float(np.percentile(coords[:, 0], p_lo))
	y1 = float(np.percentile(coords[:, 1], p_lo))
	x2 = float(np.percentile(coords[:, 2], p_hi))
	y2 = float(np.percentile(coords[:, 3], p_hi))

	# Enforce square
	side = max(x2 - x1, y2 - y1)
	cx = (x1 + x2) / 2.0
	cy = (y1 + y2) / 2.0
	x = int(round(cx - side / 2.0))
	y = int(round(cy - side / 2.0))
	s = int(round(side))

	# Optional headroom bias: shift ROI upward by a fraction of side
	if headroom_bias != 0.0:
		y = int(round(y - headroom_bias * s))

	# Clamp to image
	x = clamp(x, 0, frame_w - 1)
	y = clamp(y, 0, frame_h - 1)
	s = max(1, min(s, frame_w - x, frame_h - y))
	# Re-square if clamped
	s = min(s, frame_w - x, frame_h - y)
	return x, y, s


def adjust_roi_for_head_ratio(
	roi: Tuple[int, int, int],
	face_areas: Sequence[float],
	boxes_square: Sequence[Tuple[int, int, int, int]],
	target_ratio: float,
	ratio_band: float,
	frame_width: int,
	frame_height: int,
	face_area_percentile: float = 50.0,
	safety_scale: float = 1.0,
	headroom_bias: float = 0.0,
) -> Tuple[int, int, int]:
	if not face_areas or not boxes_square:
		return roi

	# Estimate face area and implied ratio vs ROI area
	face_areas_arr = np.array(face_areas, dtype=np.float64)
	face_stat_area = float(np.percentile(face_areas_arr, face_area_percentile))
	x, y, s = roi
	roi_area = float(max(1, s) ** 2)
	median_ratio = face_stat_area / roi_area
	low = target_ratio - ratio_band
	high = target_ratio + ratio_band
	if low <= median_ratio <= high:
		# Apply safety scale/headroom even if ratio is in band
		if safety_scale != 1.0 or headroom_bias != 0.0:
			cx = x + s / 2
			cy = y + s / 2
			s_new = int(round(s * safety_scale))
			x_new = int(round(cx - s_new / 2))
			y_new = int(round(cy - s_new / 2 - headroom_bias * s_new))
			x_new = clamp(x_new, 0, frame_width - 1)
			y_new = clamp(y_new, 0, frame_height - 1)
			s_new = min(s_new, frame_width - x_new, frame_height - y_new)
			return x_new, y_new, int(s_new)
		return roi

	# Scale ROI side to move ratio toward target
	# ratio = face_area / s^2; desire ratio_target => s_new = sqrt(face_area / ratio_target)
	s_new = int(round((face_stat_area / max(target_ratio, 1e-6)) ** 0.5))
	# Constrain s_new so that ratio is within band bounds, but favor target
	s_min = int(round((face_stat_area / max(high, 1e-6)) ** 0.5))
	s_max = int(round((face_stat_area / max(low, 1e-6)) ** 0.5))
	s_new = max(s_min, min(s_new, s_max))
	# Apply safety scaling to increase negative space if requested
	if safety_scale != 1.0:
		s_new = int(round(s_new * safety_scale))

	# Keep center, adjust and clamp
	cx = x + s / 2
	cy = y + s / 2
	x_new = int(round(cx - s_new / 2))
	y_new = int(round(cy - s_new / 2 - headroom_bias * s_new))
	x_new = clamp(x_new, 0, frame_width - 1)
	y_new = clamp(y_new, 0, frame_height - 1)
	s_new = min(s_new, frame_width - x_new, frame_height - y_new)
	return x_new, y_new, int(s_new)


