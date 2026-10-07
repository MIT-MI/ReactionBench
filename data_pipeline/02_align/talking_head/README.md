## Global 1:1 Head-and-Shoulders Crop for a Whole Video

One fixed, square ROI for an entire single-person video. Detect face per frame on GPU, expand to a head-and-shoulders box, remove outliers, take a robust consensus ROI, sanity-check head size, and crop the whole video with ffmpeg.

### Features
- **GPU face detector (InsightFace SCRFD)** via onnxruntime-gpu
- **Robust ROI estimation**: MAD outlier removal + percentile consensus
- **Square 1:1 framing** with configurable margins for headroom/shoulders
- **Head size sanity check** with automatic ROI scaling
- **Fast estimation** by sampling frames; crop applies to full stream
- **FFmpeg re-encode** with audio copy

### Requirements
- Python 3.9+
- ffmpeg installed and in PATH
- CUDA GPU + drivers (for GPU acceleration)

### Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you do not have CUDA, you can install CPU-only `onnxruntime` and it will fall back to CPU (slower):
```bash
pip uninstall -y onnxruntime-gpu || true
pip install onnxruntime
```

### Usage
```bash
source .venv/bin/activate
python -m ghs_crop \
  --input "/path/to/input.mp4" \
  --output "/path/to/output.mp4" \
  --step 3 \
  --conf-thres 0.7 \
  --margin-x 0.25 --margin-top 0.35 --margin-bot 0.50 \
  --mad-k 3.0 \
  --target-ratio 0.5 --ratio-band 0.15
```

Key arguments:
- `--input`: input video path
- `--output`: output video path
- `--step`: sample every N frames for ROI estimation (final crop applies to all frames)
- `--conf-thres`: min detection confidence to accept a frame
- `--margin-x`, `--margin-top`, `--margin-bot`: margins relative to face box
- `--mad-k`: MAD threshold multiplier for outlier removal
- `--target-ratio`, `--ratio-band`: desired median face-area/ROI-area ratio (e.g., 0.5 ± 0.15)
- `--det-size`: SCRFD detector input size (e.g., 640)
- `--providers`: override ONNX Runtime providers (comma-separated). Default tries CUDA then CPU.
- `--headroom-bias`: shift the final square up by a fraction of its side (e.g., 0.05) to add top headroom
- `--safety-scale`: scale the final square side (e.g., 1.05) to add more negative space
- `--face-area-percentile`: use this percentile of face area for sizing (robust vs. outliers; default 50)

### Notes
- The first run will download InsightFace models to `~/.insightface`.
- For best results, use upper-body single-person videos with minimal profile turns.
- You can enable `--dry-run` to compute and print the ROI without encoding.

### Example
```bash
python -m ghs_crop --input input.mp4 --output output_square.mp4 --step 3 --conf-thres 0.7 \
  --headroom-bias 0.05 --safety-scale 1.08 --face-area-percentile 60
```

### License
MIT


