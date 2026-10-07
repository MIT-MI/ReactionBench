# LibreFace Docker runtime

OS-level runtime (ffmpeg, OpenCV system libs) for the LibreFace feature scripts in
`04_features/`. Python packages come from a host conda env that has
[`libreface`](https://github.com/ihp-lab/LibreFace) installed, mounted read-only at
`/opt/conda/envs/libreface` inside the container.

## Build

```bash
docker build -t reactionbench-libreface:latest data_pipeline/04_features/docker/libreface
```

## Run

```bash
export RVB_DATA_ROOT=/path/to/clean_human_reaction_data
export LIBREFACE_ENV=/path/to/conda/envs/libreface
bash data_pipeline/04_features/docker/libreface/run_libreface.sh python --version

# AU segments (face_segments/) and per-frame AU intensity (face_intensity_series/, face_intensity_avg/)
bash data_pipeline/04_features/docker/libreface/run_libreface.sh \
  python 04_features/libreface_segment_expressions.py /workspace/data/face_origin /workspace/data/face_segments \
  --device cuda:0 --mode au --min-duration 0.5
bash data_pipeline/04_features/docker/libreface/run_libreface.sh \
  python 04_features/libreface_au_intensity.py --data_root /workspace/data
```

The helper mounts `data_pipeline/` at `/workspace`, `$RVB_DATA_ROOT` at `/workspace/data`, and
`$LIBREFACE_ENV` at `/opt/conda/envs/libreface`, with `--gpus all`. Set `IMAGE_NAME` to use a
different tag.
