#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-reactionbench-libreface:latest}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PIPELINE_ROOT="${PIPELINE_ROOT:-$(cd "$HERE/../../.." && pwd)}"   # data_pipeline/
DATA_ROOT="${RVB_DATA_ROOT:?set RVB_DATA_ROOT to the clip pool}"
LIBREFACE_ENV="${LIBREFACE_ENV:?set LIBREFACE_ENV to a host conda env with libreface installed}"

docker run --rm -it \
  --gpus all \
  --ipc=host --shm-size=16g \
  -v "${PIPELINE_ROOT}:/workspace" \
  -v "${DATA_ROOT}:/workspace/data" \
  -v "${LIBREFACE_ENV}:/opt/conda/envs/libreface:ro" \
  -e RVB_DATA_ROOT=/workspace/data \
  -w /workspace \
  "${IMAGE_NAME}" \
  "$@"
