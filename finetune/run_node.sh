#!/usr/bin/env bash
# One-shot driver for one GPU node (8xA6000). Idempotent; rerun after any interruption.
#   bash finetune/run_node.sh /path/to/dataset_dir
# The dataset dir must hold the build_finetune_split.py output under reports/phase0/finetune/.
set -euo pipefail
cd "$(dirname "$0")/.."
DATASET_REPO="${1:?path to dataset repo (for reports/phase0/finetune)}"
BASE="${BASE:-Qwen/Qwen3.5-9B}"
OUT="finetune/ckpt/$(echo "$BASE" | tr '/' '_')-lora-r16"

[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q "torch" "transformers>=4.50" "peft" "accelerate" "pillow" "scipy" "numpy" "tinker" "tinker-cookbook" 2>&1 | tail -1
command -v ffmpeg >/dev/null || { echo "ffmpeg missing"; exit 1; }

echo "=== 1. SFT data (downloads ~1.5k face videos from R2, cuts frames) ==="
.venv/bin/python -m finetune.build_sft_data --split "$DATASET_REPO/reports/phase0/finetune"

echo "=== 2. LoRA training ==="
.venv/bin/torchrun --nproc_per_node "${NGPU:-8}" -m finetune.train_lora \
  --base "$BASE" --train finetune/data/train.jsonl --val finetune/data/val.jsonl --out "$OUT" \
  --epochs "${EPOCHS:-3}" --lr "${LR:-1e-4}" --rank 16

echo "=== 3. Eval: base vs LoRA through the SAME local path (IP joint + independent, 250 videos) ==="
for M in "hf:$BASE" "hf:$BASE+$OUT/best"; do
  .venv/bin/python -m reactionbench.run_ranking --model "$M"
  .venv/bin/python -m reactionbench.run_ranking_independent --model "$M"
done
.venv/bin/python -m reactionbench.summarize
echo "=== done; results in reactionbench/results, run config in $OUT/train_args.json ==="
