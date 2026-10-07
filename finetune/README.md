# Supervised adaptation (LoRA)

Rank-16 LoRA adaptation of Qwen3.5-9B, reported in the paper's adaptation table (RQ4).
Every run is evaluated against the same local-harness base on the 250 IP videos and 112 SRM
items with `paired_test.py`.

| Run (`ckpt/<run>/train_args.json`) | Training data | Paper row | Paired results |
|---|---|---|---|
| `r2-order-5ep` | intensity mixture (joint + independent), seed 0 | Intensity mixture, seed 0 | `reactionbench/results/finetune_paired_r2.json`, `_r2_match.json` |
| `r2-seed1`, `r2-seed2` | same mixture, seeds 1 and 2 | Mixture, seeds 0–2 | `finetune_paired_r2_seed1.json`, `_r2_seed2.json` |
| `r5-indep-5ep` | independent-format IP only | Independent only | `finetune_paired_r5_indep.json` |
| `r5-joint-5ep` | joint-format IP only | Joint only | `finetune_paired_r5_joint.json` |
| `r3-match-3ep` | SRM only | Matching only | `finetune_paired_r3.json`, `_r3_match.json` |
| `r4-multi-5ep` | IP + SRM | Multi-task | `finetune_paired_r4.json`, `_r4_match.json` |
| `Qwen_Qwen3.5-9B-lora-r16` | first 3-epoch pilot | not reported | `finetune_paired.json` |

Adapter weights are not committed (only `train_args.json`).

```bash
python -m finetune.build_sft_data --split <dir from data_pipeline/07_export/build_finetune_split.py>
torchrun --nproc_per_node 8 -m finetune.train_lora --base Qwen/Qwen3.5-9B \
  --train finetune/data/train.jsonl --val finetune/data/val.jsonl --out finetune/ckpt/<run> --epochs 5 --rank 16
python -m finetune.paired_test --lora finetune/ckpt/<run>/best [--match]
```

`run_node.sh` runs all three steps on one GPU node.

**Split caveat.** Training and evaluation clips in these runs share source recordings, so the
results describe this configuration, not transfer to unseen recordings. `build_source_disjoint.py`
builds a source-recording-disjoint split for retraining.
