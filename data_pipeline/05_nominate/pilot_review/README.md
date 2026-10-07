# Phase 0 Blind Review

Scoring UI for the pre-screening pilot. Local and single-reviewer: the clips are already
on this machine, so unlike the annotation sites in `06_annotate/web/` this one needs no
Supabase and no R2 — one less thing that can fail mid-session.

## Run

```bash
python data_pipeline/05_nominate/pilot_review/serve.py --reviewer reviewer1
```

Then open <http://localhost:8770/>. Flags: `--port`, `--data_root`, `--pool`, `--out_dir`.

## The two passes

**1 · Score candidates** — 161 merged candidates in shuffled order, each played as its
own segment. Score 0–3 on the anchored scale with the number keys; `R` replays, `C`
replays with a 3 s lead-in (the count is recorded, so a candidate that always needed
context is visible later).

Score **every** candidate, including the 0s and 1s. A clip holding both a 3 and a 1 is
a same-video hard negative, and that pair only exists if the low score is written down.

**2 · Sweep for misses** — each clip end to end at adjustable speed, marking every
moment reaching ≥1 that the candidates missed. Number keys drop a mark at the playhead,
`Space` toggles play, `Enter` saves and advances. This pass is the recall denominator;
without it only precision is computable. Saving a clip with zero marks is a real answer.

## Outputs

Appended as they happen, so nothing is lost if the tab closes:

```
$RVB_WORK_DIR/review/scores_<reviewer>.jsonl
$RVB_WORK_DIR/review/sweep_<reviewer>.jsonl
```

Revising a score appends a new record rather than rewriting the old one; the loader
takes the last record per candidate, and the earlier judgement stays on disk.

## Blinding

`attribution.json` — which variant proposed which candidate — is never sent to the
browser. The displayed span is the mean of the merging variants' boundaries, so no
single method's framing defines the moment being judged. Order is shuffled with a fixed
seed by `05_nominate/build_review_pool.py`.
