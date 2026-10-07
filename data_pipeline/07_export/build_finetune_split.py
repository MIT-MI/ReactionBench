#!/usr/bin/env python3
"""Build the fine-tuning split for the intensity/ranking task: fine-tune ONE task (within-video
reaction intensity) on the Stage-A human labels that are NOT in any frozen eval set.

Labels: per moment, n raters / raw mean / std / median, plus a rater-DEBIASED intensity
(two-way additive model score_ij = mu + moment_i + rater_j, fit by alternating means; the
cross-annotator normalisation of add_debiased_consensus.py). Ranking ground truth for training = order of
debiased intensity within a video; both raw and debiased are kept.

Exclusions (video-level, zero leakage into ANY eval set): ranking set (250), match set
videos, rationale set videos. Val = every 10th train video by hashed id (deterministic),
stratified nothing else; creator is recorded for analysis.

Outputs $RVB_WORK_DIR/finetune/{train,val}.jsonl + stats.json.

    python data_pipeline/07_export/build_finetune_split.py --eval_dir reactionbench/tasks
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "common"))
import rvb_config as cfg  # noqa: E402  (loads .env, puts every stage dir on sys.path)

import argparse, hashlib, json, statistics, urllib.request
from collections import defaultdict
from pathlib import Path

REPO = cfg.PIPELINE_ROOT
OUT = cfg.work("finetune")
R2 = cfg.R2_PUBLIC  # Supabase URL/key and R2 base come from .env (see .env.example)


def sb_all(path):
    URL, KEY = cfg.supabase()
    out, step = [], 1000
    for frm in range(0, 2_000_000, step):
        req = urllib.request.Request(f"{URL}/rest/v1/{path}", headers={
            "apikey": KEY, "Authorization": f"Bearer {KEY}", "Range": f"{frm}-{frm+step-1}"})
        b = json.load(urllib.request.urlopen(req)); out += b
        if len(b) < step: return out
    return out


def debias(scores):  # scores: list of (moment_id, rater_id, score)
    m = defaultdict(list)
    for mid, _, s in scores: m[mid].append(s)
    moment = {k: statistics.mean(v) for k, v in m.items()}
    rater = defaultdict(float)
    for _ in range(30):
        acc = defaultdict(list)
        for mid, rid, s in scores: acc[rid].append(s - moment[mid])
        rater = {rid: statistics.mean(v) for rid, v in acc.items()}
        mean_r = statistics.mean(rater.values()); rater = {k: v - mean_r for k, v in rater.items()}
        acc = defaultdict(list)
        for mid, rid, s in scores: acc[mid].append(s - rater[rid])
        moment = {mid: statistics.mean(v) for mid, v in acc.items()}
    return {k: min(3.0, max(0.0, v)) for k, v in moment.items()}, rater


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval_dir", type=Path, default=cfg.PIPELINE_ROOT.parent / "reactionbench" / "tasks",
                   help="frozen eval task files (default: this repo's reactionbench/tasks)")
    p.add_argument("--min_raters", type=int, default=3)
    a = p.parse_args()

    videos = {v["video_id"]: v for v in sb_all("gold_videos?select=video_id,face_r2_key,duration_s,domain,creator")}
    cands = sb_all("gold_candidates?select=candidate_id,video_id,start_s,end_s")
    # Rater account 'test' is a regular rater (blind to model scores) and is kept.
    scores = [(r["candidate_id"], r["annotator_id"], int(r["score"]))
              for r in sb_all("gold_scores?select=candidate_id,annotator_id,score")]
    excl = set(i["video_id"] for i in json.loads((a.eval_dir / "ranking_set.json").read_text()))
    excl |= {t["video_id"] for t in json.loads((a.eval_dir / "match_set.json").read_text())["tasks"]}
    excl |= {t["video_id"] for t in json.loads((a.eval_dir / "rationale_set.json").read_text())["tasks"]}

    by_c = defaultdict(list)
    for cid, rid, s in scores: by_c[cid].append(s)
    deb, rater_bias = debias(scores)
    rows, skipped = defaultdict(list), {"eval_video": 0, "under_rated": 0}
    for c in cands:
        if c["video_id"] in excl: skipped["eval_video"] += 1; continue
        sc = by_c.get(c["candidate_id"], [])
        if len(sc) < a.min_raters: skipped["under_rated"] += 1; continue
        rows[c["video_id"]].append({
            "candidate_id": c["candidate_id"], "start_s": c["start_s"], "end_s": c["end_s"],
            "n_raters": len(sc), "mean": round(statistics.mean(sc), 4),
            "std": round(statistics.pstdev(sc), 4), "median": statistics.median(sc),
            "debiased": round(deb[c["candidate_id"]], 4), "scores": sorted(sc)})
    train, val = [], []
    for vid in sorted(rows):
        v = videos[vid]
        item = {"video_id": vid, "face_url": R2 + v["face_r2_key"], "face_r2_key": v["face_r2_key"],
                "duration_s": v["duration_s"], "domain": v["domain"], "creator": v["creator"],
                "moments": sorted(rows[vid], key=lambda m: m["start_s"])}
        (val if int(hashlib.sha1(vid.encode()).hexdigest(), 16) % 10 == 0 else train).append(item)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, data in (("train", train), ("val", val)):
        with open(OUT / f"{name}.jsonl", "w") as f:
            for it in data: f.write(json.dumps(it) + "\n")
    stats = {"train_videos": len(train), "val_videos": len(val),
             "train_moments": sum(len(i["moments"]) for i in train),
             "val_moments": sum(len(i["moments"]) for i in val),
             "excluded_eval_videos": len(excl), "skipped": skipped,
             "multi_moment_train_videos": sum(1 for i in train if len(i["moments"]) >= 2),
             "rater_bias": {k: round(v, 3) for k, v in rater_bias.items()},
             "debias_vs_raw_mean_abs_diff": round(statistics.mean(
                 abs(m["debiased"] - m["mean"]) for i in train + val for m in i["moments"]), 4)}
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
