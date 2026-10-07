#!/usr/bin/env python3
"""Final Stage C selection: pick the rationale set and the match set.

Joins the coarse frame (pool.csv: reaction strength, match-readiness, aligned stimulus) with
the Gemini fine pass (gemini.jsonl: stimulus-event quality 0-3, self-contained flag), then
picks two human-annotation sets that are strong on BOTH axes:

  - rationale_set  ~200 moments, stim_quality>=2 & self_contained, domain round-robin + creator
                   cap. These go to Stage C (write/correct a "why did they react" rationale).
  - match_set      ~150 moments that are ALSO match-ready (unique stimulus alignment, >=4
                   distinct stimulus segments) so a same-video 4-option match item can be built.
                   Non-Patreon supply is thin here (see summary) -- balancing is best-effort.

Also writes review_sample.csv: a small stratified sheet (domain x quality) for a human to eyeball
the LLM's event reading before committing to the Prolific run. Selection only, never a label.

  python data_pipeline/06_annotate/select_stage_c.py --dry_run
  python data_pipeline/06_annotate/select_stage_c.py --rationale_n 200 --match_n 150
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "common"))
import rvb_config as cfg  # noqa: E402  (loads .env, puts every stage dir on sys.path)

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_pilot_frame import default_data_root  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = cfg.work("stimulus_ranking")
    p.add_argument("--pool_csv", type=Path, default=d / "pool.csv")
    p.add_argument("--gemini_jsonl", type=Path, default=d / "gemini.jsonl")
    p.add_argument("--out", type=Path, default=d)
    p.add_argument("--data_root", type=Path, default=default_data_root())
    p.add_argument("--rationale_n", type=int, default=200)
    p.add_argument("--match_n", type=int, default=180)
    p.add_argument("--exclusive", action="store_true",
                   help="forbid a moment from being in both sets. Default allows overlap — rationale "
                        "and match are different tasks, so a strong moment can serve both.")
    p.add_argument("--creator_cap", type=float, default=0.25, help="max fraction of a set from one creator")
    p.add_argument("--dur_tol", type=float, default=0.5,
                   help="drop a clip if |dur(face) - dur(stimulus)| exceeds this (timelines misaligned)")
    p.add_argument("--exclude", type=Path, default=cfg.work("stimulus_ranking/excluded_ids.txt"),
                   help="candidate_ids to drop (one per line, '#' comments) — human QC rejects, backfilled automatically")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def probe_dur(path: Path) -> float:
    try:
        o = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout.strip()
        return float(o) if o else -1.0
    except Exception:
        return -1.0


def combined_key(r: dict):
    """Higher is better: clear self-contained event first, then strong, tight reaction."""
    return (int(r["stim_quality"]), r["self_contained"] is True,
            float(r["reaction_mean"]), -float(r["reaction_std"]))


def balanced_pick(rows: List[dict], n: int, cap_frac: float) -> List[dict]:
    """Round-robin across domains (as even as supply allows), creator-capped, best-first."""
    cap = max(1, int(n * cap_frac))
    by_domain: Dict[str, List[dict]] = defaultdict(list)
    for r in sorted(rows, key=combined_key, reverse=True):
        by_domain[r["domain"]].append(r)
    picked: List[dict] = []
    per_creator: Counter = Counter()
    domains = list(by_domain)
    cursor = {d: 0 for d in domains}
    while len(picked) < n and any(cursor[d] < len(by_domain[d]) for d in domains):
        for d in domains:
            while cursor[d] < len(by_domain[d]):
                r = by_domain[d][cursor[d]]; cursor[d] += 1
                if per_creator[r["creator"]] < cap:
                    picked.append(r); per_creator[r["creator"]] += 1
                    break
            if len(picked) >= n:
                break
    return picked


def main() -> int:
    args = parse_args()
    pool = {r["candidate_id"]: r for r in csv.DictReader(open(args.pool_csv))}
    rows: List[dict] = []
    for line in args.gemini_jsonl.read_text().splitlines():
        g = json.loads(line)
        p = pool.get(g["candidate_id"])
        if not p or g.get("error"):
            continue
        rows.append({**p, "stim_quality": g["stim_quality"], "event": g["event"],
                     "self_contained": g["self_contained"]})

    # Duration-alignment gate: face_origin/{id} and stimuli/{id} must share a timeline for the
    # windowed, sync-played clips to line up. A gross duration mismatch means they do not, so drop
    # those clips (they can be re-cut from source if the loss is too high). Probe distinct videos once.
    # human QC rejects (e.g. a reaction that is carry-over from the previous clip, not a response
    # to this stimulus) are excluded here and backfilled by the balanced pick reaching its target.
    excluded = set()
    if args.exclude.exists():
        excluded = {l.split("#")[0].strip() for l in args.exclude.read_text().splitlines() if l.split("#")[0].strip()}
    if excluded:
        before = len(rows)
        rows = [r for r in rows if r["candidate_id"] not in excluded]
        print(f"[exclude] dropped {before - len(rows)} QC-rejected candidates")

    # recut rows already carry a dur_diff (aligned by construction); only probe the rest.
    face_root, stim_root = args.data_root / "face_origin", args.data_root / "stimuli"
    need = {r["video_id"] for r in rows if not str(r.get("dur_diff", "")).strip()}
    dur_cache: Dict[str, float] = {}
    for vid in need:
        df, ds = probe_dur(face_root / f"{vid}.mp4"), probe_dur(stim_root / f"{vid}.mp4")
        dur_cache[vid] = abs(df - ds) if df > 0 and ds > 0 else 999.0
    for r in rows:
        dd = str(r.get("dur_diff", "")).strip()
        r["dur_diff"] = float(dd) if dd else round(dur_cache[r["video_id"]], 2)
    aligned = [r for r in rows if r["dur_diff"] <= args.dur_tol]
    dropped = [r for r in rows if r["dur_diff"] > args.dur_tol]

    strong = [r for r in aligned if int(r["stim_quality"]) >= 2 and r["self_contained"]]
    match_pool = [r for r in strong if r["match_ready"] == "True"]

    rationale = balanced_pick(strong, args.rationale_n, args.creator_cap)
    rset = {r["candidate_id"] for r in rationale}
    # overlap is allowed by default: a strong moment can carry both a rationale and a match item
    # (different tasks). --exclusive removes rationale picks from the match pool.
    match_src = [r for r in match_pool if r["candidate_id"] not in rset] if args.exclusive else match_pool
    match = balanced_pick(match_src, args.match_n, args.creator_cap)

    # stratified review sheet: up to 4 per (domain x quality-bucket)
    review: List[dict] = []
    seen: Counter = Counter()
    for r in sorted(strong, key=combined_key, reverse=True):
        k = (r["domain"], r["stim_quality"])
        if seen[k] < 4:
            review.append(r); seen[k] += 1
    # guarantee some recut items so the realignment can be spot-checked
    have = {r["candidate_id"] for r in review}
    recut_ranked = sorted((r for r in strong if r.get("stim_source") == "recut"), key=combined_key, reverse=True)
    for r in recut_ranked:
        if sum(1 for x in review if x.get("stim_source") == "recut") >= 8:
            break
        if r["candidate_id"] not in have:
            review.append(r); have.add(r["candidate_id"])

    summary = {
        "joined": len(rows),
        "dur_dropped": len(dropped),
        "dur_dropped_by_domain": dict(Counter(r["domain"] for r in dropped)),
        "strong_self_contained": len(strong),
        "strong_by_domain": dict(Counter(r["domain"] for r in strong)),
        "match_pool": len(match_pool),
        "match_pool_by_domain": dict(Counter(r["domain"] for r in match_pool)),
        "rationale_selected": len(rationale),
        "rationale_by_domain": dict(Counter(r["domain"] for r in rationale)),
        "rationale_by_creator": dict(Counter(r["creator"] for r in rationale)),
        "match_selected": len(match),
        "match_by_domain": dict(Counter(r["domain"] for r in match)),
        "review_sample": len(review),
    }
    print(json.dumps(summary, indent=2))
    if args.dry_run:
        return 0

    cols = ["candidate_id", "video_id", "domain", "creator", "start_s", "end_s",
            "reaction_mean", "reaction_std", "stim_quality", "self_contained",
            "match_ready", "stim_s", "stim_e", "dur_diff", "stim_source", "event", "stim_text"]

    def write(name: str, data: List[dict]) -> None:
        with open(args.out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader(); w.writerows(data)

    write("rationale_set.csv", rationale)
    write("match_set.csv", match)
    write("review_sample.csv", review)
    (args.out / "select_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote rationale_set.csv ({len(rationale)}), match_set.csv ({len(match)}), "
          f"review_sample.csv ({len(review)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
