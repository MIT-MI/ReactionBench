#!/usr/bin/env python3
"""Draw the Phase 0 pilot sample.

Protocol, as amended after measuring the pool (see the ``notes`` block in the
emitted sampling_record.json):

  * The frame is the basic-QC pool -- NEVER the LibreFace-selected
    ``avg_intense_metadata_top3000.csv``. LibreFace expressiveness enters only as a
    *stratification covariate*, so low-signal clips are guaranteed a seat.
  * Allocation is creator-primary, not domain-primary. Once the per-video creator keys
    are resolved to real channels (resolve_video_url_creators.py), the eligible frame
    holds 15,743 clips from only 22 creators, and creators are almost perfectly nested
    inside domains -- ``game`` is one channel, ``Patreon`` is two shows. Domain and
    creator are therefore very nearly the same axis, so the original "7 domains x 7 clips,
    <=1 per creator" cannot be satisfied in either variable. We instead round-robin over
    creators, which covers every domain automatically (each domain has >=1 creator) and
    bounds any single creator's share of the sample.
  * Strata within each creator: LibreFace ``au_peak`` tertile x duration bucket, with the
    tertile computed *within* the creator. Every relaxation of a stratum is recorded.

Outputs (atomic writes):
  $RVB_WORK_DIR/pilot_ids.txt        one clip id per line -- also the exclusion
                                      list for the 800-video gold pool
  $RVB_WORK_DIR/pilot_sample.csv     the drawn clips with their covariates
  $RVB_WORK_DIR/sampling_record.json full provenance, destined for the appendix
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "common"))
import rvb_config as cfg  # noqa: E402  (loads .env, puts every stage dir on sys.path)

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

DEFAULT_EXCLUDED_DOMAINS = ["Music_reactions"]
DURATION_BUCKETS = [(10.0, 30.0, "10-30s"), (30.0, 55.0, "30-55s"), (55.0, float("inf"), ">=55s")]
TERTILES = ["low", "mid", "high"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frame", type=Path, default=cfg.work("pilot_frame.csv"))
    p.add_argument("--out_dir", type=Path, default=cfg.WORK_DIR)
    p.add_argument("--seed", type=int, default=20260728)
    p.add_argument("--min_duration", type=float, default=10.0,
                   help="clips shorter than this leave no room for a top-2 nomination")
    p.add_argument("--n", type=int, default=50, help="clips to draw")
    p.add_argument("--allow_silent", action="store_true",
                   help="keep clips with no audio track. Off by default: the pilot asks whether a "
                        "multimodal model beats a face-only heuristic, which is unanswerable on a "
                        "clip with nothing to hear. Silent clips are 1.3%% of the pool but cluster "
                        "in the small domains that creator-balanced sampling over-weights.")
    p.add_argument("--min_libreface_coverage", type=float, default=0.8,
                   help="minimum fraction of the clip the LibreFace series spans, so V0 is not "
                        "structurally blind on a clip the LLM variants can see in full")
    p.add_argument("--creator_cap", type=int, default=3,
                   help="max clips per resolved creator; 22 eligible creators x 3 = 66 >= 50, and "
                        "the round-robin keeps the realised max at ceil(n / n_creators)")
    p.add_argument("--exclude_domains", nargs="*", default=DEFAULT_EXCLUDED_DOMAINS)
    p.add_argument("--exclude_ids", type=Path, default=None,
                   help="ids to drop from the frame; use the pilot ids when drawing a prompt-tuning "
                        "dev set, so prompt iteration never touches the evaluation clips")
    return p.parse_args()


def creator_kind(key: str) -> str:
    if "watch?v=" in key:
        return "watch_url"
    if "/channel/" in key or "/@" in key:
        return "channel_url"
    if key.startswith("@"):
        return "handle"
    return "named"


def duration_bucket(seconds: float) -> str:
    for low, high, label in DURATION_BUCKETS:
        if low <= seconds < high:
            return label
    return DURATION_BUCKETS[-1][2]


def assign_tertiles(values: pd.Series) -> pd.Series:
    """Tertile labels within a group; degrades gracefully when the group is tiny."""
    try:
        return pd.qcut(values.rank(method="first"), 3, labels=TERTILES).astype(str)
    except ValueError:
        return pd.Series([TERTILES[1]] * len(values), index=values.index)


def draw_from_creator(
    pool: pd.DataFrame,
    rng: np.random.Generator,
    tertile: str,
    domain_used: Counter,
    bucket_used: Counter,
) -> tuple[pd.Series, List[str]]:
    """Pick one clip from a creator's pool, preferring the requested tertile.

    Preference order within a creator: tertile coverage > least-used domain > least-used
    duration bucket. Domain matters only for the three creators who span more than one.
    """
    relaxed: List[str] = []
    candidates = pool[pool["tertile"] == tertile]
    if candidates.empty:
        relaxed.append(f"tertile={tertile}")
        candidates = pool

    domain = min(sorted(candidates["domain"].unique()), key=lambda d: (domain_used[d], d))
    candidates = candidates[candidates["domain"] == domain]
    bucket = min(sorted(candidates["bucket"].unique()), key=lambda b: (bucket_used[b], b))
    candidates = candidates[candidates["bucket"] == bucket]
    return candidates.iloc[int(rng.integers(len(candidates)))], relaxed


def allocate(
    eligible: pd.DataFrame,
    n: int,
    cap: int,
    rng: np.random.Generator,
) -> tuple[List[str], List[dict]]:
    """Round-robin over creators: everyone gets a first clip before anyone gets a second.

    This is the whole anti-bias mechanism now that domain and creator are nearly the same
    axis. It also guarantees domain coverage for free, since every domain has at least one
    creator and every creator is served in the first pass.
    """
    picked: List[str] = []
    relaxations: List[dict] = []
    creator_used: Counter = Counter()
    domain_used: Counter = Counter()
    bucket_used: Counter = Counter()
    remaining = eligible.copy()

    # Largest pools first within a pass, so ties break towards creators who can absorb more.
    creators = sorted(eligible["creator"].unique(),
                      key=lambda c: (-int((eligible["creator"] == c).sum()), c))

    for pass_index in range(cap):
        for creator in creators:
            if len(picked) >= n:
                break
            pool = remaining[remaining["creator"] == creator]
            if pool.empty:
                continue
            tertile = TERTILES[len(picked) % len(TERTILES)]
            chosen, relaxed = draw_from_creator(pool, rng, tertile, domain_used, bucket_used)
            picked.append(chosen["id"])
            creator_used[creator] += 1
            domain_used[chosen["domain"]] += 1
            bucket_used[chosen["bucket"]] += 1
            remaining = remaining[remaining["id"] != chosen["id"]]
            if relaxed:
                relaxations.append({"step": len(picked) - 1, "id": chosen["id"],
                                    "creator": creator, "pass": pass_index + 1, "relaxed": relaxed})
        if len(picked) >= n:
            break

    if len(picked) < n:
        relaxations.append({"outcome": "frame_exhausted_below_target",
                            "drawn": len(picked), "target": n, "cap": cap})
    return picked, relaxations


def write_atomic_text(text: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(dest)


def main() -> int:
    args = parse_args()
    started = datetime.now(timezone.utc)

    frame = pd.read_csv(args.frame, dtype={"id": str})
    n_raw = len(frame)

    excluded_ids: set[str] = set()
    if args.exclude_ids:
        excluded_ids = {line.strip() for line in args.exclude_ids.read_text().splitlines() if line.strip()}

    drop_reasons = Counter()
    eligible = frame.copy()
    for label, predicate in [
        ("no_libreface", lambda df: ~df["has_libreface"].fillna(False)),
        # V0 can only nominate inside the span LibreFace actually produced; a clip whose
        # series stops early would hand the LLM variants an unwinnable-for-V0 comparison.
        (f"libreface_coverage<{args.min_libreface_coverage}",
         lambda df: df["libreface_coverage"].fillna(0.0) < args.min_libreface_coverage),
        ("no_creator", lambda df: df["creator"].isna()),
        ("no_duration", lambda df: df["duration_s"].isna()),
        (f"duration<{args.min_duration}s", lambda df: df["duration_s"] < args.min_duration),
        ("excluded_domain", lambda df: df["domain"].isin(args.exclude_domains)),
        ("no_audio_track", lambda df: pd.Series(False, index=df.index) if args.allow_silent
         else ~df["has_audio"].fillna(False)),
        ("excluded_id", lambda df: df["id"].isin(excluded_ids)),
    ]:
        mask = predicate(eligible)
        drop_reasons[label] = int(mask.sum())
        eligible = eligible[~mask]

    eligible = eligible.sort_values("id").reset_index(drop=True)
    eligible["bucket"] = eligible["duration_s"].map(duration_bucket)
    eligible["tertile"] = (
        eligible.groupby("creator")["au_peak"].transform(lambda s: assign_tertiles(s))
    )
    eligible["creator_kind"] = eligible["creator_key"].map(creator_kind)

    n_creators = eligible["creator"].nunique()
    print(f"[sample] frame={n_raw} eligible={len(eligible)} creators={n_creators} "
          f"(from {eligible['creator_key'].nunique()} raw keys) domains={eligible['domain'].nunique()}", flush=True)
    if n_creators * args.creator_cap < args.n:
        print(f"[sample] WARNING: {n_creators} creators x cap {args.creator_cap} "
              f"< target {args.n}; the sample will fall short", flush=True)

    rng = np.random.default_rng(args.seed)
    picked_ids, relaxations = allocate(eligible, args.n, args.creator_cap, rng)
    sample = eligible[eligible["id"].isin(picked_ids)].sort_values(["domain", "id"]).reset_index(drop=True)

    for creator, n in sample["creator"].value_counts().items():
        pool_size = int((eligible["creator"] == creator).sum())
        print(f"[sample] {str(creator)[:46]:46} drawn={n:>2}  pool={pool_size:>6}", flush=True)

    over_cap = {c: n for c, n in sample["creator"].value_counts().items() if n > args.creator_cap}
    if over_cap:
        print(f"[sample] WARNING: {len(over_cap)} creator(s) exceed the cap of {args.creator_cap}: "
              f"{over_cap} -- see relaxations in sampling_record.json", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_atomic_text("\n".join(sorted(sample["id"])) + "\n", args.out_dir / "pilot_ids.txt")
    tmp = args.out_dir / "pilot_sample.csv.tmp"
    sample.to_csv(tmp, index=False)
    tmp.replace(args.out_dir / "pilot_sample.csv")

    record = {
        "generated_utc": started.isoformat(),
        "seed": args.seed,
        "script": "data_pipeline/05_nominate/sample_phase0.py",
        "frame_file": str(args.frame),
        "parameters": {
            "min_duration_s": args.min_duration,
            "creator_cap": args.creator_cap,
            "excluded_domains": args.exclude_domains,
            "allow_silent": args.allow_silent,
            "n_target": args.n,
            "excluded_ids_file": str(args.exclude_ids) if args.exclude_ids else None,
            "n_excluded_ids": len(excluded_ids),
            "duration_buckets": [b[2] for b in DURATION_BUCKETS],
            "tertile_basis": "LibreFace au_peak, ranked within domain",
        },
        "frame_size_raw": n_raw,
        "frame_size_eligible": int(len(eligible)),
        "drop_reasons": dict(drop_reasons),
        "n_drawn": int(len(sample)),
        "relaxations": relaxations,
        "domain_counts": sample["domain"].value_counts().to_dict(),
        "tertile_counts": sample["tertile"].value_counts().to_dict(),
        "bucket_counts": sample["bucket"].value_counts().to_dict(),
        "creator_counts": sample["creator"].value_counts().to_dict(),
        "creator_kind_counts": sample["creator_kind"].value_counts().to_dict(),
        "creators_over_cap": over_cap,
        "duration_stats": {
            "min": float(sample["duration_s"].min()),
            "median": float(sample["duration_s"].median()),
            "max": float(sample["duration_s"].max()),
            "total_s": float(sample["duration_s"].sum()),
        },
        "notes": [
            "Frame is the basic-QC pool; avg_intense_metadata_top3000.csv was deliberately NOT used.",
            "The plan's <=1 clip per creator is impossible on this pool (Comedy has 5 creator keys, "
            "Live-stream and Sports have 1 each), so the cap is 1 where a domain has enough keys and 2 otherwise.",
            "Creator keys of kind 'watch_url' (the game domain) identify a source video, not a channel; "
            "the cap there bounds videos rather than creators, and the game manifests carry no youtuber field.",
            "Nomination depth is top-2, not the plan's top-3: at a median clip length of ~60s, three "
            "nominations cover most of the clip and dilute precision into noise.",
        ],
    }
    write_atomic_text(json.dumps(record, indent=2, default=str), args.out_dir / "sampling_record.json")

    print(f"\n[sample] drew {len(sample)} clips, {sample['duration_s'].sum()/60:.1f} min total video")
    print(f"[sample] wrote {args.out_dir}/pilot_ids.txt, pilot_sample.csv, sampling_record.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
