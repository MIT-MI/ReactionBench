#!/usr/bin/env python3
"""Sanity-check the full-scale pre-screen output against the pilot baseline.

The pre-screen JSONL is append-only and resumable, so an id that failed once and
succeeded on a later pass has BOTH an error record and an ok record. Dedupe by id,
keeping the successful record, before counting anything.

The headline comparison against the pilot must be duration-stratified. The ids file
is ordered short-clips-first (ids[0:1000] average ~19 s, ids[5000:] ~62 s) and the
empty-list rate falls monotonically with duration, so a raw pooled rate compared
against the pilot's 50 clips (mean 38 s) compares two different duration mixes.

Example:
  python data_pipeline/05_nominate/prescreen_stats.py \
      --jsonl $RVB_WORK_DIR/prescreen/prescreen_flash_low.jsonl \
      --pool $RVB_WORK_DIR/prescreen_pool.csv
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "common"))
import rvb_config as cfg  # noqa: E402  (loads .env, puts every stage dir on sys.path)

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict

import pandas as pd

# gemini-3.6-flash list price, USD per 1M tokens. Keep in sync with run_gemini_variant.py.
PRICE_IN, PRICE_OUT = 1.50, 7.50

# Bin edges chosen so the pilot's mean (38 s) and the pool's mean (55 s) land in
# different bins -- that separation is the whole point of stratifying.
DURATION_BINS = [0, 15, 30, 45, 58, 1e9]
DURATION_LABELS = ["<=15s", "15-30s", "30-45s", "45-58s", ">58s"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jsonl", type=Path,
                   default=cfg.work("prescreen/prescreen_flash_low.jsonl"))
    p.add_argument("--pool", type=Path, default=cfg.work("prescreen_pool.csv"))
    p.add_argument("--ids_file", type=Path,
                   default=cfg.work("prescreen_ids_excl_pilot.txt"))
    p.add_argument("--baseline", type=Path,
                   default=cfg.work("nominations/V2_flash_low.jsonl"),
                   help="pilot flash_low run, for the empty-rate/noms-per-clip comparison")
    p.add_argument("--baseline_pool", type=Path, default=cfg.work("pilot_sample.csv"))
    p.add_argument("--out_json", type=Path,
                   default=cfg.work("prescreen/prescreen_stats.json"))
    return p.parse_args()


def load_deduped(path: Path) -> tuple[Dict[str, dict], Counter]:
    """id -> record, preferring the successful record. Also counts error kinds."""
    best: Dict[str, dict] = {}
    error_kinds: Counter = Counter()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            error_kinds["malformed_jsonl_line"] += 1
            continue
        if record.get("error"):
            error_kinds[str(record["error"]).split(":")[0]] += 1
        prior = best.get(record["id"])
        # An ok record always wins; otherwise keep the first seen.
        if prior is None or (prior.get("error") and not record.get("error")):
            best[record["id"]] = record
    return best, error_kinds


def stratify(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["bin"] = pd.cut(frame["duration_s"], DURATION_BINS, labels=DURATION_LABELS)
    table = frame.groupby("bin", observed=True).agg(
        clips=("n_noms", "size"),
        empty_rate=("n_noms", lambda s: (s == 0).mean()),
        noms_per_clip=("n_noms", "mean"),
        mean_duration_s=("duration_s", "mean"),
    )
    return table.round(3)


def baseline_stats(jsonl: Path, pool: Path) -> dict:
    if not jsonl.exists():
        return {}
    records = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
    ok = [r for r in records if not r.get("error")]
    if not ok:
        return {}
    out = {
        "n": len(ok),
        "empty_rate": round(sum(1 for r in ok if not r["nominations"]) / len(ok), 4),
        "noms_per_clip": round(sum(len(r["nominations"]) for r in ok) / len(ok), 3),
    }
    if pool.exists():
        frame = pd.read_csv(pool, dtype={"id": str}).set_index("id")
        durations = [frame.at[r["id"], "duration_s"] for r in ok if r["id"] in frame.index]
        if durations:
            out["mean_duration_s"] = round(float(pd.Series(durations).mean()), 2)
    return out


def main() -> int:
    args = parse_args()
    best, error_kinds = load_deduped(args.jsonl)
    expected = [i.strip() for i in args.ids_file.read_text().splitlines() if i.strip()]

    ok = {i: r for i, r in best.items() if not r.get("error")}
    missing = [i for i in expected if i not in ok]
    unexpected = [i for i in best if i not in set(expected)]

    pool = pd.read_csv(args.pool, dtype={"id": str}).set_index("id")
    frame = pd.DataFrame({
        "id": list(ok),
        "n_noms": [len(r["nominations"]) for r in ok.values()],
    }).set_index("id")
    frame["duration_s"] = pool.reindex(frame.index)["duration_s"]

    n = len(ok)
    empty = int((frame["n_noms"] == 0).sum())
    total_noms = int(frame["n_noms"].sum())
    in_tok = sum(r["usage"]["input"] or 0 for r in ok.values())
    out_tok = sum(r["usage"]["output"] or 0 for r in ok.values())
    cost = in_tok / 1e6 * PRICE_IN + out_tok / 1e6 * PRICE_OUT

    scores = Counter(x["score"] for r in ok.values() for x in r["nominations"])
    flags = Counter(f for r in ok.values() for x in r["nominations"] for f in x["flags"])
    depth = Counter(frame["n_noms"])

    table = stratify(frame.dropna(subset=["duration_s"]))
    base = baseline_stats(args.baseline, args.baseline_pool)

    print(f"expected {len(expected)} ids | completed {n} | missing {len(missing)} | "
          f"unexpected {len(unexpected)}")
    print(f"error records in file: {sum(error_kinds.values())} {dict(error_kinds)}")
    print(f"empty lists: {empty} ({empty / n:.1%}) | nominations: {total_noms} "
          f"({total_noms / n:.2f}/clip)")
    print(f"nominations per clip: {sorted(depth.items())}")
    print(f"scores: {dict(sorted(scores.items()))} | flags: {dict(flags) or 'none'}")
    print(f"tokens: in={in_tok:,} out={out_tok:,} -> ${cost:.2f}")
    print(f"\nduration-stratified:\n{table.to_string()}")
    if base:
        print(f"\npilot baseline ({args.baseline.name}): {base}")
    if missing:
        print(f"\nfirst missing ids: {missing[:20]}")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({
        "jsonl": str(args.jsonl),
        "n_expected": len(expected),
        "n_completed": n,
        "missing_ids": missing,
        "unexpected_ids": unexpected,
        "error_records": dict(error_kinds),
        "empty_lists": empty,
        "empty_rate": round(empty / n, 4) if n else None,
        "total_nominations": total_noms,
        "noms_per_clip": round(total_noms / n, 3) if n else None,
        "noms_per_clip_hist": {str(k): v for k, v in sorted(depth.items())},
        "score_hist": {str(k): v for k, v in sorted(scores.items())},
        "flag_hist": dict(flags),
        "tokens_in": in_tok,
        "tokens_out": out_tok,
        "cost_usd": round(cost, 4),
        "by_duration": json.loads(table.reset_index().to_json(orient="records")),
        "pilot_baseline": base,
    }, indent=2))
    print(f"\nwrote {args.out_json}")
    return 1 if (missing or flags) else 0


if __name__ == "__main__":
    raise SystemExit(main())
