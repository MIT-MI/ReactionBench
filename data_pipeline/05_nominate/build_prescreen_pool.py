#!/usr/bin/env python3
"""Apply the pre-screen QC gates to the sampling frame -> the pool the nominator runs on.

This filter originally lived in an analysis notebook; this script reproduces the archived pool
exactly (14,623 clips from the 31,834-clip frame;
14,582 after removing the 41 pilot clips that fall inside it).

Gates, applied to ``pilot_frame.csv`` from build_pilot_frame.py:

  has LibreFace output and series coverage >= 0.8   truncated AU series are a blind spot
  creator resolved, duration >= 10 s                need a real channel; too short to nominate in
  domain not Music / Sports                         music excluded by design; sports watch-alongs
                                                    never show the broadcast (no visible stimulus)
  has an audio track                                the prompt asks for audible cues
  not the last 1-2 segments of a Patreon source     outro segments are ~3x more often a frozen
                                                    stimulus (drop seg >= nseg - 1)

Patreon segment positions are parsed from ``original_face_path`` in ``_meta/mapping.csv``
(``.../<src>_face<k>_seg<NNN>.mp4``). Output ids are in id order, which for this pool is also
roughly short-clips-first (see prescreen_stats.py on why that matters).

Outputs (in --out_dir, default $RVB_WORK_DIR):
  prescreen_pool.csv             frame rows that pass every gate
  prescreen_ids.txt              their ids
  prescreen_ids_excl_pilot.txt   minus --pilot_ids (the ids the full-scale run used)
  prescreen_pool_record.json     per-gate counts
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "common"))
import rvb_config as cfg  # noqa: E402  (loads .env, puts every stage dir on sys.path)

import argparse
import json
import re
from pathlib import Path

import pandas as pd

EXCLUDED_DOMAINS = ("Music_reactions", "Sports_fan_watch-alongs")
SEG_RE = re.compile(r"/(\d+)_face(\d+)_seg(\d+)\.mp4$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frame", type=Path, default=cfg.work("pilot_frame.csv"))
    p.add_argument("--mapping", type=Path, default=cfg.meta_file("mapping.csv"))
    p.add_argument("--pilot_ids", type=Path, default=cfg.work("pilot_ids.txt"))
    p.add_argument("--out_dir", type=Path, default=cfg.WORK_DIR)
    p.add_argument("--min_coverage", type=float, default=0.8)
    p.add_argument("--min_duration", type=float, default=10.0)
    p.add_argument("--patreon_tail", type=int, default=2,
                   help="drop the last N segments of each Patreon source video (0 disables)")
    return p.parse_args()


def patreon_tail_ids(mapping: Path, tail: int) -> set:
    if tail <= 0:
        return set()
    mp = pd.read_csv(mapping, dtype=str)
    rows = []
    for new_id, path in zip(mp["new_id"], mp["original_face_path"].fillna("")):
        m = SEG_RE.search(path)
        if m and "patreon" in path.lower():
            rows.append((str(new_id).zfill(5), f"{m.group(1)}_{m.group(2)}", int(m.group(3))))
    seg = pd.DataFrame(rows, columns=["id", "src", "seg"])
    seg["nseg"] = seg.groupby("src")["seg"].transform("max")
    return set(seg.loc[seg["seg"] >= seg["nseg"] - (tail - 1), "id"])


def main() -> int:
    args = parse_args()
    f = pd.read_csv(args.frame, dtype={"id": str})
    gates = {
        "has_libreface": f["has_libreface"].astype(bool),
        "coverage": f["libreface_coverage"] >= args.min_coverage,
        "creator_resolved": f["creator"].notna(),
        "duration": f["duration_s"] >= args.min_duration,
        "domain": ~f["domain"].isin(EXCLUDED_DOMAINS),
        "has_audio": f["has_audio"].astype(bool),
    }
    keep = pd.Series(True, index=f.index)
    record = {"frame": int(len(f)), "after_gate": {}}
    for name, mask in gates.items():
        keep &= mask
        record["after_gate"][name] = int(keep.sum())
    tails = patreon_tail_ids(args.mapping, args.patreon_tail)
    keep &= ~((f["domain"] == "Patreon") & f["id"].isin(tails))
    record["after_gate"]["patreon_tail"] = int(keep.sum())

    pool = f[keep].sort_values("id")
    ids = pool["id"].tolist()
    pilot = set()
    if args.pilot_ids.exists():
        pilot = {l.strip() for l in args.pilot_ids.read_text().splitlines() if l.strip()}
    excl = [i for i in ids if i not in pilot]
    record.update(pool=len(ids), pilot_in_pool=len(ids) - len(excl), pool_excl_pilot=len(excl),
                  domains=pool["domain"].value_counts().to_dict(),
                  creators=int(pool["creator"].nunique()))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pool.to_csv(args.out_dir / "prescreen_pool.csv", index=False)
    (args.out_dir / "prescreen_ids.txt").write_text("\n".join(ids) + "\n")
    (args.out_dir / "prescreen_ids_excl_pilot.txt").write_text("\n".join(excl) + "\n")
    (args.out_dir / "prescreen_pool_record.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
