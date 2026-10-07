"""Split the intensity-mixture SFT files (run r2) by the `format` field for the single-format
ablations (runs r5-indep / r5-joint).

The ablation asks which of the two training formats drives the mixture's effect, with everything
else held fixed (same rows, same labels, same part order, same joint oversampling — the joint files keep the
x2 oversampled rows exactly as built, so JOINT-only sees each video twice per epoch just as
the mixture does).

    python -m finetune.split_by_format
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).parent


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=HERE / "data")
    a = p.parse_args()

    for split in ("train", "val"):
        rows = [json.loads(l) for l in (a.data / f"{split}.jsonl").read_text().splitlines() if l.strip()]
        for fmt in ("indep", "joint"):
            sel = [r for r in rows if r["format"] == fmt]
            out = a.data / f"{split}_{fmt}.jsonl"
            out.write_text("".join(json.dumps(r) + "\n" for r in sel))
            print(f"[split] {split:5s} {fmt:5s} -> {len(sel):5d} rows  {out}")
        assert sum(1 for r in rows if r["format"] in ("indep", "joint")) == len(rows), \
            "unexpected format value in " + str(split)


if __name__ == "__main__":
    main()
