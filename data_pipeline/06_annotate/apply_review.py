#!/usr/bin/env python3
"""Apply the rationale-set quality review: delete rejects, re-cut trimmed clips, report the count.

Reads gold_rationale_review. verdict='delete' -> drop the task (record it first) and remove its
R2 clips. verdict='keep' with a trim window -> re-cut face+stimulus to [w0+trim_start, w0+trim_end]
(w0 is the clip's original window start) and overwrite the R2 clips. Prints how many kept tasks
remain so backfill knows how many to add.

  python data_pipeline/06_annotate/apply_review.py --dry_run
  python data_pipeline/06_annotate/apply_review.py
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "common"))
import rvb_config as cfg  # noqa: E402  (loads .env, puts every stage dir on sys.path)

import argparse
import csv
import json
import os
import subprocess
import tempfile
import urllib.request
from pathlib import Path

REPO = cfg.PIPELINE_ROOT
R2_BUCKET = cfg.R2_BUCKET
PAD = 1.5


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = cfg.work("stimulus_ranking")
    p.add_argument("--set_csv", type=Path, nargs="+", default=[d / "rationale_set.csv", d / "backfill_set.csv"])
    p.add_argument("--deleted_out", type=Path, default=d / "rationale_deleted.csv")
    p.add_argument("--face_dir", type=Path, default=cfg.DATA_ROOT / "face_origin")
    p.add_argument("--stim_dir", type=Path, default=cfg.DATA_ROOT / "stimuli")
    p.add_argument("--recut_dir", type=Path, default=cfg.RECUT_ROOT / "stimuli")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def sb(path):
    U, K = os.environ["SUPABASE_URL"], os.environ["SUPABASE_SECRET_KEY"]
    out, step = [], 1000
    for frm in range(0, 10_000_000, step):
        req = urllib.request.Request(f"{U}/rest/v1/{path}",
            headers={"apikey": K, "Authorization": f"Bearer {K}", "Range": f"{frm}-{frm + step - 1}"})
        b = json.load(urllib.request.urlopen(req)); out += b
        if len(b) < step:
            return out
    return out


def sb_delete(path):
    U, K = os.environ["SUPABASE_URL"], os.environ["SUPABASE_SECRET_KEY"]
    urllib.request.urlopen(urllib.request.Request(f"{U}/rest/v1/{path}", method="DELETE",
        headers={"apikey": K, "Authorization": f"Bearer {K}"}))


def sb_patch(path, body):
    U, K = os.environ["SUPABASE_URL"], os.environ["SUPABASE_SECRET_KEY"]
    urllib.request.urlopen(urllib.request.Request(f"{U}/rest/v1/{path}", data=json.dumps(body).encode(),
        method="PATCH", headers={"apikey": K, "Authorization": f"Bearer {K}", "Content-Type": "application/json",
        "Prefer": "return=minimal"}))


def r2():
    import boto3
    from botocore.config import Config
    return boto3.client("s3", endpoint_url=os.environ["R2_S3_API"],
        aws_access_key_id=os.environ["R2_ACCESS_SECRET_ID"], aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto", config=Config(signature_version="s3v4"))


def cut(src, s, e, dest):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{max(0.0, s):.2f}", "-to", f"{e:.2f}",
        "-i", str(src), "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(dest)], check=True, capture_output=True, timeout=300)


def main():
    args = parse_args()
    rv = {r["candidate_id"]: r for r in sb("gold_rationale_review?select=candidate_id,verdict,trim_start,trim_end,note")}
    # window params for trims come from the original set plus every backfill round
    set_csvs = list(args.set_csv) + sorted((cfg.work("stimulus_ranking")).glob("backfill_set*.csv"))
    setrows = {}
    for sc in dict.fromkeys(set_csvs):          # de-dup while keeping order
        if Path(sc).exists():
            for r in csv.DictReader(open(sc)):
                setrows[r["candidate_id"]] = r
    deletes = [cid for cid, r in rv.items() if r["verdict"] == "delete"]
    trims = [cid for cid, r in rv.items() if r["verdict"] == "keep" and (r["trim_start"] is not None or r["trim_end"] is not None)]
    keeps = [cid for cid, r in rv.items() if r["verdict"] == "keep"]
    print(f"reviewed {len(rv)}: {len(keeps)} keep ({len(trims)} trimmed), {len(deletes)} delete")
    print(f"-> kept tasks after apply: {len(keeps)}; need {200 - len(keeps)} backfill to reach 200")
    if args.dry_run:
        return 0

    # record deletes cumulatively (keep the earlier round's record)
    prior = {r["candidate_id"]: r.get("note", "") for r in csv.DictReader(open(args.deleted_out))} if args.deleted_out.exists() else {}
    for cid in deletes:
        prior[cid] = rv[cid].get("note") or prior.get(cid, "")
    with open(args.deleted_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["candidate_id", "note"]); w.writeheader()
        for cid, note in prior.items():
            w.writerow({"candidate_id": cid, "note": note})

    c = r2()
    # deletes: clear all FK references (rationales, review) then task + R2 clips. Idempotent.
    for cid in deletes:
        for suf in ("_face.mp4", "_stim.mp4"):
            try:
                c.delete_object(Bucket=R2_BUCKET, Key=f"stagec/{cid}{suf}")
            except Exception:
                pass
        sb_delete(f"gold_rationales?candidate_id=eq.{cid}")
        sb_delete(f"gold_rationale_review?candidate_id=eq.{cid}")
        sb_delete(f"gold_rationale_tasks?candidate_id=eq.{cid}")
    print(f"deleted {len(deletes)} tasks + clips")

    # trims: re-cut to the kept sub-window and overwrite
    with tempfile.TemporaryDirectory() as tmp:
        for cid in trims:
            r, row = rv[cid], setrows.get(cid)
            if not row:
                print(f"  trim skip {cid}: not in set_csv"); continue
            start, end = float(row["start_s"]), float(row["end_s"])
            ss, se = float(row["stim_s"]), float(row["stim_e"])
            w0 = max(0.0, min(ss, start) - PAD)
            ta = w0 + (r["trim_start"] if r["trim_start"] is not None else 0.0)
            tb = w0 + (r["trim_end"] if r["trim_end"] is not None else (max(se, end) + PAD - w0))
            if tb - ta < 0.8:
                print(f"  trim skip {cid}: window too short {ta:.2f}-{tb:.2f}"); continue
            stim_src = (args.recut_dir if row.get("stim_source") == "recut" else args.stim_dir) / f'{row["video_id"]}.mp4'
            for src, key in ((args.face_dir / f'{row["video_id"]}.mp4', f"stagec/{cid}_face.mp4"),
                             (stim_src, f"stagec/{cid}_stim.mp4")):
                d = Path(tmp) / Path(key).name
                cut(src, ta, tb, d)
                c.upload_file(str(d), R2_BUCKET, key, ExtraArgs={"ContentType": "video/mp4"}); d.unlink(missing_ok=True)
            sb_patch(f"gold_rationale_tasks?candidate_id=eq.{cid}",
                     {"start_s": round(max(0.0, start - ta), 2), "end_s": round(max(0.0, end - ta), 2)})
    print(f"re-cut {len(trims)} trimmed clips. kept gold rationale tasks: {len(keeps)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
