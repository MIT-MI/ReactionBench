#!/usr/bin/env python3
"""Load the 200-clip rationale gold set into Stage C: R2 window clips + gold_rationale_tasks.

For each candidate, cut ONE shared window (stimulus event through reaction) from the face and the
stimulus — the recut overlay for stim_source=recut — so the Stage C site can sync-play them, and
upload both to R2 under stagec/. Insert a task carrying the Qwen draft (pre-filled for correction),
the cultural-specificity tag, and the consensus score. Idempotent: skips clips already uploaded.

  python data_pipeline/06_annotate/load_stage_c_rationale.py --dry_run
  python data_pipeline/06_annotate/load_stage_c_rationale.py
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
import sys
import tempfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = cfg.PIPELINE_ROOT
R2_BUCKET = cfg.R2_BUCKET
R2_PUBLIC = cfg.R2_PUBLIC
PAD = 1.5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = cfg.work("stimulus_ranking")
    p.add_argument("--set_csv", type=Path, default=d / "rationale_set.csv")
    p.add_argument("--drafts", type=Path, default=d / "rationale_drafts.jsonl")
    p.add_argument("--cultural", type=Path, default=d / "cultural.jsonl")
    p.add_argument("--face_dir", type=Path, default=cfg.DATA_ROOT / "face_origin")
    p.add_argument("--stim_dir", type=Path, default=cfg.DATA_ROOT / "stimuli")
    p.add_argument("--recut_dir", type=Path, default=cfg.RECUT_ROOT / "stimuli")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def r2_client():
    import boto3
    from botocore.config import Config
    return boto3.client("s3", endpoint_url=os.environ["R2_S3_API"],
        aws_access_key_id=os.environ["R2_ACCESS_SECRET_ID"], aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto", config=Config(signature_version="s3v4"))


def r2_has(c, key: str) -> bool:
    try:
        c.head_object(Bucket=R2_BUCKET, Key=key); return True
    except Exception:
        return False


def cut(src: Path, w0: float, w1: float, dest: Path) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{max(0.0, w0):.2f}", "-to", f"{w1:.2f}",
        "-i", str(src), "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(dest)], check=True, capture_output=True, timeout=300)


def sb_upsert(table: str, on_conflict: str, rows: list) -> None:
    U, K = os.environ["SUPABASE_URL"], os.environ["SUPABASE_SECRET_KEY"]
    req = urllib.request.Request(f"{U}/rest/v1/{table}?on_conflict={on_conflict}",
        data=json.dumps(rows).encode(), method="POST",
        headers={"apikey": K, "Authorization": f"Bearer {K}", "Content-Type": "application/json",
                 "Prefer": "resolution=merge-duplicates,return=minimal"})
    urllib.request.urlopen(req)


def load_jsonl(path: Path, field: str) -> dict:
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if not r.get("error") and field in r:
                    out[r["candidate_id"]] = r[field]
    return out


def main() -> int:
    args = parse_args()
    rows = list(csv.DictReader(open(args.set_csv)))
    drafts = load_jsonl(args.drafts, "draft")
    cultural = load_jsonl(args.cultural, "cultural_specificity")
    print(f"[stagec] {len(rows)} rationale clips, {len(drafts)} drafts, {len(cultural)} cultural tags")
    if args.dry_run:
        miss = [r["candidate_id"] for r in rows if r["candidate_id"] not in drafts]
        print(f"  missing drafts: {len(miss)} {miss[:5]}")
        return 0

    r2 = r2_client()
    tasks = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, r in enumerate(rows):
            cid, vid = r["candidate_id"], r["video_id"]
            start, end = float(r["start_s"]), float(r["end_s"])
            ss, se = float(r["stim_s"]), float(r["stim_e"])
            w0 = max(0.0, min(ss, start) - PAD); w1 = max(se, end) + PAD
            fk, sk = f"stagec/{cid}_face.mp4", f"stagec/{cid}_stim.mp4"
            stim_src = (args.recut_dir if r.get("stim_source") == "recut" else args.stim_dir) / f"{vid}.mp4"
            for src, key in ((args.face_dir / f"{vid}.mp4", fk), (stim_src, sk)):
                if args.force or not r2_has(r2, key):
                    dpath = Path(tmp) / Path(key).name
                    cut(src, w0, w1, dpath)
                    r2.upload_file(str(dpath), R2_BUCKET, key, ExtraArgs={"ContentType": "video/mp4"})
                    dpath.unlink(missing_ok=True)
            tasks.append({
                "candidate_id": cid, "video_id": vid,
                "start_s": round(start - w0, 2), "end_s": round(end - w0, 2),
                "face_r2_key": fk, "stimuli_r2_key": sk,
                "llm_draft": drafts.get(cid, ""),
                "consensus_score": float(r["reaction_mean"]),
                "cultural_specificity": cultural.get(cid),
                "stim_source": r.get("stim_source", "orig"),
            })
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(rows)} uploaded", flush=True)
    sb_upsert("gold_rationale_tasks", "candidate_id", tasks)
    print(f"[stagec] inserted {len(tasks)} rationale tasks. Base URL {R2_PUBLIC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
