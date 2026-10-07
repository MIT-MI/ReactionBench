#!/usr/bin/env python3
"""Cut the review-sample clips, upload to R2, and emit the review page's data.js.

For each row in review_sample.csv this cuts two short windows -- the reviewer's FACE reaction
and the STIMULUS event that supposedly triggered it -- uploads both to R2 under review/, and
writes 06_annotate/web/stimulus_review/data.js (window.ITEMS = [...]). The static page
lets a human eyeball, per item, whether the stimulus really explains the reaction before we
commit the 200/150 sets to Prolific. Data is embedded in data.js so the page opens from file://
with no manifest fetch. Idempotent: existing R2 objects are skipped.

  python data_pipeline/06_annotate/build_review_page.py --csv $RVB_WORK_DIR/stimulus_ranking/review_sample.csv
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
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_pilot_frame import default_data_root  # noqa: E402

REPO = cfg.PIPELINE_ROOT
R2_BUCKET = cfg.R2_BUCKET
R2_PUBLIC = cfg.R2_PUBLIC
# Face and stimulus share the clip timeline, so BOTH are cut to ONE common window spanning the
# stimulus event through the reaction. Equal length by construction, and playable in sync (see
# the review page). PAD gives lead-in before the event and tail after the reaction.
PAD_BEFORE, PAD_AFTER = 1.5, 1.5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", type=Path, default=default_data_root())
    p.add_argument("--csv", type=Path, default=cfg.work("stimulus_ranking/review_sample.csv"))
    p.add_argument("--out_js", type=Path, default=REPO / "06_annotate/web/stimulus_review/data.js")
    p.add_argument("--recut_stim_dir", type=Path,
                   default=cfg.RECUT_ROOT / "stimuli",
                   help="rows with stim_source=recut take their (aligned) stimulus from here")
    p.add_argument("--force", action="store_true", help="re-cut and overwrite even if the R2 object exists")
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


def cut(src: Path, s: float, e: float, dest: Path) -> None:
    ss = max(0.0, s)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{ss:.2f}", "-to", f"{e:.2f}",
        "-i", str(src), "-c:v", "libx264", "-preset", "veryfast", "-crf", "24", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(dest)],
        check=True, capture_output=True, timeout=300)


def main() -> int:
    args = parse_args()
    rows = list(csv.DictReader(open(args.csv)))
    face_root, stim_root = args.data_root / "face_origin", args.data_root / "stimuli"
    print(f"[review] {len(rows)} items")
    if args.dry_run:
        return 0

    r2 = r2_client()
    items = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, r in enumerate(rows):
            cid, vid = r["candidate_id"], r["video_id"]
            fk, sk = f"review/{cid}_face.mp4", f"review/{cid}_stim.mp4"
            start, end = float(r["start_s"]), float(r["end_s"])
            stim_s, stim_e = float(r["stim_s"]), float(r["stim_e"])
            # one window covering stimulus event + reaction, used for BOTH videos -> equal length, syncable
            w0 = max(0.0, min(stim_s, start) - PAD_BEFORE)
            w1 = max(stim_e, end) + PAD_AFTER
            stim_src = args.recut_stim_dir if r.get("stim_source") == "recut" else stim_root
            for src_root, key in ((face_root, fk), (stim_src, sk)):
                if args.force or not r2_has(r2, key):
                    d = Path(tmp) / Path(key).name
                    cut(src_root / f"{vid}.mp4", w0, w1, d)
                    r2.upload_file(str(d), R2_BUCKET, key, ExtraArgs={"ContentType": "video/mp4"})
                    d.unlink(missing_ok=True)
            items.append({
                "cid": cid, "video_id": vid, "domain": r["domain"],
                "reaction_mean": float(r["reaction_mean"]), "stim_quality": int(r["stim_quality"]),
                "self_contained": r["self_contained"] == "True", "match_ready": r["match_ready"] == "True",
                "dur_diff": float(r.get("dur_diff") or 0.0), "window_s": round(w1 - w0, 2),
                "stim_source": r.get("stim_source", "orig"),
                "event": r["event"], "faceUrl": R2_PUBLIC + fk, "stimUrl": R2_PUBLIC + sk,
            })
            print(f"  [{i + 1}/{len(rows)}] {cid}  window {w1 - w0:.1f}s", flush=True)

    args.out_js.parent.mkdir(parents=True, exist_ok=True)
    args.out_js.write_text("window.ITEMS = " + json.dumps(items, indent=2, ensure_ascii=False) + ";\n")
    print(f"\nwrote {args.out_js} ({len(items)} items)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
