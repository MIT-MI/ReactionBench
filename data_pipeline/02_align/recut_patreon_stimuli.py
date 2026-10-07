#!/usr/bin/env python3
"""Regenerate an aligned stimulus for each desynced Patreon clip (needs the 180 s SOURCE segments).

Root cause: the clean pipeline stream-copied 60s segments out of the aligned 180s source, so
face and stimulus snapped to different keyframes and drifted apart. The 180s SOURCE face and
stimulus are perfectly aligned. Fix (annotation-preserving): keep the cleaned FACE untouched,
find its exact offset within the source face by audio cross-correlation, then cut the source
STIMULUS at that same window -> a stimulus aligned to the existing face, same duration.

Reads a list of clip ids; resolves source paths from mapping.csv and the intended window start
from mapping_lineage_part1.csv (used only to narrow the xcorr search). Writes recut stimuli to
OUT_DIR and a manifest.jsonl. Idempotent: existing outputs are skipped. Never touches raw data.

  python data_pipeline/02_align/recut_patreon_stimuli.py --ids ids.txt \
      [--data_root $RVB_DATA_ROOT] [--out_dir $RVB_RECUT_ROOT/stimuli] \
      [--path_rewrite OLD_PREFIX=NEW_PREFIX ...]

The source paths recorded in mapping.csv are absolute paths from the machine that built the
pool; ``--path_rewrite`` (repeatable) maps their prefix onto wherever the sources live now.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "common"))
import rvb_config as cfg  # noqa: E402  (loads .env, puts every stage dir on sys.path)
import argparse, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

# Set from CLI args in main().
CLEAN = cfg.DATA_ROOT
MAPPING = CLEAN / "mapping.csv"
LINEAGE = CLEAN / "mapping_lineage_part1.csv"
OUT_DIR = cfg.RECUT_ROOT / "stimuli"
SR = 16000
PATH_REWRITES: list = []  # [(old_prefix, new_prefix), ...]


def _meta_file(root: Path, name: str) -> Path:
    """mapping tables live at <root>/<name> on some hosts and <root>/_meta/<name> on others."""
    for cand in (root / name, root / "_meta" / name):
        if cand.exists():
            return cand
    return root / "_meta" / name


def resolve(p: str) -> str:
    if os.path.exists(p):
        return p
    for old, new in PATH_REWRITES:
        if p.startswith(old):
            alt = new + p[len(old):]
            if os.path.exists(alt):
                return alt
    return p


def audio(path: str, ss=None, t=None) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error"]
    if ss is not None: cmd += ["-ss", f"{ss:.3f}"]
    if t is not None: cmd += ["-t", f"{t:.3f}"]
    cmd += ["-i", path, "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32)


def probe_dur(path: str) -> float:
    o = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "csv=p=0", path], capture_output=True, text=True).stdout.strip()
    return float(o) if o else -1.0


def load_mapping() -> dict:
    m = {}
    import csv
    with open(MAPPING) as f:
        for row in csv.DictReader(f):
            stem = Path(row["new_face_origin_path"]).stem  # padded id
            m[stem] = (row["original_face_path"], row["original_stimuli_path"])
    return m


def load_lineage() -> dict:
    ln = {}
    import csv
    with open(LINEAGE) as f:
        for row in csv.DictReader(f):
            stem = Path(row["new_face_origin_path"]).stem
            try:
                ln[stem] = float(row["start_seconds"])
            except (ValueError, KeyError):
                pass
    return ln


def find_offset(cleaned_face: str, src_face: str, hint: float, face_dur: float):
    """Absolute offset (s) of the cleaned face within the source face, via audio xcorr.
    Search is narrowed to a window around the lineage hint when available."""
    cf = audio(cleaned_face); cf = cf - cf.mean()
    if hint is not None:
        w0 = max(0.0, hint - 8.0); w1 = min(180.5, hint + face_dur + 3.0)
    else:
        w0, w1 = 0.0, 181.0
    sw = audio(src_face, ss=w0, t=w1 - w0); sw = sw - sw.mean()
    if len(sw) <= len(cf) or len(cf) == 0:
        return None, 0.0
    n = 1
    while n < len(sw) + len(cf):
        n <<= 1
    corr = np.fft.irfft(np.fft.rfft(sw, n) * np.conj(np.fft.rfft(cf, n)), n)
    valid = corr[: len(sw) - len(cf) + 1]
    k = int(np.argmax(valid))
    # correlation coefficient at the peak, for QC
    seg = sw[k : k + len(cf)]
    cc = float(np.dot(seg, cf) / (np.linalg.norm(seg) * np.linalg.norm(cf) + 1e-9))
    return w0 + k / SR, cc


def cut(src_stim: str, offset: float, dur: float, dest: Path) -> None:
    tmp = dest.with_suffix(".tmp.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{offset:.3f}", "-t", f"{dur:.3f}",
        "-i", src_stim, "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(tmp)],
        check=True, capture_output=True, timeout=300)
    tmp.replace(dest)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--min_cc", type=float, default=0.12, help="flag offsets below this correlation")
    ap.add_argument("--data_root", type=Path, default=cfg.DATA_ROOT, help="clip pool (face_origin/, mapping.csv)")
    ap.add_argument("--out_dir", type=Path, default=cfg.RECUT_ROOT / "stimuli")
    ap.add_argument("--path_rewrite", action="append", default=[], metavar="OLD=NEW",
                    help="rewrite a source-path prefix recorded in mapping.csv (repeatable)")
    args = ap.parse_args()
    global CLEAN, MAPPING, LINEAGE, OUT_DIR, PATH_REWRITES
    CLEAN = args.data_root
    MAPPING = _meta_file(CLEAN, "mapping.csv")
    LINEAGE = _meta_file(CLEAN, "mapping_lineage_part1.csv")
    OUT_DIR = args.out_dir
    PATH_REWRITES = [tuple(x.split("=", 1)) for x in args.path_rewrite if "=" in x]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ids = [x.split(",")[0].strip() for x in Path(args.ids).read_text().splitlines() if x.strip()]
    mp, ln = load_mapping(), load_lineage()
    manifest = OUT_DIR / "manifest.jsonl"
    done = set()
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            try:
                r = json.loads(line)
                if r.get("ok"):
                    done.add(r["id"])
            except json.JSONDecodeError:
                pass
    todo = [i for i in ids if i not in done]
    print(f"[recut] {len(ids)} ids, {len(done)} done, {len(todo)} to run", flush=True)

    lock_lines = []
    import threading
    lock = threading.Lock()

    def run_one(cid: str) -> None:
        rec = {"id": cid}
        try:
            if cid not in mp:
                rec.update(ok=False, error="not in mapping"); return
            sf, ss = resolve(mp[cid][0]), resolve(mp[cid][1])
            cface = str(CLEAN / "face_origin" / f"{cid}.mp4")
            fdur = probe_dur(cface)
            if fdur <= 0 or not os.path.exists(sf) or not os.path.exists(ss):
                rec.update(ok=False, error=f"missing src (sf={os.path.exists(sf)} ss={os.path.exists(ss)} fdur={fdur})"); return
            # the same-window cut only aligns if the SOURCE face and stimulus are themselves aligned
            sfd, ssd = probe_dur(sf), probe_dur(ss)
            if sfd > 0 and ssd > 0 and abs(sfd - ssd) > 0.5:
                rec.update(ok=False, error=f"source not aligned (face={sfd:.2f} stim={ssd:.2f}) - cannot fix by same-window cut"); return
            off, cc = find_offset(cface, sf, ln.get(cid), fdur)
            if off is None:
                rec.update(ok=False, error="xcorr failed"); return
            dest = OUT_DIR / f"{cid}.mp4"
            cut(ss, off, fdur, dest)
            sdur = probe_dur(str(dest))
            rec.update(ok=True, offset=round(off, 3), corr=round(cc, 3), face_dur=round(fdur, 2),
                       stim_dur=round(sdur, 2), low_conf=cc < args.min_cc)
        except Exception as ex:  # noqa: BLE001
            rec.update(ok=False, error=f"{type(ex).__name__}: {ex}")
        finally:
            with lock:
                with open(manifest, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            tag = "OK " if rec.get("ok") else "ERR"
            print(f"  {tag} {cid} cc={rec.get('corr')} off={rec.get('offset')} {rec.get('error','')}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(run_one, todo))

    oks = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
    ok = [r for r in oks if r.get("ok")]
    low = [r for r in ok if r.get("low_conf")]
    print(f"\n[recut] done: {len(ok)} ok, {len(low)} low-confidence (cc<{args.min_cc}), "
          f"{len(oks) - len(ok)} errors. dir={OUT_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
