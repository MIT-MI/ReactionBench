#!/usr/bin/env python3

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "common"))
import rvb_config as cfg  # noqa: E402  (loads .env, puts every stage dir on sys.path)
import argparse, os, re, csv, subprocess, json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd

# ---- Label subfolder names used by the download script ----
LABEL_MAP = {
    # "Anime_reactions": "Anime reactions",
    # "Comedy_internet_culture": "Comedy & internet culture",
    # "Experts_react": "Experts react",
    # "Food_chef_reactions": "Food & chef reactions",
    # "game": "Game",
    # "General_pop-culture_React": "General pop-culture React",
    # "K-pop_international": "K-pop international",
    # "Live-stream_reacts": "Live-stream reacts",
    # "Movies_TV": "Movies & TV",
    # "Music_reactions": "Music reactions",
    # "Sports_fan_watch-alongs": "Sports & fan watch-alongs",
    "TikTok_reactions": "TikTok reactions",
}
def canon_label_from_folder(folder: str) -> str:
    return LABEL_MAP.get(folder, folder.replace("_"," "))

# ---- Duration and path helpers ----
DUR_RE = re.compile(r"^(?:(\d+):)?(\d{1,2}):(\d{2})$")
def parse_duration_to_seconds(s: str) -> Optional[int]:
    if not isinstance(s,str) or not s.strip(): return None
    m = DUR_RE.match(s.strip()); 
    if not m: return None
    h = int(m.group(1) or 0); m_ = int(m.group(2)); s_ = int(m.group(3))
    return h*3600 + m_*60 + s_

def ffprobe_duration_seconds(p: Path) -> Optional[int]:
    try:
        out = subprocess.run(
            ["ffprobe","-v","error","-select_streams","v:0",
             "-show_entries","format=duration",
             "-of","default=nk=1:nw=1", str(p)],
            check=True, capture_output=True, text=True
        ).stdout.strip()
        val = float(out); 
        if val<=0: return None
        return int(round(val))
    except Exception:
        return None

def ffprobe_tags(p: Path) -> dict:
    """Best-effort read of container tags such as title/comment; they may be absent."""
    try:
        out = subprocess.run(
            ["ffprobe","-v","error",
             "-show_entries","format_tags",
             "-of","json", str(p)],
            check=True, capture_output=True, text=True
        ).stdout
        j = json.loads(out)
        return j.get("format",{}).get("tags",{}) or {}
    except Exception:
        return {}

def find_low_variant(hi_path: Path) -> Optional[Path]:
    stem = hi_path.stem
    cand = hi_path.with_name(stem + "_low.mp4")
    return cand if cand.exists() else None

def find_subtitle_path(subs_dir: Path, stem: str) -> str:
    p = subs_dir / f"{stem}.en.srt"
    if p.exists(): return str(p)
    hits = sorted(subs_dir.glob(f"{stem}.*.srt"))
    return str(hits[0]) if hits else ""

# ---- Core: global one-to-one matching within each label ----
def global_match_by_duration(
    locals_list: List[Tuple[Path,int]],  # [(hi_path, dur)]
    candidates_df: pd.DataFrame,         # must contain duration_seconds, video_link, ...
    tol: int
):
    """
    Returns:
      matches: List[(i_local, row(Series), hi_path)]
      ambiguous: List[(hi_path, dur, [video_link...])]
    Strategy:
        1) build edges for every pair with |dt| <= tol
        2) first match every degree-1 (unambiguous) edge
        3) for the remaining nodes, greedily take the free candidate with the smallest |dt|
    """
    # Build adjacency
    idxs = list(range(len(candidates_df)))
    cand = candidates_df.reset_index(drop=True)
    neigh = {}  # hi_idx -> [cand_idx...]
    for i,(p,dur) in enumerate(locals_list):
        ok = cand.index[(cand["duration_seconds"]-dur).abs()<=tol].tolist()
        neigh[i] = ok

    used_cand = set()
    matched = {}
    # (1) Unambiguous degree-1 matches first
    changed = True
    while changed:
        changed = False
        for i, nb in neigh.items():
            if i in matched: continue
            free = [j for j in nb if j not in used_cand]
            if len(free)==1:
                j = free[0]
                matched[i] = j
                used_cand.add(j)
                changed = True

    # (2) Greedy by smallest |dt| for the rest
    for i,(p,dur) in enumerate(locals_list):
        if i in matched:
            continue
        candidates = [(j, abs(int(cand.loc[j,"duration_seconds"])-dur)) 
                      for j in neigh[i] if j not in used_cand]
        candidates.sort(key=lambda x: x[1])
        if candidates:
            j,_ = candidates[0]
            matched[i] = j
            used_cand.add(j)

    # Assemble results; unmatched local files are reported as ambiguous
    matches = []
    ambiguous = []
    for i,(p,dur) in enumerate(locals_list):
        if i in matched:
            j = matched[i]
            row = cand.loc[j]
            matches.append((i, row, locals_list[i][0]))
        else:
            cands = [cand.loc[j,"video_link"] for j in neigh[i]]
            ambiguous.append((locals_list[i][0], dur, cands))
    return matches, ambiguous

def main():
    ap = argparse.ArgumentParser("Recover mapping from initial CSV + raw folder")
    ap.add_argument("--orig-csv", required=True, help="Initial catalog CSV (with youtuber,playlist,video_link,video_title,label,duration,num_reactors)")
    ap.add_argument("--raw-root", required=True, help="raw root directory, e.g. $RVB_RAW_ROOT/raw")
    ap.add_argument("--transcript-root", default=None, help="Subtitle root directory (default <raw-root>/../transcript)")
    ap.add_argument("--out-dir", required=True, help="Output directory: one *_recovered.csv per label plus a report")
    ap.add_argument("--tolerance", type=int, default=2, help="Duration matching tolerance in seconds (default 2)")
    ap.add_argument("--try-tags", action="store_true", help="Also use container metadata (title/comment) to pin matches (fast, but often missing)")
    ap.add_argument("--only-nr-eq1", action="store_true", default=True, help="Only use candidates with num_reactors==1 (on by default)")
    args = ap.parse_args()

    raw_root = Path(args.raw_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    transcript_root = Path(args.transcript_root).resolve() if args.transcript_root else (raw_root.parent/"transcript")

    # 1) Read the initial catalog CSV
    orig = pd.read_csv(args.orig_csv)
    need = ["youtuber","playlist","video_link","video_title","label","duration","num_reactors"]
    for c in need:
        if c not in orig.columns: raise SystemExit(f"Missing column: {c}")

    if "duration_seconds" not in orig.columns:
        orig["duration_seconds"] = orig["duration"].apply(parse_duration_to_seconds)
    orig = orig[orig["duration_seconds"].notna()].copy()
    orig["duration_seconds"] = orig["duration_seconds"].astype(int)

    if args.only_nr_eq1:
        def _f(x):
            try: return float(x)
            except: return None
        orig = orig[orig["num_reactors"].apply(_f)==1.0].copy()

    # 2) Group candidates by label
    by_label = {lbl: df.copy() for lbl, df in orig.groupby("label")}

    # 3) Scan local raw/*
    label_dirs = [d for d in raw_root.iterdir() if d.is_dir()]
    out_cols = ["youtuber","playlist","video_link","video_title","label",
                "duration","num_reactors","duration_seconds","selected",
                "local_path","local_path_low","subtitle_path"]

    total_match, total_amb = 0, 0
    rep_rows = []

    for d in sorted(label_dirs):
        canon = canon_label_from_folder(d.name)
        cand_df = by_label.get(canon)
        if cand_df is None or cand_df.empty:
            # Label exists locally but has no candidates in the initial CSV
            for p in sorted(d.glob("*.mp4")):
                if p.stem.endswith("_low"): continue
                dur = ffprobe_duration_seconds(p) or -1
                rep_rows.append([canon, str(p), dur, ""])
            continue

        # Full-resolution files for this label
        locals_list = []
        stems = []  # keep stems to locate subtitles
        for p in sorted(d.glob("*.mp4")):
            if p.stem.endswith("_low"): continue
            dur = ffprobe_duration_seconds(p)
            if dur is None: 
                continue
            locals_list.append((p, dur))
            stems.append(p.stem)

        if not locals_list:
            continue

        # Optionally try a fast metadata hit first:
        # a unique candidate found this way is reserved before the global matching
        pre_matches = {}
        used_links = set()
        if args.try_tags:
            link_set = set(cand_df["video_link"])
            for i,(p,dur) in enumerate(locals_list):
                tags = ffprobe_tags(p)
                text = " ".join([str(v) for v in tags.values()]).lower()
                # Minimal heuristic: look for the link itself in the tag text
                hits = [link for link in link_set if link.lower() in text]
                if len(hits)==1:
                    pre_matches[i] = cand_df[cand_df["video_link"]==hits[0]].index[0]
                    used_links.add(hits[0])

        # Drop pre-matched candidates, then run the global duration matching
        rest_locals = [(p,d) for k,(p,d) in enumerate(locals_list) if k not in pre_matches]
        rest_cands = cand_df[~cand_df["video_link"].isin(used_links)].copy().reset_index(drop=True)
        matches, amb = global_match_by_duration(rest_locals, rest_cands, args.tolerance)

        # Merge pre-matches
        full_matches = []
        # Map rest_idx back to the original locals index
        rest_idx_map = {}
        r = 0
        for k in range(len(locals_list)):
            if k not in pre_matches:
                rest_idx_map[r] = k
                r += 1
        for i_rest, row, p in matches:
            k = rest_idx_map[i_rest]  # original locals index
            full_matches.append((row, p, stems[k]))
        for k,j in pre_matches.items():
            row = cand_df.loc[j]
            full_matches.append((row, locals_list[k][0], stems[k]))

        # Write the per-label CSV
        out_rows = []
        seen_links = set()
        for row, hi_path, stem in full_matches:
            low_path = find_low_variant(hi_path)
            subs_dir = transcript_root / d.name
            out_rows.append({
                "youtuber": row.get("youtuber",""),
                "playlist": row.get("playlist",""),
                "video_link": row.get("video_link",""),
                "video_title": row.get("video_title",""),
                "label": row.get("label", canon),
                "duration": row.get("duration",""),
                "num_reactors": row.get("num_reactors",""),
                "duration_seconds": int(row.get("duration_seconds",0) or 0),
                "selected": False,
                "local_path": str(hi_path),
                "local_path_low": str(low_path) if low_path else "",
                "subtitle_path": find_subtitle_path(subs_dir, stem),
            })
            seen_links.add(row.get("video_link",""))

        # Record ambiguous/unmatched files
        for (p, dur, cand_links) in amb:
            rep_rows.append([canon, str(p), dur, ";".join(cand_links)])

        if out_rows:
            safe = re.sub(r"[^a-z0-9]+","_", canon.lower()).strip("_")
            out_csv = out_dir / f"{safe}_recovered.csv"
            pd.DataFrame(out_rows, columns=out_cols).to_csv(out_csv, index=False, quoting=csv.QUOTE_MINIMAL)
            print(f"[Write] {canon}: {len(out_rows)} -> {out_csv}")
            total_match += len(out_rows)

    # Overall report
    report = out_dir / "ambiguous_or_unmatched.csv"
    with open(report,"w",newline="") as f:
        w = csv.writer(f); w.writerow(["label","local_path","local_duration_seconds","candidate_video_links"])
        w.writerows(rep_rows)
    total_amb = len(rep_rows)
    print(f"[Summary] matched={total_match}, ambiguous/unmatched={total_amb}, report={report}")

if __name__ == "__main__":
    main()