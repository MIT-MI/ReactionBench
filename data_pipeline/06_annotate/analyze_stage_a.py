#!/usr/bin/env python3
"""Analyse the Stage A gold scores: agreement, subjectivity structure, and covariates.

Pulls gold_scores joined with creator/domain, computes chance-corrected agreement and the
structure of disagreement, and writes figures + a stats.json for the analysis write-up.
Note that "within-1 on a 4-point scale" is not evidence of agreement (chance agreement is
already high), so we report Krippendorff's alpha and characterise WHERE raters disagree.

Outputs: $RVB_WORK_DIR/gold_analysis/*.png and stats.json
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "common"))
import rvb_config as cfg  # noqa: E402  (loads .env, puts every stage dir on sys.path)

import json
import os
import urllib.request
from collections import defaultdict
from pathlib import Path

import krippendorff
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = cfg.work("gold_analysis")
BG, FG, AC, WARN, OK = "#0f1115", "#e6e9ef", "#6ea8fe", "#f87171", "#4ade80"


def style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color("#333")
    ax.tick_params(colors=FG, labelsize=9)
    ax.title.set_color(FG); ax.xaxis.label.set_color(FG); ax.yaxis.label.set_color(FG)
    if title: ax.set_title(title, fontsize=11)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(BG)
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=130, facecolor=BG)
    plt.close(fig)


def sb_all(path):
    U, K = cfg.supabase()
    out, step = [], 1000
    for frm in range(0, 2_000_000, step):
        req = urllib.request.Request(f"{U}/rest/v1/{path}",
            headers={"apikey": K, "Authorization": f"Bearer {K}", "Range": f"{frm}-{frm+step-1}"})
        b = json.load(urllib.request.urlopen(req)); out += b
        if len(b) < step:
            return out
    return out


def load():
    scores = pd.DataFrame(sb_all("gold_scores?select=candidate_id,annotator_id,score,cue_type"))
    cands = pd.DataFrame(sb_all("gold_candidates?select=candidate_id,video_id"))
    samp = pd.read_csv(cfg.work("gold/gold_stage_a_sample.csv"), dtype={"id": str})[["id", "creator", "domain"]]
    return scores.merge(cands, on="candidate_id").merge(samp, left_on="video_id", right_on="id", how="left")


def short_creator(c):
    if not isinstance(c, str):
        return "unknown"
    if c.startswith("http"):
        return c.rstrip("/").split("/")[-1][:10]
    return c[:16]


def main():
    df = load()
    byc = df.groupby("candidate_id")["score"].apply(list)
    stats = {"n_scores": len(df), "n_candidates": int(byc.size), "n_annotators": int(df.annotator_id.nunique())}

    # ---- chance-corrected agreement ----
    marg = df.score.value_counts(normalize=True)
    p = np.array([marg.get(i, 0) for i in range(4)])
    chance_w1 = float(sum(p[i]*p[j] for i in range(4) for j in range(4) if abs(i-j) <= 1))
    chance_exact = float((p**2).sum())
    obs_w1 = float(np.mean([max(v)-min(v) <= 1 for v in byc]))
    obs_exact = float(np.mean([len(set(v)) == 1 for v in byc]))
    piv = df.pivot_table(index="annotator_id", columns="candidate_id", values="score", aggfunc="first")
    alpha_ord = float(krippendorff.alpha(reliability_data=piv.values, level_of_measurement="ordinal"))
    alpha_nom = float(krippendorff.alpha(reliability_data=piv.values, level_of_measurement="nominal"))
    stats.update(within1_obs=obs_w1, within1_chance=chance_w1, exact_obs=obs_exact,
                 exact_chance=chance_exact, alpha_ordinal=alpha_ord, alpha_nominal=alpha_nom,
                 score_marginals={str(i): float(p[i]) for i in range(4)})

    # Fig 1: observed vs chance agreement
    fig, ax = plt.subplots(figsize=(6, 3.4))
    x = np.arange(2)
    ax.bar(x-0.2, [obs_w1, obs_exact], 0.4, label="observed", color=AC)
    ax.bar(x+0.2, [chance_w1, chance_exact], 0.4, label="chance", color="#555")
    ax.set_xticks(x); ax.set_xticklabels(["within-1", "exact"])
    for i, (o, c) in enumerate([(obs_w1, chance_w1), (obs_exact, chance_exact)]):
        ax.text(i-0.2, o+.02, f"{o:.2f}", ha="center", color=FG, fontsize=9)
        ax.text(i+0.2, c+.02, f"{c:.2f}", ha="center", color=FG, fontsize=9)
    style(ax, "Agreement vs chance — 'within-1' is below chance", ylabel="fraction of candidates")
    ax.set_ylim(0, 1.05); ax.legend(facecolor=BG, labelcolor=FG, edgecolor="#333")
    save(fig, "01_agreement_vs_chance.png")

    # Fig 2: Krippendorff alpha gauge + rating-pair |diff|
    pd_ = defaultdict(int)
    for v in byc:
        for i in range(len(v)):
            for j in range(i+1, len(v)):
                pd_[abs(v[i]-v[j])] += 1
    tot = sum(pd_.values())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
    a1.barh([0], [alpha_ord], color=AC); a1.barh([0], [1-alpha_ord], left=[alpha_ord], color="#333")
    a1.set_xlim(0, 1); a1.set_yticks([])
    for t, lbl in [(0.667, "0.667"), (0.8, "0.8")]:
        a1.axvline(t, color=WARN, ls="--", lw=1); a1.text(t, 0.4, lbl, color=WARN, fontsize=8, ha="center")
    a1.text(alpha_ord, -0.42, f"α_ordinal = {alpha_ord:.3f}", color=FG, ha="center", fontsize=11)
    style(a1, "Krippendorff's ordinal α (chance-corrected)")
    diffs = sorted(pd_)
    a2.bar([str(d) for d in diffs], [pd_[d]/tot for d in diffs], color=AC)
    for d in diffs: a2.text(str(d), pd_[d]/tot+.01, f"{pd_[d]/tot:.2f}", ha="center", color=FG, fontsize=9)
    style(a2, "Distance between two raters on the same moment", xlabel="|score difference|", ylabel="fraction of rating pairs")
    save(fig, "02_alpha_and_pairdiff.png")

    # Fig 3: where disagreement lives — adjacent-boundary breakdown
    b = defaultdict(int)
    for v in byc:
        for i in range(len(v)):
            for j in range(i+1, len(v)):
                if abs(v[i]-v[j]) == 1:
                    b[tuple(sorted((v[i], v[j])))] += 1
    tb = sum(b.values())
    keys = [(0, 1), (1, 2), (2, 3)]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    vals = [b[k]/tb for k in keys]
    cols = [OK, WARN, "#fbbf24"]
    ax.bar([f"{k[0]}↔{k[1]}" for k in keys], vals, color=cols)
    for i, k in enumerate(keys):
        ax.text(i, vals[i]+.01, f"{vals[i]:.0%}", ha="center", color=FG, fontsize=10)
    style(ax, "Adjacent disagreements: where do raters split?", xlabel="boundary", ylabel="share of 1-point disagreements")
    ax.text(0.5, 0.9, '"is there a reaction" (0↔1) is agreed;\n"how strong" (1↔2) is where it splits',
            transform=ax.transAxes, color=FG, fontsize=8, ha="center")
    stats["disagreement_boundary"] = {f"{k[0]}-{k[1]}": b[k]/tb for k in keys}
    save(fig, "03_disagreement_boundary.png")

    # Fig 4: rating std vs consensus mean (fuzzy region)
    mean = byc.apply(np.mean); std = byc.apply(lambda v: float(np.std(v)))
    bins = [(0, 0.5), (0.5, 1.5), (1.5, 2.5), (2.5, 3.01)]
    labels = ["~0", "~1", "~2", "~3"]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ys, ns = [], []
    for lo, hi in bins:
        m = (mean >= lo) & (mean < hi); ys.append(std[m].mean()); ns.append(int(m.sum()))
    ax.bar(labels, ys, color=AC)
    for i, (y, n) in enumerate(zip(ys, ns)):
        ax.text(i, y+.005, f"{y:.2f}\n(n={n})", ha="center", color=FG, fontsize=8)
    style(ax, "Disagreement by intensity level", xlabel="consensus mean score", ylabel="mean rating std within a moment")
    save(fig, "04_std_by_level.png")

    # Fig 5: pass/fail (>=2) contestedness
    pf = byc.apply(lambda v: np.mean([s >= 2 for s in v]))
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.hist(pf, bins=np.linspace(0, 1, 11), color=AC, edgecolor=BG)
    style(ax, "Pass threshold (score ≥ 2): fraction of raters who 'pass' a moment",
          xlabel="fraction of raters scoring ≥ 2", ylabel="candidates")
    unan = float(np.mean((pf == 1) | (pf == 0)))
    ax.text(0.5, 0.9, f"{unan:.0%} unanimous on pass/fail\n{1-unan:.0%} contested", transform=ax.transAxes,
            color=FG, ha="center", fontsize=9)
    stats["passfail_unanimous"] = unan
    save(fig, "05_passfail_contested.png")

    # Fig 6: within-video RANK agreement (does the ordering agree even if absolute scores don't?)
    # for videos with >=2 candidates, avg pairwise Spearman between raters who scored all its candidates
    from itertools import combinations
    wide = df.pivot_table(index="annotator_id", columns="candidate_id", values="score", aggfunc="first")
    vid_of = dict(zip(df.candidate_id, df.video_id))
    cands_by_vid = defaultdict(list)
    for cid in byc.index:
        cands_by_vid[vid_of[cid]].append(cid)
    rho_all = []
    for vid, cs in cands_by_vid.items():
        if len(cs) < 2:
            continue
        sub = wide[cs].dropna(how="any")
        for r1, r2 in combinations(sub.index, 2):
            a, bb = sub.loc[r1].values.astype(float), sub.loc[r2].values.astype(float)
            if np.std(a) == 0 or np.std(bb) == 0:
                continue
            rho_all.append(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(bb)))[0, 1])
    rho_all = [r for r in rho_all if not np.isnan(r)]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    if rho_all:
        ax.hist(rho_all, bins=np.linspace(-1, 1, 21), color=OK, edgecolor=BG)
        ax.axvline(np.mean(rho_all), color=WARN, ls="--", lw=1)
        ax.text(0.02, 0.9, f"mean ρ = {np.mean(rho_all):.2f}\n(n={len(rho_all)} rater pairs,\nvideos with ≥2 moments)",
                transform=ax.transAxes, color=FG, fontsize=9)
    style(ax, "Within-video RANK agreement between raters", xlabel="pairwise Spearman ρ", ylabel="rater pairs")
    stats["within_video_rank_rho_mean"] = float(np.mean(rho_all)) if rho_all else None
    save(fig, "06_rank_agreement.png")

    # Fig 7: consensus score distribution
    med = byc.apply(np.median)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    vc = med.value_counts().sort_index()
    ax.bar([str(x) for x in vc.index], vc.values, color=AC)
    style(ax, "Consensus (median) score distribution", xlabel="median score", ylabel="candidates")
    save(fig, "07_consensus_dist.png")

    # Fig 8: per-creator score distribution + n
    def stacked_by(col, fname, title, top=None):
        g = df.copy(); g["grp"] = g[col].map(short_creator) if col == "creator" else g[col]
        order = g.grp.value_counts().index if top is None else g.grp.value_counts().index[:top]
        fig, ax = plt.subplots(figsize=(max(6, len(order)*0.9), 3.8))
        bottom = np.zeros(len(order))
        colors = ["#555", "#8a6d3b", AC, OK]
        for sc in range(4):
            frac = [np.mean(g[g.grp == o].score == sc) for o in order]
            ax.bar(range(len(order)), frac, bottom=bottom, label=f"score {sc}", color=colors[sc])
            bottom += frac
        ns = [int((g.grp == o).sum()) for o in order]
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([f"{o}\nn={n}" for o, n in zip(order, ns)], rotation=30, ha="right", fontsize=7)
        style(ax, title, ylabel="share of scores")
        ax.legend(facecolor=BG, labelcolor=FG, edgecolor="#333", fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18))
        save(fig, fname)
    stacked_by("creator", "08_by_creator.png", "Score distribution by creator")
    stacked_by("domain", "09_by_domain.png", "Score distribution by domain (category)")

    # Fig 10: per-annotator score distribution (leniency / calibration)
    g = df.copy(); order = g.annotator_id.value_counts().index
    fig, ax = plt.subplots(figsize=(max(6, len(order)*0.9), 3.8))
    bottom = np.zeros(len(order)); colors = ["#555", "#8a6d3b", AC, OK]
    for sc in range(4):
        frac = [np.mean(g[g.annotator_id == o].score == sc) for o in order]
        ax.bar(range(len(order)), frac, bottom=bottom, label=f"score {sc}", color=colors[sc]); bottom += frac
    ns = [int((g.annotator_id == o).sum()) for o in order]
    ax.set_xticks(range(len(order))); ax.set_xticklabels([f"{o[:8]}\nn={n}" for o, n in zip(order, ns)], rotation=30, ha="right", fontsize=7)
    style(ax, "Score distribution by annotator (leniency differs)", ylabel="share of scores")
    ax.legend(facecolor=BG, labelcolor=FG, edgecolor="#333", fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    save(fig, "10_by_annotator.png")

    # Fig 11: agreement (alpha) per domain
    fig, ax = plt.subplots(figsize=(6, 3.4))
    doms = df.domain.dropna().unique()
    da = []
    for d in doms:
        sub = df[df.domain == d]
        piv_d = sub.pivot_table(index="annotator_id", columns="candidate_id", values="score", aggfunc="first")
        try:
            da.append((d, float(krippendorff.alpha(reliability_data=piv_d.values, level_of_measurement="ordinal"))))
        except Exception:
            da.append((d, float("nan")))
    da.sort(key=lambda x: -(x[1] if x[1] == x[1] else -9))
    ax.bar([d for d, _ in da], [a for _, a in da], color=AC)
    for i, (_, a) in enumerate(da):
        if a == a: ax.text(i, a+.01, f"{a:.2f}", ha="center", color=FG, fontsize=9)
    ax.set_xticklabels([d[:14] for d, _ in da], rotation=30, ha="right", fontsize=7)
    style(ax, "Ordinal α by domain (subjectivity varies by content)", ylabel="Krippendorff α")
    stats["alpha_by_domain"] = {d: a for d, a in da}
    save(fig, "11_alpha_by_domain.png")

    # Fig 12: cue-type distribution among >=2 scores
    cue = df[df.score >= 2].cue_type.fillna("untagged").value_counts()
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.bar(cue.index.astype(str), cue.values, color=AC)
    style(ax, "Cue type on ≥2 scores (visual / audio / both)", ylabel="scores")
    stats["cue_counts"] = {str(k): int(v) for k, v in cue.items()}
    save(fig, "12_cue_type.png")

    (OUT / "stats.json").write_text(json.dumps(stats, indent=2))
    print("wrote figures + stats.json to", OUT)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    import argparse
    _ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _ap.add_argument("--out", type=Path, default=OUT, help="figure + stats.json directory")
    OUT = _ap.parse_args().out
    main()
