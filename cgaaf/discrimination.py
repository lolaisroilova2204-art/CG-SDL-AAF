"""Nuisance-vs-error discrimination analysis used to compare the framework with baselines.

For every method (the framework score or a single baseline metric) a similarity value s is
oriented so that larger = more similar to the reference. A good assessment metric keeps s
high under *nuisance* variation (changes a teacher would not penalise: re-sampling noise,
resolution, compression, pose, tessellation) and lowers it under genuine *errors*.

AUC  = P(s_nuisance > s_error) (+ 0.5 for ties) — 1.0 = perfect separation, 0.5 = chance.
d    = Cohen's d between the nuisance and the error distributions (pooled SD).
CI   = percentile bootstrap that resamples *references* (clusters), not single samples.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def auc(pos, neg):
    from scipy.stats import rankdata
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return np.nan
    ranks = rankdata(np.concatenate([pos, neg]))  # mid-ranks for ties
    rp = ranks[: len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def cohen_d(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) < 2 or len(neg) < 2:
        return np.nan
    sp = np.sqrt(((len(pos) - 1) * pos.var(ddof=1) + (len(neg) - 1) * neg.var(ddof=1)) / (len(pos) + len(neg) - 2))
    return float((pos.mean() - neg.mean()) / sp) if sp > 0 else np.inf


def _oriented(df, col, sign):
    v = df[col].astype(float)
    return v * sign


def compare_methods(df, methods, kind_col="kind", cluster_col="reference", min_level=0.0, B=1000, seed=3):
    """methods: list of (column, label, sign). Rows with kind 'nuisance' vs 'error' (level >= min_level)."""
    rng = np.random.default_rng(seed)
    nu = df[df[kind_col] == "nuisance"]
    er = df[(df[kind_col] == "error") & (df["level"] >= min_level)]
    clusters = sorted(df[cluster_col].unique())
    rows = []
    for col, lab, sign in methods:
        if col not in df or df[col].isna().all():
            continue
        p, n = _oriented(nu, col, sign), _oriented(er, col, sign)
        a, d = auc(p, n), cohen_d(p, n)
        boots = []
        if len(clusters) > 1:
            pc = {c: p[nu[cluster_col] == c].to_numpy() for c in clusters}
            nc = {c: n[er[cluster_col] == c].to_numpy() for c in clusters}
            for _ in range(B):
                pick = rng.choice(clusters, len(clusters), replace=True)
                boots.append(auc(np.concatenate([pc[c] for c in pick]), np.concatenate([nc[c] for c in pick])))
        lo, hi = (np.nanpercentile(boots, 2.5), np.nanpercentile(boots, 97.5)) if boots else (np.nan, np.nan)
        rows.append({"method": lab, "AUC": a, "AUC 95% CI": f"[{lo:.3f}, {hi:.3f}]" if np.isfinite(lo) else "—", "Cohen d": d,
                     "n nuisance": int(p.notna().sum()), "n error": int(n.notna().sum())})
    return pd.DataFrame(rows).sort_values("AUC", ascending=False, ignore_index=True)


def detectability(df, methods, kind_col="kind"):
    """AUC(nuisance vs errors at one severity level) for every level and method."""
    nu = df[df[kind_col] == "nuisance"]
    out = []
    for lv in sorted(df[df[kind_col] == "error"]["level"].unique()):
        er = df[(df[kind_col] == "error") & (df["level"] == lv)]
        for col, lab, sign in methods:
            if col in df and not df[col].isna().all():
                out.append({"level": lv, "method": lab, "AUC": auc(_oriented(nu, col, sign), _oriented(er, col, sign))})
    return pd.DataFrame(out)


def min_detectable(det, thr=0.9):
    """Smallest severity at which each method reaches AUC >= thr."""
    rows = []
    for lab, g in det.groupby("method"):
        ok = g[g["AUC"] >= thr].sort_values("level")
        rows.append({"method": lab, f"min severity with AUC≥{thr}": float(ok["level"].iloc[0]) if len(ok) else np.nan})
    return pd.DataFrame(rows)
