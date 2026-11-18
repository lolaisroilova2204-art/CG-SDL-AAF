"""Nuisance-vs-error discrimination analysis used to compare the framework with baselines.

For every method (the framework score or a single baseline metric) a similarity value s is
oriented so that larger = more similar to the reference. A good assessment metric keeps s
high under *nuisance* variation (changes a teacher would not penalise: re-sampling noise,
resolution, compression, pose, tessellation) and lowers it under genuine *errors*.

AUC  = P(s_nuisance > s_error) (+ 0.5 for ties) — 1.0 = perfect separation, 0.5 = chance.
d    = Cohen's d between the nuisance and the error distributions (pooled SD).
CI   = percentile bootstrap that resamples *references* (clusters), not single samples;
       dAUC = paired difference to the framework score on the same bootstrap samples.
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


def compare_methods(df, methods, kind_col="kind", cluster_col="reference", min_level=0.0, B=1000, seed=3, ref_label="Framework score"):
    """methods: list of (column, label, sign). Rows with kind 'nuisance' vs 'error' (level >= min_level).

    For every method: AUC with a percentile-bootstrap CI that resamples reference objects (clusters);
    the paired difference to the framework score (same resampled clusters for both methods), whose CI
    tests whether the two methods differ; and the range of the AUC when one reference object is left out.
    Cohen's d is kept in the output only for continuity: under ceiling effects (nuisance scores of exactly
    100) it is not interpretable, and the paper reports AUC differences instead."""
    rng = np.random.default_rng(seed)
    nu = df[df[kind_col] == "nuisance"]
    er = df[(df[kind_col] == "error") & (df["level"] >= min_level)]
    clusters = sorted(set(nu[cluster_col]) & set(er[cluster_col]))
    meths = [(c, l, s) for c, l, s in methods if c in df and not df[c].isna().all()]
    P = {l: {c_: _oriented(nu[nu[cluster_col] == c_], c, s).to_numpy() for c_ in clusters} for c, l, s in meths}
    N = {l: {c_: _oriented(er[er[cluster_col] == c_], c, s).to_numpy() for c_ in clusters} for c, l, s in meths}
    point = {l: auc(np.concatenate(list(P[l].values())), np.concatenate(list(N[l].values()))) for _, l, _ in meths}
    boots = {l: [] for _, l, _ in meths}
    if len(clusters) > 1:
        for _ in range(B):
            pick = rng.choice(clusters, len(clusters), replace=True)
            for _, l, _ in meths:
                boots[l].append(auc(np.concatenate([P[l][c] for c in pick]), np.concatenate([N[l][c] for c in pick])))
    rows = []
    for c, l, s in meths:
        b = np.asarray(boots[l], float)
        lo, hi = (np.nanpercentile(b, 2.5), np.nanpercentile(b, 97.5)) if len(b) else (np.nan, np.nan)
        row = {"method": l, "AUC": point[l], "AUC 95% CI": f"[{lo:.3f}, {hi:.3f}]" if np.isfinite(lo) else "—", "ci_lo": lo, "ci_hi": hi}
        if ref_label in boots and len(b) and l != ref_label:
            dlt = b - np.asarray(boots[ref_label], float)
            dlo, dhi = np.nanpercentile(dlt, 2.5), np.nanpercentile(dlt, 97.5)
            row.update({"dAUC vs framework": point[l] - point.get(ref_label, np.nan), "dAUC 95% CI": f"[{dlo:+.3f}, {dhi:+.3f}]",
                        "d_lo": dlo, "d_hi": dhi, "differs": bool(dlo > 0 or dhi < 0)})
        loo = [auc(np.concatenate([P[l][k] for k in clusters if k != c_]), np.concatenate([N[l][k] for k in clusters if k != c_]))
               for c_ in clusters] if len(clusters) > 2 else []
        row.update({"LOOO min": float(np.nanmin(loo)) if loo else np.nan, "LOOO max": float(np.nanmax(loo)) if loo else np.nan,
                    "Cohen d": cohen_d(np.concatenate(list(P[l].values())), np.concatenate(list(N[l].values()))),
                    "n nuisance": int(sum(len(v) for v in P[l].values())), "n error": int(sum(len(v) for v in N[l].values())),
                    "n objects": len(clusters)})
        rows.append(row)
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
