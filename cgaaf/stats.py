"""Validation statistics for automated vs expert scores."""
from __future__ import annotations

import numpy as np
from scipy import stats as ss

from .scoring import grade5


def icc(M):
    """M: n subjects x k raters -> ICC(2,1) absolute agreement, ICC(3,1) consistency (Shrout & Fleiss)."""
    M = np.asarray(M, float)
    n, k = M.shape
    g = M.mean()
    rm, cm = M.mean(1), M.mean(0)
    msr = k * ((rm - g) ** 2).sum() / (n - 1)
    msc = n * ((cm - g) ** 2).sum() / (k - 1)
    mse = ((M - rm[:, None] - cm[None, :] + g) ** 2).sum() / ((n - 1) * (k - 1))
    return {"icc21": (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n), "icc31": (msr - mse) / (msr + (k - 1) * mse)}


def qwk(a, b, cats=(2, 3, 4, 5)):
    K = len(cats)
    idx = {c: i for i, c in enumerate(cats)}
    O = np.zeros((K, K))
    for x, y in zip(a, b):
        O[idx[x], idx[y]] += 1
    W = np.array([[(i - j) ** 2 / (K - 1) ** 2 for j in range(K)] for i in range(K)])
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    den = (W * E).sum()
    return (1 - (W * O).sum() / den if den else 1.0), O


def bootstrap_ci(fn, x, y, B=2000, seed=1, clusters=None):
    """Percentile bootstrap; with `clusters` (e.g. student ids) whole clusters are resampled,
    so repeated observations of one student are not treated as independent."""
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x), np.asarray(y)
    groups = None
    if clusters is not None:
        cl = np.asarray(clusters)
        groups = [np.where(cl == u)[0] for u in np.unique(cl)]
    v = []
    for _ in range(B):
        if groups is None:
            i = rng.integers(0, len(x), len(x))
        else:
            i = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
        try:
            v.append(fn(x[i], y[i]))
        except Exception:
            pass
    v = np.array([q for q in v if np.isfinite(q)])
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) else (np.nan, np.nan)


def agreement(auto, expert, boot=True, clusters=None):
    a, e = np.asarray(auto, float), np.asarray(expert, float)
    n = len(a)
    pr = ss.pearsonr(a, e)
    ci = pr.confidence_interval(0.95)
    sp = ss.spearmanr(a, e)
    ic = icc(np.c_[a, e])
    d = a - e
    k, O = qwk([grade5(x) for x in e], [grade5(x) for x in a])
    out = {"n": n, "pearson_r": float(pr.statistic), "pearson_lo": float(ci.low), "pearson_hi": float(ci.high), "pearson_p": float(pr.pvalue),
           "spearman_rho": float(sp.statistic), "spearman_p": float(sp.pvalue), "mae": float(np.abs(d).mean()), "rmse": float(np.sqrt((d ** 2).mean())),
           "icc21": float(ic["icc21"]), "icc31": float(ic["icc31"]), "ba_bias": float(d.mean()), "ba_lo": float(d.mean() - 1.96 * d.std(ddof=1)),
           "ba_hi": float(d.mean() + 1.96 * d.std(ddof=1)), "qwk": float(k), "exact_grade": float(np.trace(O) / O.sum()), "confusion": O.astype(int).tolist()}
    if boot and n >= 8:
        out["icc21_ci"] = bootstrap_ci(lambda x, y: icc(np.c_[x, y])["icc21"], a, e, clusters=clusters)
        out["mae_ci"] = bootstrap_ci(lambda x, y: float(np.abs(x - y).mean()), a, e, clusters=clusters)
        out["r_ci_boot"] = bootstrap_ci(lambda x, y: float(np.corrcoef(x, y)[0, 1]), a, e, clusters=clusters)
        out["bootstrap"] = "cluster (student)" if clusters is not None else "observation"
    return out


def paired_gain(first, best):
    f, b = np.asarray(first, float), np.asarray(best, float)
    if len(f) < 2:
        return {"n": len(f)}
    t = ss.ttest_rel(b, f)
    d = b - f
    try:
        w = ss.wilcoxon(b, f, zero_method="zsplit")
        wp = float(w.pvalue)
    except ValueError:
        wp = float("nan")
    return {"n": len(f), "mean_first": float(f.mean()), "mean_best": float(b.mean()), "gain": float(d.mean()),
            "t": float(t.statistic), "p": float(t.pvalue), "dz": float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else float("nan"), "wilcoxon_p": wp}


def fmt_p(p):
    return "—" if p is None or not np.isfinite(p) else ("< .001" if p < 0.001 else f"{p:.3f}")
