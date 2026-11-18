"""Sensitivity of the benchmark results to the default constants (fixed a priori in v1.3).

For every perturbed constant the framework score is recomputed from the stored raw metrics of every
benchmark case, and two quantities are compared with the defaults:
  * the AUC of nuisance versus errors (severity >= 0.2), per channel / pipeline;
  * the share of cases whose five-point grade is unchanged.
Constants: the channel weights (each x0.5 and x1.5, and 200 random weight vectors drawn from a
Dirichlet distribution centred on the defaults), the linear mapping (P0, P1), the colour constant of
m_color, the floors and scales of the mesh mappings, the F-score threshold tau and the topology tolerance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .discrimination import auc
from .scoring import DEFAULT_WEIGHTS, P0, P1, grade5

IMG_K = ["ssim", "msssim", "lpips", "color", "edge"]
MESH_K = ["cd", "h95", "fscore", "topo"]


def img_m(df, c_de=12.0):
    m = pd.DataFrame({"ssim": df["ssim"].astype(float).clip(0, 1), "msssim": df["msssim"].astype(float).clip(0, 1),
                      "color": 0.5 * df["color_hist"].astype(float) + 0.5 * np.exp(-df["deltaE00"].astype(float) / c_de),
                      "edge": df["edge_f1"].astype(float)}, index=df.index)
    if "lpips" in df and df["lpips"].notna().any():
        m["lpips"] = (1 - df["lpips"].astype(float) / 0.6).clip(0, 1)
    return m


def mesh_m(df, cd_floor=0.03, cd_sigma=0.12, h_floor=0.06, h_sigma=0.30, tau=0.1, topo_frac=0.05):
    f = {0.05: "fscore_005", 0.1: "fscore", 0.15: "fscore_015"}[tau]
    bad = df["excess_boundary"] + 2 * df["excess_nonmanifold"] + df["excess_degenerate"]
    return pd.DataFrame({"cd": np.exp(-np.maximum(0, df["cd"] - cd_floor) / cd_sigma),
                         "h95": np.exp(-np.maximum(0, df["h95"] - h_floor) / h_sigma),
                         "fscore": df[f].astype(float),
                         "topo": np.clip(1 - bad / (topo_frac * df["E"].clip(lower=1)), 0, 1)}, index=df.index)


def score_from(m: pd.DataFrame, w: dict, p0=P0, p1=P1):
    ks = [k for k in m.columns if w.get(k, 0) > 0]
    W = np.array([w[k] for k in ks], float)
    P = (m[ks].to_numpy(float) * W).sum(1) / W.sum()
    return np.clip(100 * (P - p0) / (p1 - p0), 0, 100)


def _auc(df, s, min_level=0.2):
    nu = s[(df["kind"] == "nuisance").to_numpy()]
    er = s[((df["kind"] == "error") & (df["level"] >= min_level)).to_numpy()]
    return auc(nu, er)


def _variants(keys, rng, n_random=200):
    base = {k: DEFAULT_WEIGHTS[k] for k in keys}
    out = [("default", "default", base, {})]
    for k in keys:
        for f in (0.5, 1.5):
            out.append(("weights (one at a time, x0.5 / x1.5)", f"w_{k} x{f}", {**base, k: base[k] * f}, {}))
    tot = sum(base.values())
    alpha = np.array([10 * base[k] / tot for k in keys]) * len(keys)
    for i in range(n_random):
        d = rng.dirichlet(alpha)
        out.append(("weights (200 random Dirichlet draws)", f"random {i}", dict(zip(keys, d)), {}))
    for p0 in (0.30, 0.50):
        out.append(("mapping P0 (0.30 / 0.50)", f"P0={p0}", base, {"p0": p0}))
    for p1 in (0.95, 1.00):
        out.append(("mapping P1 (0.95 / 1.00)", f"P1={p1}", base, {"p1": p1}))
    return out


def _table(name, df, make_m, keys, extra, rng):
    """extra: list of (group, label, kwargs for make_m)."""
    if df is None or not len(df):
        return []
    m0 = make_m(df)
    keys = [k for k in keys if k in m0.columns]
    s0 = score_from(m0, {k: DEFAULT_WEIGHTS[k] for k in keys})
    g0 = np.array([grade5(v) for v in s0])
    a0 = _auc(df, s0)
    res = []
    for group, label, w, kw in _variants(keys, rng):
        s = score_from(m0, w, kw.get("p0", P0), kw.get("p1", P1))
        res.append((group, label, _auc(df, s), float((np.array([grade5(v) for v in s]) == g0).mean())))
    for group, label, kw in extra:
        m = make_m(df, **kw)
        s = score_from(m, {k: DEFAULT_WEIGHTS[k] for k in keys})
        res.append((group, label, _auc(df, s), float((np.array([grade5(v) for v in s]) == g0).mean())))
    T = pd.DataFrame(res, columns=["group", "variant", "AUC", "grades unchanged"])
    rows = []
    for group, g in T.groupby("group", sort=False):
        rows.append({"channel": name, "constants": group, "n variants": len(g), "AUC default": a0, "AUC min": g.AUC.min(),
                     "AUC max": g.AUC.max(), "max |dAUC|": float(np.abs(g.AUC - a0).max()),
                     "grades unchanged (min)": g["grades unchanged"].min(), "grades unchanged (median)": g["grades unchanged"].median()})
    return rows


def sensitivity_tables(img: pd.DataFrame | None, mesh: pd.DataFrame | None, rend: pd.DataFrame | None, seed=5):
    rng = np.random.default_rng(seed)
    rows = []
    img_extra = [("colour constant of m_color (6 / 24)", f"c={c}", {"c_de": c}) for c in (6.0, 24.0)]
    if img is not None and len(img):
        for mode, g in img.groupby("mode"):
            rows += _table(f"image ({mode} mode)", g.reset_index(drop=True), img_m, IMG_K, img_extra, rng)
    if rend is not None and len(rend):
        for pipe, g in rend.groupby("pipeline"):
            rows += _table(f"scene ({pipe})", g.reset_index(drop=True), img_m, IMG_K, img_extra, rng)
    if mesh is not None and len(mesh):
        mx = []
        for k, v in (("cd_floor", 0.03), ("cd_sigma", 0.12), ("h_floor", 0.06), ("h_sigma", 0.30)):
            mx += [(f"mesh mapping {k} (x0.5 / x1.5)", f"{k} x{f}", {k: v * f}) for f in (0.5, 1.5)]
        mx += [("F-score threshold tau (0.05 / 0.15)", f"tau={t}", {"tau": t}) for t in (0.05, 0.15)]
        mx += [("topology tolerance (2.5 % / 10 % of edges)", f"frac={f}", {"topo_frac": f}) for f in (0.025, 0.10)]
        rows += _table("mesh", mesh.reset_index(drop=True), mesh_m, MESH_K, mx, rng)
    return {"sensitivity_constants": pd.DataFrame(rows)}
