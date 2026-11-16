"""Score model: weighted aggregate of [0,1] metrics with linear mapping, or an
expert-calibrated non-negative linear model (NNLS with free intercept)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

IMG_KEYS = ["ssim", "msssim", "lpips", "color", "edge"]
MESH_KEYS = ["cd", "h95", "fscore", "topo", "size"]  # "size" only when the task checks absolute size
DEFAULT_WEIGHTS = {"ssim": 0.15, "msssim": 0.30, "lpips": 0.25, "color": 0.15, "edge": 0.15,
                   "cd": 0.40, "h95": 0.20, "fscore": 0.30, "topo": 0.10, "size": 0.20}
P0, P1 = 0.40, 0.98


def keys_for(task_type):
    return IMG_KEYS if task_type == "image" else MESH_KEYS if task_type == "mesh" else IMG_KEYS + MESH_KEYS


def _part(m, keys, w):
    ks = [k for k in keys if k in m and m[k] is not None and w.get(k, 0) > 0]  # missing metric (e.g. LPIPS) -> renormalise
    tot = sum(w[k] for k in ks)
    return sum(w[k] * m[k] for k in ks) / tot if tot else 0.0


def composite(task: dict, m: dict) -> float:
    """Unclipped weighted aggregate P in [0, 1] (before the linear mapping to the 0-100 score)."""
    w = {**DEFAULT_WEIGHTS, **(task.get("weights") or {})}
    t = task.get("type", "image")
    if t == "image":
        return _part(m, IMG_KEYS, w)
    if t == "mesh":
        return _part(m, MESH_KEYS, w)
    a = task.get("alpha", 0.5)
    return a * _part(m, IMG_KEYS, w) + (1 - a) * _part(m, MESH_KEYS, w)


def score(task: dict, m: dict) -> float:
    cal = task.get("calibration")
    if cal and cal.get("b"):
        s = cal["b0"] + sum(b * m.get(k, 0.0) for k, b in cal["b"].items())
        return float(np.clip(s, 0, 100))
    w = {**DEFAULT_WEIGHTS, **(task.get("weights") or {})}
    t = task.get("type", "image")
    if t == "image":
        P = _part(m, IMG_KEYS, w)
    elif t == "mesh":
        P = _part(m, MESH_KEYS, w)
    else:
        a = task.get("alpha", 0.5)
        P = a * _part(m, IMG_KEYS, w) + (1 - a) * _part(m, MESH_KEYS, w)
    p0, p1 = task.get("p0", P0), task.get("p1", P1)
    return float(np.clip(100 * (P - p0) / (p1 - p0), 0, 100))


def grade5(s):
    return 5 if s >= 86 else 4 if s >= 71 else 3 if s >= 55 else 2


def fit_nnls(X, y):
    X, y = np.asarray(X, float), np.asarray(y, float)
    xm, ym = X.mean(0), y.mean()
    b, _ = nnls(X - xm, y - ym)
    return float(ym - xm @ b), b


def calibrate(M: list[dict], y, keys):
    """M: list of metric dicts, y: expert scores. Returns coefficients and LOOCV."""
    keys = [k for k in keys if all(k in m and m[k] is not None for m in M)]
    X = np.array([[m[k] for k in keys] for m in M], float)
    y = np.asarray(y, float)
    b0, b = fit_nnls(X, y)
    pred = np.clip(b0 + X @ b, 0, 100)
    loo = np.empty(len(y))
    for i in range(len(y)):
        msk = np.arange(len(y)) != i
        c0, c = fit_nnls(X[msk], y[msk])
        loo[i] = np.clip(c0 + X[i] @ c, 0, 100)
    r = lambda a, c: float(np.corrcoef(a, c)[0, 1]) if np.std(a) > 0 and np.std(c) > 0 else float("nan")
    return {"b0": b0, "b": dict(zip(keys, map(float, b))), "keys": keys, "n": int(len(y)),
            "r_in": r(pred, y), "mae_in": float(np.abs(pred - y).mean()),
            "r_loo": r(loo, y), "mae_loo": float(np.abs(loo - y).mean()), "loo_pred": loo.tolist()}
