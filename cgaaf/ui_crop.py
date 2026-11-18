"""Remove application UI (Blender / 3ds Max windows) from student screenshots.

detect_viewport(img) -> (x0, y0, x1, y1) or None
    Recursive XY-cut along thin dark separator lines that frame the editor
    areas of the Blender/3ds Max user interface; among the leaf regions the
    largest textured one whose dominant colour is not the dark, unsaturated
    grey of UI panels is the 3D viewport (falls back to the largest textured one). Uniform header/menu strips at the top and
    bottom of that region are trimmed. Returns None for clean renders (no
    separators), so it is safe to apply to every image.
mock_ui(render, layout) -> screenshot-like image used by tests and the benchmark.
A manual crop stored for a submission always overrides the automatic one.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

MIN_SEG = 24


def _lum(a):
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def _sep_runs(Y, axis):
    """Thin (1-6 px) uniform lines darker than their surroundings: area separators of the UI."""
    L = Y if axis == 0 else Y.T
    n = L.shape[0]
    med = np.median(L, axis=1)
    uni = (np.abs(L - med[:, None]) < 6).mean(1) >= 0.9
    idx = np.arange(n)
    lo, hi = med[np.clip(idx - 8, 0, n - 1)], med[np.clip(idx + 8, 0, n - 1)]
    edge = (idx < 2) | (idx > n - 3)
    nb = np.minimum(lo, hi)
    # light themes (v1.4): a thin uniform line whose grey differs clearly from the rows 4 px on either side
    m_lo, m_hi = med[np.clip(idx - 4, 0, n - 1)], med[np.clip(idx + 4, 0, n - 1)]
    distinct = (np.abs(med - m_lo) >= 25) & (np.abs(med - m_hi) >= 25)
    cand = uni & (((med < 36) & ((med <= nb - 5) | edge)) | distinct)
    runs, i = [], 0
    while i < n:
        if cand[i]:
            j = i
            while j + 1 < n and cand[j + 1]:
                j += 1
            if j - i + 1 <= 6:
                runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


def _segments(n, runs):
    segs, start = [], 0
    for a, b in runs:
        if a - start >= MIN_SEG:
            segs.append((start, a))
        start = b + 1
    if n - start >= MIN_SEG:
        segs.append((start, n))
    return segs


def _xycut(Y, x0, y0, x1, y1, depth=0, out=None):
    """Classic recursive XY-cut: split on rows first, else on columns."""
    out = [] if out is None else out
    sub = Y[y0:y1, x0:x1]
    h, w = sub.shape
    if depth < 10:
        rs = _segments(h, _sep_runs(sub, 0))
        if len(rs) > 1 or (rs and rs[0] != (0, h)):
            for ra, rb in rs:
                _xycut(Y, x0, y0 + ra, x1, y0 + rb, depth + 1, out)
            return out
        cs = _segments(w, _sep_runs(sub, 1))
        if len(cs) > 1 or (cs and cs[0] != (0, w)):
            for ca, cb in cs:
                _xycut(Y, x0 + ca, y0, x0 + cb, y1, depth + 1, out)
            return out
    out.append((x0, y0, x1, y1))
    return out


def _band(med, sat, rev=False):
    """Length of a uniform, unsaturated UI band (header/footer) at one end, else 0."""
    m, s = (med[::-1], sat[::-1]) if rev else (med, sat)
    k = 0
    while k < len(m) and abs(m[k] - m[0]) <= 4 and s[k] < 12:
        k += 1
    if 10 <= k <= 40 and k < len(m) and abs(m[k] - m[0]) > 8:
        return k
    return 0


def _trim_strips(a, Y, box):
    x0, y0, x1, y1 = box
    R, A = Y[y0:y1, x0:x1], a[y0:y1, x0:x1]
    med = np.median(R, axis=1)
    sat = np.median(A.max(2) - A.min(2), axis=1)
    t, b = _band(med, sat), _band(med, sat, rev=True)
    return (x0, y0 + t, x1, y1 - b)


def detect_viewport(img):
    a = np.asarray(img, dtype=np.float64)
    Y = _lum(a)
    H, W = Y.shape
    leaves = _xycut(Y, 0, 0, W, H)
    if len(leaves) <= 1 and leaves[0] == (0, 0, W, H):
        return None
    best, bs = None, -1
    for (x0, y0, x1, y1) in leaves:
        area = (x1 - x0) * (y1 - y0)
        if area < 0.08 * W * H:
            continue
        sub = a[y0:y1, x0:x1]
        tex = float(_lum(sub).std())
        sat = float((sub.max(2) - sub.min(2)).mean())
        med = np.median(sub.reshape(-1, 3), axis=0)
        q = (sub.astype(int) // 4).reshape(-1, 3)
        _, cnt = np.unique(q[:, 0] * 4096 + q[:, 1] * 64 + q[:, 2], return_counts=True)
        top3 = float(np.sort(cnt)[::-1][:3].sum() / len(q))
        # an interface panel (dark or light theme) is unsaturated and drawn in a few flat colours; a
        # solid-shaded grey viewport is unsaturated too, but shaded, so it is not mistaken for a panel
        ui_like = (med.max() - med.min()) < 10 and top3 >= 0.97
        s = area * (1.0 if tex > 6 else 0.2) * (1 + sat / 40) * (0.15 if ui_like else 1.0)
        if s > bs:
            best, bs = (x0, y0, x1, y1), s
    if best is None:
        return None
    best = _trim_strips(a, Y, best)
    if (best[2] - best[0]) * (best[3] - best[1]) > 0.97 * W * H:
        return None
    return tuple(int(v) for v in best)


def apply_crop(a: np.ndarray, crop):
    """crop: None | 'auto' | (x0, y0, x1, y1) in pixels | dict with fractions {x0,y0,x1,y1}."""
    if crop is None or crop is False:
        return a, None
    H, W = a.shape[:2]
    if crop == "auto":
        box = detect_viewport(a)
        if box is None:
            return a, None
    elif isinstance(crop, dict):
        box = (int(crop["x0"] * W), int(crop["y0"] * H), int(crop["x1"] * W), int(crop["y1"] * H))
    else:
        box = tuple(int(v) for v in crop)
    x0, y0, x1, y1 = max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3])
    if x1 - x0 < 16 or y1 - y0 < 16:
        return a, None
    return a[y0:y1, x0:x1], (x0, y0, x1, y1)


# ------------------------------------------------------------------ mock UI for tests / benchmark
THEMES = {"dark": dict(SEP=(22, 22, 22), AREA=(48, 48, 48), PANEL=(61, 61, 61), HEAD=(40, 40, 40), TEXT=(190, 190, 190), TEXT2=(120, 120, 120), BTN=(130, 130, 130)),
          "light": dict(SEP=(140, 140, 140), AREA=(205, 205, 205), PANEL=(220, 220, 220), HEAD=(188, 188, 188), TEXT=(60, 60, 60), TEXT2=(120, 120, 120), BTN=(150, 150, 150))}


def mock_ui(render: Image.Image, layout=0, seed=0, theme="dark", return_box=False):
    """Blender-like UI (dark or light theme) around a viewport showing `render`.
    return_box=True also returns the true viewport rectangle (x0, y0, x1, y1) without its header."""
    rng = np.random.default_rng(seed)
    W, H = render.size
    T = THEMES[theme]
    SEP, AREA, PANEL, HEAD, TEXT = T["SEP"], T["AREA"], T["PANEL"], T["HEAD"], T["TEXT"]
    left = [0, 46, 70][layout % 3]
    right = [300, 360, 260][layout % 3]
    top = [34, 30, 60][layout % 3]
    bottom = [0, 140, 90][layout % 3]
    head = 26
    sw, sh = W + left + right + 6, H + top + bottom + head + 6
    im = Image.new("RGB", (sw, sh), SEP)
    d = ImageDraw.Draw(im)

    def area(x0, y0, x1, y1, col, text=True):
        d.rectangle([x0, y0, x1 - 1, y1 - 1], fill=col)
        if text:
            for yy in range(y0 + 8, y1 - 14, 22):
                L = int(rng.integers(30, max(31, (x1 - x0) - 20)))
                d.rectangle([x0 + 10, yy, x0 + 10 + L, yy + 8], fill=TEXT if rng.random() < .5 else T["TEXT2"])
    area(0, 0, sw, top, HEAD)
    vx0, vy0 = left + 2, top + 2
    if left:
        area(0, top + 2, left, sh, PANEL)
    area(vx0 + W + 2, top + 2, sw, top + 2 + (H + head) // 2, AREA)
    area(vx0 + W + 2, top + 4 + (H + head) // 2, sw, sh, PANEL)
    if bottom:
        area(vx0, vy0 + head + H + 2, vx0 + W, sh, AREA)
    d.rectangle([vx0, vy0, vx0 + W - 1, vy0 + head - 1], fill=HEAD)
    for k in range(8):
        d.rectangle([vx0 + 8 + 26 * k, vy0 + 6, vx0 + 22 + 26 * k, vy0 + 20], fill=T["BTN"])
    im.paste(render, (vx0, vy0 + head))
    box = (vx0, vy0 + head, vx0 + W, vy0 + head + H)
    return (im, box) if return_box else im


def box_iou(a, b):
    if a is None or b is None:
        return 0.0
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0
