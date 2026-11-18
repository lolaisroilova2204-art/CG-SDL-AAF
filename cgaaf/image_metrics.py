"""Perceptual image similarity between a student render and a reference render.

Pipeline: load -> normalise (background estimate, optional trim, aspect-preserving
fit into N x N) -> foreground weight mask -> SSIM / MS-SSIM (weighted),
LPIPS (optional, AlexNet), colour fidelity (HSV-histogram Bhattacharyya +
CIEDE2000 on block means), edge F1 (Sobel, 2-px tolerance).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.color import deltaE_ciede2000, rgb2lab

N_DEFAULT = 256
_C1, _C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
_MS_W = np.array([0.0448, 0.2856, 0.3001, 0.2363])
_MS_W = _MS_W / _MS_W.sum()


# ----------------------------------------------------------------- loading
def load_image(path) -> np.ndarray:
    """RGB float64 array in [0, 255]; transparency composited on white."""
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    return np.asarray(im.convert("RGB"), dtype=np.float64)


def _corner_bg(a: np.ndarray, k: int = 4) -> np.ndarray:
    """Background colour: median of the four k x k corner patches (v1.4: medians instead of means, so
    that a stray interface row left by an inexact crop does not shift the estimate)."""
    patches = [a[:k, :k], a[:k, -k:], a[-k:, :k], a[-k:, -k:]]
    return np.median(np.stack([np.median(p.reshape(-1, 3), axis=0) for p in patches]), axis=0)


def normalise_geometry(img: np.ndarray, trim: bool = True, n: int = N_DEFAULT):
    """Geometry of `normalise`: returns (scale0, (y0, y1, x0, x1), s, (offx, offy)) so that an original
    pixel (x, y) maps to ((x * scale0 - x0) * s + offx, (y * scale0 - y0) * s + offy)."""
    a = img
    scale0 = min(1.0, 1200 / max(a.shape[:2]))
    if scale0 < 1:
        a = np.asarray(Image.fromarray(a.astype(np.uint8)).resize(
            (max(1, round(a.shape[1] * scale0)), max(1, round(a.shape[0] * scale0))), Image.LANCZOS), dtype=np.float64)
    bg = _corner_bg(a)
    h, w, _ = a.shape
    y0, y1, x0, x1 = 0, h, 0, w
    if trim:
        diff = np.abs(a - bg).sum(2) > 36
        diff[:2, :] = diff[-2:, :] = False  # v1.4: a 1-2 px line left at the border (e.g. by an inexact
        diff[:, :2] = diff[:, -2:] = False  # interface crop) must not stretch the object's bounding box
        ys, xs = np.where(diff)
        if len(ys):
            m = int(round(0.05 * max(ys.max() - ys.min(), xs.max() - xs.min())))
            y0, y1 = max(0, ys.min() - m), min(h, ys.max() + 1 + m)
            x0, x1 = max(0, xs.min() - m), min(w, xs.max() + 1 + m)
    s = min(n / (x1 - x0), n / (y1 - y0))
    dw, dh = max(1, round((x1 - x0) * s)), max(1, round((y1 - y0) * s))
    return a, bg, scale0, (y0, y1, x0, x1), s, ((n - dw) // 2, (n - dh) // 2), (dw, dh)


def normalise(img: np.ndarray, trim: bool = True, n: int = N_DEFAULT) -> np.ndarray:
    """Estimate background from corners, optionally trim to object bbox (+5 %),
    then fit into an n x n canvas filled with the background colour."""
    a, bg, _, (y0, y1, x0, x1), _, (ox, oy), (dw, dh) = normalise_geometry(img, trim, n)
    crop = Image.fromarray(a[y0:y1, x0:x1].astype(np.uint8)).resize((dw, dh), Image.LANCZOS)
    canvas = Image.new("RGB", (n, n), tuple(int(round(v)) for v in bg))
    canvas.paste(crop, (ox, oy))
    return np.asarray(canvas, dtype=np.float64)


def map_point(img: np.ndarray, x: float, y: float, trim: bool = True, n: int = N_DEFAULT):
    """Position of the original pixel (x, y) in the normalised n x n canvas."""
    _, _, s0, (y0, _, x0, _), s, (ox, oy), _ = normalise_geometry(img, trim, n)
    return (x * s0 - x0) * s + ox, (y * s0 - y0) * s + oy


def luminance(a: np.ndarray) -> np.ndarray:
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


# ----------------------------------------------------------------- SSIM family
def foreground_weight(ref: np.ndarray, stu: np.ndarray, bg_w: float = 0.1):
    bg = _corner_bg(ref)
    fg = (np.abs(ref - bg).sum(2) > 30) | (np.abs(stu - bg).sum(2) > 30)
    r = max(1, ref.shape[0] // 32)
    fg = ndimage.binary_dilation(fg, structure=np.ones((2 * r + 1, 2 * r + 1), bool))
    return np.where(fg, 1.0, bg_w), float(fg.mean())


def _gauss(x):  # 11x11 window, sigma 1.5 (Wang et al. 2004)
    return ndimage.gaussian_filter(x, sigma=1.5, truncate=3.5, mode="nearest")


def ssim_components(a, b, w=None):
    mu1, mu2 = _gauss(a), _gauss(b)
    s11 = np.maximum(_gauss(a * a) - mu1 ** 2, 0)
    s22 = np.maximum(_gauss(b * b) - mu2 ** 2, 0)
    s12 = _gauss(a * b) - mu1 * mu2
    l = (2 * mu1 * mu2 + _C1) / (mu1 ** 2 + mu2 ** 2 + _C1)
    cs = (2 * s12 + _C2) / (s11 + s22 + _C2)
    w = np.ones_like(a) if w is None else w
    sw = w.sum()
    return {"ssim": float((w * l * cs).sum() / sw), "l": float((w * l).sum() / sw),
            "cs": float((w * cs).sum() / sw), "map": l * cs}


def _down2(x):
    h, w = (x.shape[0] // 2) * 2, (x.shape[1] // 2) * 2
    x = x[:h, :w]
    return 0.25 * (x[0::2, 0::2] + x[1::2, 0::2] + x[0::2, 1::2] + x[1::2, 1::2])


def ms_ssim(a, b, w=None):
    prod, first = 1.0, None
    for j, wj in enumerate(_MS_W):
        r = ssim_components(a, b, w)
        first = first or r
        prod *= max(r["cs"], 1e-4) ** wj
        if j == len(_MS_W) - 1:
            prod *= max(r["l"], 1e-4) ** wj
        else:
            a, b = _down2(a), _down2(b)
            w = None if w is None else _down2(w)
    return prod, first


# ----------------------------------------------------------------- colour & edges
def hsv_hist(a: np.ndarray) -> np.ndarray:
    x = a.reshape(-1, 3) / 255.0
    mx, mn = x.max(1), x.min(1)
    v, d = mx, mx - mn
    s = np.where(mx > 0, d / np.maximum(mx, 1e-12), 0)
    r, g, b = x.T
    hh = np.zeros_like(mx)
    m = d > 0
    rm, gm = m & (mx == r), m & (mx == g) & (mx != r)
    bm = m & ~rm & ~gm
    hh[rm] = ((g - b)[rm] / d[rm]) % 6
    hh[gm] = (b - r)[gm] / d[gm] + 2
    hh[bm] = (r - g)[bm] / d[bm] + 4
    hh = hh * 60 % 360
    grey = (s < 0.12) | (v < 0.08)
    hb = np.minimum(17, (hh / 20).astype(int))
    sb = np.minimum(2, ((s - 0.12) / 0.88 * 3).astype(int).clip(0))
    vb = np.minimum(2, (v * 3).astype(int))
    bins = np.where(grey, 162 + np.minimum(7, (v * 8).astype(int)), hb * 9 + sb * 3 + vb)
    hist = np.bincount(bins, minlength=170).astype(float)
    return hist / hist.sum()


def block_mean(a, g=64):
    n = a.shape[0] // g
    return a[:g * n, :g * n].reshape(g, n, g, n, -1).mean((1, 3))


def colour_fidelity(ref, stu, w):
    bc = float(np.sqrt(hsv_hist(ref) * hsv_hist(stu)).sum())
    L1, L2 = rgb2lab(block_mean(ref) / 255.0), rgb2lab(block_mean(stu) / 255.0)
    de = deltaE_ciede2000(L1, L2)
    wb = block_mean(w[..., None])[..., 0]
    de_mean = float((de * wb).sum() / wb.sum())
    return {"color_hist": bc, "deltaE00": de_mean, "color": 0.5 * bc + 0.5 * float(np.exp(-de_mean / 12.0))}


def edge_map(y):
    mag = np.hypot(ndimage.sobel(y, 1), ndimage.sobel(y, 0))
    t = max(60.0, float(np.percentile(mag, 90)))
    e = mag > t
    e[[0, -1], :] = False
    e[:, [0, -1]] = False
    return e


def edge_f1(y1, y2, tol=2):
    e1, e2 = edge_map(y1), edge_map(y2)
    n1, n2 = e1.sum(), e2.sum()
    if n1 == 0 and n2 == 0:
        return 1.0, 1.0, 1.0
    if n1 == 0 or n2 == 0:
        return 0.0, 0.0, 0.0
    st = np.ones((2 * tol + 1, 2 * tol + 1), bool)
    p = (e2 & ndimage.binary_dilation(e1, st)).sum() / n2
    r = (e1 & ndimage.binary_dilation(e2, st)).sum() / n1
    return (2 * p * r / (p + r) if p + r else 0.0), float(p), float(r)


def worst_region(smap, g=3):
    n = smap.shape[0]
    best = None
    for gy in range(g):
        for gx in range(g):
            v = smap[gy * n // g:(gy + 1) * n // g, gx * n // g:(gx + 1) * n // g].mean()
            if best is None or v < best[1]:
                best = (gy * g + gx, float(v))
    return best


# ----------------------------------------------------------------- LPIPS
_LPIPS = {"model": None, "error": None}


def lpips_available() -> bool:
    if _LPIPS["model"] is not None:
        return True
    if _LPIPS["error"] is not None:
        return False
    try:
        import lpips  # noqa: F401
        import torch  # noqa: F401
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _LPIPS["model"] = lpips.LPIPS(net="alex", verbose=False).eval()
        return True
    except Exception as e:  # missing package or weights cannot be downloaded
        _LPIPS["error"] = repr(e)
        return False


def lpips_distance(ref, stu):
    if not lpips_available():
        return None
    import torch
    t = lambda a: torch.from_numpy((a / 127.5 - 1.0).transpose(2, 0, 1)[None].astype(np.float32))
    with torch.no_grad():
        return float(_LPIPS["model"](t(ref), t(stu)).item())


# ----------------------------------------------------------------- additional baselines (not part of the score)
def flip_error(R, S):
    """Mean FLIP error (Andersson et al.) between two normalised sRGB images; None if the
    flip-evaluator package is not installed. Lower = more similar."""
    try:
        import flip_evaluator as flip
    except Exception:
        return None
    _, mean, _ = flip.evaluate(np.ascontiguousarray(R, dtype=np.float32) / 255.0,
                               np.ascontiguousarray(S, dtype=np.float32) / 255.0, "LDR", applyMagma=False)
    return float(mean)


def sncd(R, S):
    """Symmetric normalised compression distance (Roman-Gonzalez 2013): mean of NCD(x, y) and NCD(y, x),
    computed with the LZMA compressor on the 8-bit luminance of the normalised images (its dictionary
    spans both images, unlike the 32 kB window of deflate). Lower = more similar."""
    import lzma
    a = np.clip(luminance(R), 0, 255).astype(np.uint8).tobytes()
    b = np.clip(luminance(S), 0, 255).astype(np.uint8).tobytes()
    C = lambda x: len(lzma.compress(x, preset=6))
    ca, cb, cab, cba = C(a), C(b), C(a + b), C(b + a)
    lo, hi = min(ca, cb), max(ca, cb)
    return float(0.5 * ((cab - lo) / hi + (cba - lo) / hi))


# ----------------------------------------------------------------- main entry
@dataclass
class ImageResult:
    raw: dict
    m: dict
    ssim_map: np.ndarray = field(repr=False)
    ref_norm: np.ndarray = field(repr=False)
    stu_norm: np.ndarray = field(repr=False)
    worst: tuple = (0, 1.0)


def compare_images(ref, stu, trim=True, use_lpips=True, n=N_DEFAULT, ref_is_normalised=False, crop=None, ref_crop=None,
                   baselines=False) -> ImageResult:
    """ref, stu: paths or RGB arrays. crop / ref_crop: None | 'auto' | (x0,y0,x1,y1) | {x0,y0,x1,y1 fractions}
    removes application UI from screenshots (see ui_crop).
    trim=True ("object" mode, default for tasks): both images are trimmed to the object's bounding box,
    so the placement and size of the object in the frame are not assessed. trim=False ("composition"
    mode): the frames are compared as they are, so framing differences count as errors.
    baselines=True adds FLIP and SNCD (benchmark baselines, not used by the score)."""
    from .ui_crop import apply_crop
    ref = load_image(ref) if not isinstance(ref, np.ndarray) else ref
    stu = load_image(stu) if not isinstance(stu, np.ndarray) else stu
    ref, rbox = apply_crop(ref, ref_crop)
    stu, sbox = apply_crop(stu, crop)
    R = ref if ref_is_normalised and ref.shape[:2] == (n, n) else normalise(ref, trim, n)
    S = normalise(stu, trim, n)
    w, fg_frac = foreground_weight(R, S)
    yR, yS = luminance(R), luminance(S)
    ms, first = ms_ssim(yR, yS, w)
    col = colour_fidelity(R, S, w)
    f1, ep, er = edge_f1(yR, yS)
    d = lpips_distance(R, S) if use_lpips else None
    mse = float(((R - S) ** 2).mean())
    psnr = 99.0 if mse <= 1e-10 else float(10 * np.log10(255.0 ** 2 / mse))
    plain = ssim_components(yR, yS)["ssim"]  # baseline: unweighted SSIM
    raw = {"ssim": first["ssim"], "msssim": ms, "lpips": d, **col, "edge_f1": f1, "edge_p": ep, "edge_r": er, "fg_frac": fg_frac,
           "psnr": psnr, "ssim_plain": plain}
    m = {"ssim": float(np.clip(first["ssim"], 0, 1)), "msssim": float(np.clip(ms, 0, 1)), "color": col["color"], "edge": f1}
    if d is not None:
        m["lpips"] = float(np.clip(1 - d / 0.6, 0, 1))
    raw["crop_box"] = list(sbox) if sbox else None
    raw["ref_crop_box"] = list(rbox) if rbox else None
    worst = worst_region(first["map"])
    raw["worst_cell"], raw["worst_ssim"] = int(worst[0]), float(worst[1])
    if baselines:
        raw["flip"] = flip_error(R, S)
        raw["sncd"] = sncd(R, S)
    return ImageResult(raw, m, first["map"], R, S, worst)


def heatmap(res: ImageResult, out_path, size=320):
    """Dissimilarity overlay: grey base + red where SSIM is low."""
    g = luminance(res.stu_norm)[..., None]
    base = 140 + 0.45 * g
    v = np.clip((1 - res.ssim_map) / 0.6, 0, 1)[..., None]
    col = np.concatenate([np.full_like(v, 230), np.where(v < 0.5, 180, 40) * np.ones_like(v), np.full_like(v, 30)], 2)
    img = (base * (1 - v) + col * v).clip(0, 255).astype(np.uint8)
    Image.fromarray(img).resize((size, size), Image.LANCZOS).save(out_path)
    return out_path
