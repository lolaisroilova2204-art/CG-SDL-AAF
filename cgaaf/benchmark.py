"""Controlled-perturbation benchmark (technical validation, no student data).

A. Image channel : reference images x nuisance + error families x severity levels, in two modes
                   object mode      (trim on, the default of a task): placement and size of the object
                                    in the frame are not assessed -> reframing is a nuisance
                   composition mode (trim off): the frame is assessed as it is -> translation, rotation
                                    and down-scaling are errors
B. Mesh channel  : reference meshes (primitives + Blender assets) x 5 nuisance + 5 error families
C. Blender       : 12 compositions (6 development + 6 held-out), re-render vs render- and viewport-
                   screenshot pipelines, student render settings, re-tessellation, 512 px check (bench_render.py)
D. Robustness    : application UI in screenshots (dark / light theme, crop IoU), false crops on clean images,
                   auxiliary floors and thin parts (helper rule of v1.3 vs v1.4)
E. Discrimination: AUC of nuisance vs error with cluster-bootstrap CIs, paired AUC differences against the
                   framework score, leave-one-object-out range, development vs held-out objects
F. Feedback      : whether the rule-based messages name the injected error; localisation of occlusions
G. Sensitivity   : AUC and grades under perturbed default constants (sensitivity.py)
All results are written to CSV / XLSX, figures and benchmark_report_en.md.
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh
from PIL import Image, ImageEnhance, ImageFilter
from scipy.stats import spearmanr

from . import ALGO_VERSION
from .feedback import feedback_keys
from .image_metrics import compare_images, load_image, map_point
from .mesh_metrics import compare_meshes, load_mesh, prepare_reference, prepare_reference_raw, topology, torus
from .scoring import composite, score

LEVELS = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
NUIS_LEVELS = [1 / 3, 2 / 3, 1.0]
IMG_NUISANCE = ["n_jpeg", "n_noise", "n_shift", "n_resample"]   # changes a teacher would not penalise
IMG_ERR_COMP = ["translation", "rotation", "scale", "hue", "brightness", "blur", "noise", "jpeg", "occlusion", "greyscale"]
IMG_ERR_OBJ = ["rotation", "hue", "brightness", "blur", "noise", "jpeg", "occlusion", "greyscale"]
IMG_MODES = {"object": (True, IMG_NUISANCE + ["n_reframe"], IMG_ERR_OBJ),
             "composition": (False, IMG_NUISANCE, IMG_ERR_COMP)}
IMG_FAMILIES = IMG_ERR_COMP  # all image error families
MESH_INVARIANT = ["rigid_rotation", "uniform_scale", "retriangulation", "micro_noise"]
MESH_ERRORS = ["anisotropic_scale", "vertex_noise", "hole", "part_removal", "extra_part"]
STOCHASTIC_IMG = ("noise", "occlusion", "n_noise")
# (column, label, sign): sign orients the value so that larger = more similar (for AUC)
IMG_METHODS = [("score", "Framework score", 1), ("composite", "Framework composite P (unclipped)", 1), ("psnr", "PSNR", 1),
               ("ssim_plain", "SSIM (unweighted)", 1), ("ssim", "SSIM (fg-weighted)", 1), ("msssim", "MS-SSIM", 1), ("lpips", "LPIPS", -1),
               ("flip", "FLIP", -1), ("sncd", "SNCD", -1), ("deltaE00", "ΔE00", -1), ("color_hist", "Colour histogram", 1), ("edge_f1", "Edge F1", 1)]
MESH_METHODS = [("score", "Framework score", 1), ("composite", "Framework composite P (unclipped)", 1), ("cd_noalign", "Chamfer, no alignment", -1),
                ("cd_icp", "Chamfer, ICP only", -1), ("cd_pca", "Chamfer, PCA only", -1), ("cd", "Chamfer, PCA+ICP", -1), ("hd", "Hausdorff", -1),
                ("h95", "Hausdorff-95", -1), ("fscore", "F-score@0.1", 1), ("d2", "D2 shape distribution", -1)]
IMG_METRICS = [("score", "Score"), ("psnr", "PSNR"), ("ssim_plain", "SSIM (unweighted)"), ("ssim", "SSIM (weighted)"), ("msssim", "MS-SSIM"),
               ("lpips", "LPIPS"), ("flip", "FLIP"), ("sncd", "SNCD"), ("deltaE00", "ΔE00"), ("color_hist", "Hist. BC"), ("edge_f1", "Edge F1")]
MESH_METRICS = [("score", "Score"), ("cd_noalign", "CD (no align)"), ("cd", "CD"), ("hd", "HD"), ("h95", "H95"), ("fscore", "F-score"),
                ("d2", "D2"), ("boundary_edges", "Boundary edges")]
# messages that name the injected error (any of them counts as a hit); fixed before the run
IMG_EXPECTED = {"hue": {"fb_color"}, "greyscale": {"fb_color"}, "brightness": {"fb_color"},
                "blur": {"fb_edge_missing", "fb_shape", "fb_region"}, "noise": {"fb_edge_extra", "fb_shape", "fb_region"},
                "jpeg": {"fb_shape", "fb_edge_missing", "fb_edge_extra", "fb_region"}, "occlusion": {"fb_region", "fb_edge_missing", "fb_shape"},
                "rotation": {"fb_shape", "fb_region"}, "translation": {"fb_shape", "fb_region"}, "scale": {"fb_shape", "fb_region"}}
MESH_EXPECTED = {"anisotropic_scale": {"fb_mesh_shape", "fb_mesh_outlier"}, "vertex_noise": {"fb_mesh_outlier", "fb_mesh_shape"},
                 "hole": {"fb_open", "fb_mesh_missing"}, "part_removal": {"fb_mesh_missing", "fb_open"}, "extra_part": {"fb_mesh_extra"}}


# ------------------------------------------------------------------ image perturbations
def _bg(img):
    a = np.asarray(img, float)
    return tuple(int(v) for v in np.mean([a[:4, :4].reshape(-1, 3).mean(0), a[:4, -4:].reshape(-1, 3).mean(0),
                                          a[-4:, :4].reshape(-1, 3).mean(0), a[-4:, -4:].reshape(-1, 3).mean(0)], 0))


def perturb_image(img: Image.Image, fam: str, lv: float, rng, meta: dict | None = None) -> Image.Image:
    """meta (optional dict) receives the occluder rectangle of the occlusion family."""
    if lv == 0:
        return img.copy()
    W, H = img.size
    bg = _bg(img)
    if fam == "n_jpeg":
        b = io.BytesIO(); img.save(b, "JPEG", quality=int(round(100 - 15 * lv)))
        return Image.open(io.BytesIO(b.getvalue())).convert("RGB")
    if fam == "n_noise":
        a = np.asarray(img, float) + rng.normal(0, 3 * lv, (H, W, 3))
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if fam == "n_shift":
        d = max(1, int(round(2 * lv)))
        return img.transform(img.size, Image.AFFINE, (1, 0, -d, 0, 1, -d), fillcolor=bg)
    if fam == "n_resample":
        s_ = 1 - 0.4 * lv
        return img.resize((max(8, int(W * s_)), max(8, int(H * s_))), Image.BILINEAR).resize((W, H), Image.BICUBIC)
    if fam == "n_reframe":  # object mode nuisance: the picture is reduced and moved, the object stays fully visible
        s_ = 1 - 0.3 * lv
        d = int(round(0.1 * lv * W))
        w, h = int(W * s_), int(H * s_)
        c = Image.new("RGB", img.size, bg)
        c.paste(img.resize((w, h), Image.LANCZOS), ((W - w) // 2 + d, (H - h) // 2 + d // 2))
        return c
    if fam == "greyscale":  # colours lost (e.g. materials not assigned)
        return Image.blend(img, img.convert("L").convert("RGB"), float(lv))
    if fam == "translation":
        d = int(lv * 0.25 * W)
        return img.transform(img.size, Image.AFFINE, (1, 0, -d, 0, 1, 0), fillcolor=bg)
    if fam == "rotation":
        return img.rotate(lv * 25, resample=Image.BICUBIC, fillcolor=bg)
    if fam == "scale":
        s = 1 - 0.4 * lv
        w, h = int(W * s), int(H * s)
        c = Image.new("RGB", img.size, bg)
        c.paste(img.resize((w, h), Image.LANCZOS), ((W - w) // 2, (H - h) // 2))
        return c
    if fam == "hue":
        hsv = np.asarray(img.convert("HSV")).copy()
        hsv[..., 0] = (hsv[..., 0].astype(int) + int(lv * 128)) % 256
        return Image.fromarray(hsv, "HSV").convert("RGB")
    if fam == "brightness":
        return ImageEnhance.Brightness(img).enhance(1 - 0.6 * lv)
    if fam == "blur":
        return img.filter(ImageFilter.GaussianBlur(lv * 6))
    if fam == "noise":
        a = np.asarray(img, float) + rng.normal(0, 40 * lv, (H, W, 3))
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if fam == "jpeg":
        b = io.BytesIO()
        img.save(b, "JPEG", quality=max(2, int(95 - 93 * lv)))
        return Image.open(io.BytesIO(b.getvalue())).convert("RGB")
    if fam == "occlusion":
        a = np.asarray(img).copy()
        diff = np.abs(a.astype(int) - np.array(bg)).sum(2) > 36
        ys, xs = np.where(diff)
        cy, cx = (int(ys.mean()), int(xs.mean())) if len(ys) else (H // 2, W // 2)
        cy += int(rng.normal(0, H * 0.05)); cx += int(rng.normal(0, W * 0.05))
        s = int(lv * 0.35 * min(W, H))
        y0, y1, x0, x1 = max(0, cy - s // 2), cy + s // 2, max(0, cx - s // 2), cx + s // 2
        a[y0:y1, x0:x1] = bg
        if meta is not None:
            meta["occluder"] = (x0, y0, min(W, x1), min(H, y1))
        return Image.fromarray(a)
    raise ValueError(fam)


# ------------------------------------------------------------------ mesh perturbations
def _rand_rot(rng, angle):
    ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
    return trimesh.transformations.rotation_matrix(angle, ax)


def perturb_mesh(m: trimesh.Trimesh, fam: str, lv: float, rng) -> trimesh.Trimesh:
    m = m.copy()
    if lv == 0:
        return m
    diag = float(np.linalg.norm(m.bounds[1] - m.bounds[0]))
    if fam == "rigid_rotation":
        m.apply_transform(_rand_rot(rng, lv * np.pi)); return m
    if fam == "uniform_scale":
        m.apply_scale(1 + 3 * lv); m.apply_translation(rng.normal(0, diag, 3)); return m
    if fam == "micro_noise":
        v = m.vertices + rng.normal(0, 0.002 * lv * diag, m.vertices.shape)
        return trimesh.Trimesh(v, m.faces, process=False)
    if fam == "retriangulation":
        v, f = trimesh.remesh.subdivide(m.vertices, m.faces)
        if lv > 0.5:
            v, f = trimesh.remesh.subdivide(v, f)
        return trimesh.Trimesh(v, f, process=False)
    if fam == "anisotropic_scale":
        m.apply_transform(np.diag([1, 1, 1 - 0.6 * lv, 1])); return m
    if fam == "vertex_noise":
        v = m.vertices + rng.normal(0, 0.03 * lv * diag, m.vertices.shape)
        return trimesh.Trimesh(v, m.faces, process=False)
    if fam == "hole":
        c = m.triangles_center
        p = c[rng.integers(len(c))]
        d = np.linalg.norm(c - p, axis=1)
        keep = d > np.quantile(d, 0.25 * lv)
        return trimesh.Trimesh(m.vertices, m.faces[keep], process=False)
    if fam == "part_removal":
        c = m.triangles_center[:, 0]
        keep = c < np.quantile(c, 1 - 0.4 * lv)
        return trimesh.Trimesh(m.vertices, m.faces[keep], process=False)
    if fam == "extra_part":
        b = trimesh.creation.box(extents=[0.35 * lv * diag] * 3)
        b.apply_translation(m.bounds[1])
        return trimesh.util.concatenate([m, b])
    raise ValueError(fam)


def builtin_meshes():
    cap = trimesh.creation.capsule(height=1.2, radius=0.4, count=[32, 32])
    cyl = trimesh.creation.cylinder(radius=0.5, height=1.4, sections=48)
    ico = trimesh.creation.icosphere(subdivisions=3)
    ico.apply_transform(np.diag([1.0, 0.7, 0.5, 1]))  # ellipsoid (breaks PCA symmetry)
    return {"torus": torus(), "capsule": cap, "cylinder": cyl, "ellipsoid": ico}


def builtin_mesh_variants():
    """The same primitives built with half / double the segment counts: independently tessellated
    but correct models (nuisance 're-tessellation')."""
    def ell(k):
        ico = trimesh.creation.icosphere(subdivisions=k)
        ico.apply_transform(np.diag([1.0, 0.7, 0.5, 1]))
        return ico
    return {"torus": [("half", torus(nu=24, nv=12)), ("double", torus(nu=96, nv=48))],
            "capsule": [("half", trimesh.creation.capsule(height=1.2, radius=0.4, count=[16, 16])),
                        ("double", trimesh.creation.capsule(height=1.2, radius=0.4, count=[64, 64]))],
            "cylinder": [("half", trimesh.creation.cylinder(radius=0.5, height=1.4, sections=24)),
                         ("double", trimesh.creation.cylinder(radius=0.5, height=1.4, sections=96))],
            "ellipsoid": [("half", ell(2)), ("double", ell(4))]}


def builtin_images():
    from .demo import scene
    return {"scene_A": scene(), "scene_B": scene(dx=-70, dy=15)}


# ------------------------------------------------------------------ runners
def uniform_background(img: Image.Image, tol=6.0) -> bool:
    """True if the image border has one colour (an object on a background, e.g. a canonical render);
    full-frame pictures with gradients have no object boundary that object mode could trim to."""
    a = np.asarray(img.convert("RGB"), float)
    border = np.concatenate([a[:3].reshape(-1, 3), a[-3:].reshape(-1, 3), a[:, :3].reshape(-1, 3), a[:, -3:].reshape(-1, 3)])
    return bool(border.std(0).max() < tol)


def run_image_bench(refs: dict, repeats=3, use_lpips=True, log=print, modes=("object", "composition"), subsets=None):
    rows = []
    subsets = subsets or {}
    for name, ref in refs.items():
        R = np.asarray(ref.convert("RGB"), float)
        uni = uniform_background(ref)
        for mode in modes:
            trim, nuis_fams, err_fams = IMG_MODES[mode]
            for fam in nuis_fams + err_fams:
                if fam == "n_reframe" and not uni:
                    continue
                nuis = fam in nuis_fams
                for lv in (NUIS_LEVELS if nuis else LEVELS):
                    for rep in range(repeats if fam in STOCHASTIC_IMG else 1):
                        rng = np.random.default_rng(1000 * rep + int(lv * 100))
                        meta = {}
                        S_img = perturb_image(ref, fam, lv, rng, meta)
                        S = np.asarray(S_img, float)
                        t0 = time.perf_counter()
                        res = compare_images(R, S, trim=trim, use_lpips=use_lpips, baselines=True)
                        dt_ = time.perf_counter() - t0
                        sc = score({"type": "image"}, res.m)
                        row = {"reference": name, "subset": subsets.get(name, "development"), "mode": mode, "family": fam,
                               "kind": "nuisance" if nuis else ("identity" if lv == 0 else "error"), "level": lv, "repeat": rep, **res.raw,
                               "score": sc, "composite": composite({"type": "image"}, res.m), "seconds": dt_,
                               "feedback": ";".join(feedback_keys(image_raw=res.raw, worst=res.worst, image_score=sc))}
                        if "occluder" in meta:
                            x0, y0, x1, y1 = meta["occluder"]
                            px, py = map_point(S, (x0 + x1) / 2, (y0 + y1) / 2, trim)
                            n = res.ssim_map.shape[0]
                            row["occluder_cell"] = int(min(2, max(0, py * 3 // n)) * 3 + min(2, max(0, px * 3 // n)))
                        rows.append(row)
        log(f"image bench: {name} ✓")
    return pd.DataFrame(rows)


def run_mesh_bench(refs: dict, repeats=3, log=print, variants=None, subsets=None):
    rows = []
    variants, subsets = variants or {}, subsets or {}
    for name, m in refs.items():
        ref_pts, ref_raw, ref_topo = prepare_reference(m, "pca_icp"), prepare_reference_raw(m), topology(m)

        def add(stu, fam, kind, lv, rep, variant=""):
            t0 = time.perf_counter()
            res = compare_meshes(stu, ref_pts, "pca_icp", ref_topo=ref_topo, baselines=True, ref_raw=ref_raw)
            dt_ = time.perf_counter() - t0
            rows.append({"reference": name, "subset": subsets.get(name, "development"), "family": fam, "kind": kind, "level": lv,
                         "repeat": rep, "variant": variant, **res.raw, **{"m_" + k: v for k, v in res.m.items()},
                         "score": score({"type": "mesh"}, res.m), "composite": composite({"type": "mesh"}, res.m), "seconds": dt_,
                         "feedback": ";".join(feedback_keys(mesh_raw=res.raw, mesh_m=res.m, mesh_score=score({"type": "mesh"}, res.m)))})

        for fam in MESH_INVARIANT + MESH_ERRORS:
            levels = [0.0, 0.5, 1.0] if fam == "retriangulation" else LEVELS
            for lv in levels:
                for rep in range(repeats):
                    rng = np.random.default_rng(7919 * rep + int(lv * 100))
                    kind = "identity" if lv == 0 else ("nuisance" if fam in MESH_INVARIANT else "error")
                    add(perturb_mesh(m, fam, lv, rng), fam, kind, lv, rep)
        for label, vm in variants.get(name, []):
            add(vm, "retessellation", "nuisance", 1.0, 0, label)
        log(f"mesh bench: {name} ✓")
    return pd.DataFrame(rows)


def _drop_planes_v13(m: trimesh.Trimesh):
    """Helper-component rule of v1.3 (flat and at most 128 faces), kept for the comparison."""
    parts = m.split(only_watertight=False)
    if len(parts) <= 1:
        return m
    keep = []
    for p in parts:
        ext = np.sort(p.bounding_box_oriented.extents) if len(p.vertices) >= 4 else np.zeros(3)
        if not (ext[2] > 0 and ext[0] < 0.01 * ext[2] and len(p.faces) <= 128):
            keep.append(p)
    return trimesh.util.concatenate(keep) if keep and len(keep) < len(parts) else m


def run_robustness_bench(imgs: dict, meshes: dict, viewport_shots: dict | None = None, clean_extra: dict | None = None, log=print):
    """(i) screenshots with a Blender-like interface (dark / light theme): crop IoU and scores with and without
    automatic cropping, for the exact render (geometry of the crop) and for solid-shaded viewport content;
    (ii) false crops on clean images; (iii) auxiliary floors and a thin model part, helper rule v1.3 vs v1.4."""
    from .mesh_metrics import drop_plane_components
    from .ui_crop import box_iou, detect_viewport, mock_ui
    rows, crop_rows = [], []
    for src, pool in (("render", imgs), ("viewport content", viewport_shots or {})):
        for name, ref in pool.items():
            R = np.asarray(ref.convert("RGB"), float)
            for theme in ("dark", "light"):
                for lay in range(3):
                    for seed in range(2 if src == "render" else 1):
                        ui, box = mock_ui(ref, lay, seed, theme, return_box=True)
                        U = np.asarray(ui, float)
                        det = detect_viewport(U)
                        crop_rows.append({"content": src, "theme": theme, "reference": name, "layout": lay, "seed": seed,
                                          "iou": box_iou(det, box), "detected": det is not None})
                        for crop in (None, "auto"):
                            r = compare_images(R, U, trim=True, use_lpips=False, crop=crop)
                            rows.append({"test": f"UI screenshot ({src}, {theme})", "reference": name, "variant": f"layout {lay}",
                                         "method": "auto crop" if crop else "no crop", "score": score({"type": "image"}, r.m)})
    clean = dict(imgs, **(clean_extra or {}))
    for name, im in clean.items():
        det = detect_viewport(np.asarray(im.convert("RGB"), float))
        crop_rows.append({"content": "clean image", "theme": "-", "reference": name, "layout": -1, "seed": 0, "iou": np.nan,
                          "detected": det is not None})
    # ---- floors and a thin part of the model
    for name, m in meshes.items():
        ext = float(np.max(m.extents))
        z0 = float(m.bounds[0][2]) - 0.01 * ext

        def floor(s, t=0.0025, sub=0):
            f = trimesh.creation.box(extents=[s * ext, s * ext, t * s * ext])
            for _ in range(sub):
                f = f.subdivide()
            return f.apply_translation([float(m.centroid[0]), float(m.centroid[1]), z0 - t * s * ext / 2])

        plate = trimesh.creation.box(extents=[0.4 * ext, 0.003 * ext, 0.4 * ext]).apply_translation(
            [float(m.centroid[0]), float(m.bounds[0][1]) - 0.002 * ext, float(m.centroid[2])])
        ref_floor = trimesh.util.concatenate([m, floor(4)])
        cases = [("no floor", m), ("thin floor 3x", trimesh.util.concatenate([m, floor(3)])),
                 ("thin floor 4x", trimesh.util.concatenate([m, floor(4)])), ("thin floor 6x", trimesh.util.concatenate([m, floor(6)])),
                 ("subdivided floor 4x", trimesh.util.concatenate([m, floor(4, sub=3)])),
                 ("slab floor 4x (2 % thick)", trimesh.util.concatenate([m, floor(4, t=0.02)]))]
        for rule, fn in (("no removal", lambda x: x), ("helper rule v1.3", _drop_planes_v13), ("helper rule v1.4", lambda x: drop_plane_components(x)[0])):
            rf = fn(ref_floor.copy())
            ref_pts, ref_t = prepare_reference(rf), topology(rf)
            for lab, stu in cases:
                r = compare_meshes(fn(stu.copy()), ref_pts, ref_topo=ref_t)
                rows.append({"test": "auxiliary floor", "reference": name, "variant": lab, "method": rule, "score": score({"type": "mesh"}, r.m)})
            # a thin plate that belongs to the model: a student who omits it must lose points
            rp = fn(trimesh.util.concatenate([m, plate]).copy())
            ref_pl, ref_plt = prepare_reference(rp), topology(rp)
            for lab, stu in (("plate present", trimesh.util.concatenate([m, plate])), ("plate missing (error)", m)):
                r = compare_meshes(fn(stu.copy()), ref_pl, ref_topo=ref_plt)
                rows.append({"test": "thin model part", "reference": name, "variant": lab, "method": rule, "score": score({"type": "mesh"}, r.m)})
    log("robustness bench ✓")
    return pd.DataFrame(rows), pd.DataFrame(crop_rows)


# ------------------------------------------------------------------ analysis
def _plt():
    from .reports import _plt as p
    return p()


def _mono(g):
    m = g.groupby("level")["score"].mean()
    inc = np.diff(m.values)
    return {"rho_level_score": spearmanr(g["level"], g["score"]).statistic, "score_L0": m.iloc[0], "score_L0.2": m.get(0.2, np.nan),
            "score_L0.4": m.get(0.4, np.nan), "score_L1": m.iloc[-1], "monotone_means": bool((inc <= 1e-6).all()),
            "max_increase": float(max(0.0, inc.max())) if len(inc) else 0.0}


def summarise(img: pd.DataFrame, mesh: pd.DataFrame):
    out = {}
    if img is not None and len(img):
        mono, nuis, sens = [], [], {}
        for mode, gm in img.groupby("mode"):
            _, nf, ef = IMG_MODES[mode]
            ie = gm[gm.family.isin(ef)]
            for fam, g in ie.groupby("family"):
                mono.append({"mode": mode, "family": fam, **_mono(g)})
            t = gm[gm.kind == "nuisance"].groupby("family")["score"].agg(["count", "mean", "std", "min"]).reset_index()
            t.insert(0, "mode", mode)
            nuis.append(t)
            if mode == "composition":
                for col, lab in IMG_METRICS:
                    if col in ie and ie[col].notna().any():
                        sens[lab] = {fam: abs(spearmanr(g["level"], g[col].astype(float)).statistic) for fam, g in ie.groupby("family")
                                     if g[col].astype(float).std() > 0}
        out["image_monotonicity"] = pd.DataFrame(mono)
        out["image_nuisance"] = pd.concat(nuis, ignore_index=True)
        out["image_sensitivity"] = pd.DataFrame(sens).T
    if mesh is not None and len(mesh):
        inv = []
        for fam, g in mesh[mesh.kind == "nuisance"].groupby("family"):
            inv.append({"family": fam, "n": len(g), "score_mean": g.score.mean(), "score_sd": g.score.std(), "score_min": g.score.min(),
                        "cd_mean": g.cd.mean(), "cd_noalign_mean": g.cd_noalign.mean(), "fscore_mean": g.fscore.mean()})
        out["mesh_invariance"] = pd.DataFrame(inv)
        me = mesh[mesh.family.isin(MESH_ERRORS)]
        out["mesh_monotonicity"] = pd.DataFrame([{"family": fam, **_mono(g)} for fam, g in me.groupby("family")])
        sens = {}
        for col, lab in MESH_METRICS:
            if col in me:
                sens[lab] = {fam: abs(spearmanr(g["level"], g[col].astype(float)).statistic) for fam, g in me.groupby("family")
                             if g[col].astype(float).std() > 0}
        out["mesh_sensitivity"] = pd.DataFrame(sens).T
        noise = mesh[mesh.kind == "identity"]
        out["mesh_noise_floor"] = {"cd_mean": float(noise.cd.mean()), "h95_mean": float(noise.h95.mean())}
    return out


def _fb_row(channel, fam, exp, g, top=86):
    """Hit rate over all error cases and over the cases graded below the top grade (score < top), and the share
    of those penalised cases that received no problem message at all."""
    keys = g.feedback.fillna("").map(lambda f: set(k for k in f.split(";") if k))
    hit = keys.map(lambda k: bool(k & exp))
    pen = g.score < top
    return {"channel": channel, "family": fam, "expected": ", ".join(sorted(exp)), "n": len(g), "rate": float(hit.mean()),
            "n_penalised": int(pen.sum()), "rate_penalised": float(hit[pen].mean()) if pen.any() else float("nan"),
            "silent_penalised": float((keys[pen].map(len) == 0).mean()) if pen.any() else float("nan")}


def feedback_validation(img, mesh, min_level=0.2):
    """Hit rate: share of error cases (severity >= min_level) for which at least one message that names the
    injected error is issued, overall and among the cases graded below 5 (score < 86); false alarms: share of
    nuisance cases with any problem message."""
    rows = []
    if img is not None and len(img):
        for mode, gm in img.groupby("mode"):
            nu = gm[gm.kind == "nuisance"]
            rows.append({"channel": f"image ({mode} mode)", "family": "nuisance (false alarms)", "expected": "none",
                         "n": len(nu), "rate": float((nu.feedback.fillna("") != "").mean())})
            for fam, g in gm[(gm.kind == "error") & (gm.level >= min_level)].groupby("family"):
                rows.append(_fb_row(f"image ({mode} mode)", fam, IMG_EXPECTED.get(fam, set()), g))
            oc = gm[(gm.family == "occlusion") & (gm.level >= min_level) & gm.get("occluder_cell", pd.Series(dtype=float)).notna()]
            if len(oc):
                rows.append({"channel": f"image ({mode} mode)", "family": "occlusion: worst cell = occluder", "expected": "fb_region location",
                             "n": len(oc), "rate": float((oc.worst_cell == oc.occluder_cell).mean())})
    if mesh is not None and len(mesh):
        nu = mesh[mesh.kind == "nuisance"]
        rows.append({"channel": "mesh", "family": "nuisance (false alarms)", "expected": "none", "n": len(nu),
                     "rate": float((nu.feedback.fillna("") != "").mean())})
        for fam, g in mesh[(mesh.kind == "error") & (mesh.level >= min_level)].groupby("family"):
            rows.append(_fb_row("mesh", fam, MESH_EXPECTED[fam], g))
    return pd.DataFrame(rows)


def discrimination(img, mesh, rend):
    """Framework vs baselines: AUC of nuisance vs error (E), overall and per subset."""
    from .discrimination import compare_methods, detectability, min_detectable
    D = {}

    def per_subset(df, methods, key, **kw):
        parts = []
        for sub in ("all", "development", "held-out"):
            g = df if sub == "all" else df[df["subset"] == sub]
            if g["reference"].nunique() < 2 or not (g["kind"] == "error").any():
                continue
            t = compare_methods(g, methods, **kw)
            t.insert(0, "objects", sub)
            parts.append(t)
        D[key] = pd.concat(parts, ignore_index=True)

    if img is not None and len(img):
        for mode, gm in img.groupby("mode"):
            per_subset(gm, IMG_METHODS, f"image_auc_{mode}", min_level=0.2)
            D[f"image_detectability_{mode}"] = detectability(gm, IMG_METHODS)
            D[f"image_min_severity_{mode}"] = min_detectable(D[f"image_detectability_{mode}"])
    if mesh is not None and len(mesh):
        per_subset(mesh, MESH_METHODS, "mesh_auc", min_level=0.2)
        D["mesh_detectability"] = detectability(mesh, MESH_METHODS)
        D["mesh_min_severity"] = min_detectable(D["mesh_detectability"])
    if rend is not None and len(rend):
        parts = []
        for pipe, g in rend.groupby("pipeline"):
            for sub in ("all", "development", "held-out"):
                gg = g if sub == "all" else g[g["subset"] == sub]
                if gg["reference"].nunique() < 2:
                    continue
                t = compare_methods(gg, IMG_METHODS, min_level=0.0)
                t.insert(0, "objects", sub)
                t.insert(0, "pipeline", pipe)
                parts.append(t)
        D["render_auc"] = pd.concat(parts, ignore_index=True)
        D["render_scores"] = rend.groupby(["pipeline", "kind", "family"])["score"].agg(["mean", "std", "min", "max", "count"]).reset_index()
    return D


# ------------------------------------------------------------------ figures (repository versions)
MARKERS = ["o", "s", "^", "v", "D", "P", "X", "*", "<", ">", "h"]
STYLES = ["-", "--", "-.", ":"]


def figures(img, mesh, rend, S, D, out: Path):
    plt = _plt()
    figs = {}

    def save(fig, name):
        fig.savefig(out / f"{name}.png"); fig.savefig(out / f"{name}.pdf"); plt.close(fig)
        figs[name] = out / f"{name}.png"

    if img is not None and len(img):
        for mode, gm in img.groupby("mode"):
            fig, ax = plt.subplots(figsize=(4.6, 3.2))
            for k, (fam, g) in enumerate(gm[gm.kind.isin(["error", "identity"]) & gm.family.isin(IMG_MODES[mode][2])].groupby("family")):
                m = g.groupby("level")["score"].mean()
                ax.plot(m.index, m.values, marker=MARKERS[k % len(MARKERS)], ms=3.5, lw=1.1, ls=STYLES[k % 4], label=fam)
            ax.set(xlabel="Error severity", ylabel="Framework score", ylim=(0, 102), title=f"Image channel ({mode} mode)")
            ax.legend(frameon=False, fontsize=6.5, ncol=2)
            save(fig, f"fig_bench_image_curves_{mode}")
    if mesh is not None and len(mesh):
        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        for k, (fam, g) in enumerate(mesh[mesh.family != "retessellation"].groupby("family")):
            m = g.groupby("level")["score"].mean()
            ax.plot(m.index, m.values, marker=MARKERS[k % len(MARKERS)], ms=3.5, lw=1.1, ls="--" if fam in MESH_INVARIANT else "-", label=fam)
        ax.set(xlabel="Perturbation severity", ylabel="Framework score", ylim=(0, 102), title="Mesh channel (dashed = nuisance)")
        ax.legend(frameon=False, fontsize=6.5, ncol=2)
        save(fig, "fig_bench_mesh_curves")
    if "render_auc" in D:
        R = D["render_auc"][D["render_auc"].objects == "all"]
        meths = [m for m in R["method"].unique()]
        pipes = [p for p in ("viewport", "screenshot", "re-render", "re-render 512px") if p in set(R.pipeline)]
        x = np.arange(len(meths))
        fig, ax = plt.subplots(figsize=(5.6, 3.2))
        wdt = 0.8 / len(pipes)
        for k, pipe in enumerate(pipes):
            g = R[R.pipeline == pipe].set_index("method").reindex(meths)
            ax.bar(x + (k - (len(pipes) - 1) / 2) * wdt, g["AUC"], wdt, label=pipe, hatch=["", "//", "..", "xx"][k % 4], edgecolor="black", lw=0.3)
        ax.axhline(0.5, color="#667085", lw=0.8, ls=":")
        ax.set(xticks=x, ylim=(0, 1.05), ylabel="AUC (nuisance vs error)", title="Render-level errors by evidence pipeline")
        ax.set_xticklabels(meths, rotation=35, ha="right", fontsize=7)
        ax.legend(frameon=False, fontsize=7, loc="lower right")
        save(fig, "fig_bench_render_auc")
    return figs


def report_md(S, D, F, SE, det, runtime, meta):
    L = [f"# Controlled-perturbation benchmark — {ALGO_VERSION}", "", f"References: {meta}", ""]
    for key, v in list(D.items()) + list(S.items()) + [("feedback_validation", F)] + list((SE or {}).items()):
        if isinstance(v, pd.DataFrame) and len(v):
            L += [f"## {key}", "", v.round(4).to_markdown(index=key.endswith("sensitivity")), ""]
    L += [f"Determinism: {det}", "", "## Runtime (s per assessment)", "", pd.DataFrame([runtime]).round(3).to_markdown(index=False), ""]
    return "\n".join(L)


def run_benchmark(out, ref_images_dir=None, ref_meshes_dir=None, blender=True, repeats=3, use_lpips=True, log=print):
    from .bench_render import COMPOSITIONS, HELD_OUT, run_render_bench
    from .sensitivity import sensitivity_tables
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rend, det, assets = run_render_bench(out / "_render", use_lpips=use_lpips, log=log) if blender else (None, None, {})
    held = {COMPOSITIONS[c] for c in HELD_OUT}
    imgs, subsets = builtin_images(), {}
    for name in ("suzanne", "still_life", "chair", "mug", "desk_lamp") + tuple(sorted(held)):
        p = out / "_render" / name / "ref.png"
        if p.exists():
            imgs["render_" + name] = Image.open(p).convert("RGB")
            subsets["render_" + name] = "held-out" if name in held else "development"
    if ref_images_dir:
        for p in sorted(Path(ref_images_dir).glob("*")):
            if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
                imgs[p.stem] = Image.open(p).convert("RGB")
    meshes, variants = builtin_meshes(), builtin_mesh_variants()
    for name in ("suzanne", "chair", "mug", "desk_lamp") + tuple(sorted(held)):
        a = assets.get(name)
        if a and Path(a["ref"]).exists():
            meshes["asset_" + name] = load_mesh(a["ref"])
            subsets["asset_" + name] = "held-out" if name in held else "development"
            variants["asset_" + name] = [(lab, load_mesh(a[f"tess{k}"])) for k, lab in ((1, "half"), (2, "double")) if Path(a.get(f"tess{k}", "")).exists()]
    if ref_meshes_dir:
        for p in sorted(Path(ref_meshes_dir).glob("*")):
            if p.suffix.lower() in (".obj", ".stl", ".ply", ".off", ".glb"):
                meshes[p.stem] = load_mesh(p)
    img = run_image_bench(imgs, repeats, use_lpips, log, subsets=subsets)
    mesh = run_mesh_bench(meshes, repeats, log, variants=variants, subsets=subsets)
    shots, clean_extra = {}, {}
    if rend is not None:
        for name in sorted({COMPOSITIONS[c] for c in COMPOSITIONS}):
            d = out / "_render" / name
            if (d / "nuis_view0.png").exists():
                shots[name] = Image.open(d / "nuis_view0.png").convert("RGB")
            for f in ("nuis_shot0.png", "nuis_view1.png", "err0_rr.png", "err6_rr.png"):
                if (d / f).exists():
                    clean_extra[f"{name}/{f}"] = Image.open(d / f).convert("RGB")
    rb, crops = run_robustness_bench(imgs, meshes, shots, clean_extra, log)
    S = summarise(img, mesh)
    S["robustness"] = rb.groupby(["test", "method", "variant"])["score"].agg(["mean", "std", "min", "max", "count"]).reset_index() \
        if len(rb) else pd.DataFrame()
    S["crop_accuracy"] = crops.groupby(["content", "theme"]).agg(n=("detected", "size"), detected=("detected", "mean"),
                                                                  iou_mean=("iou", "mean"), iou_ge_09=("iou", lambda v: float((v >= 0.9).mean()))).reset_index()
    D = discrimination(img, mesh, rend)
    F = feedback_validation(img, mesh)
    SE = sensitivity_tables(img, mesh, rend)
    rt = {"image_s": float(img.seconds.mean()), "mesh_s": float(mesh.seconds.mean())}
    if det:
        rt["blender_render_s"] = det.get("render_seconds")
    img.to_csv(out / "bench_image.csv", index=False)
    mesh.to_csv(out / "bench_mesh.csv", index=False)
    rb.to_csv(out / "bench_robustness.csv", index=False)
    crops.to_csv(out / "bench_crop_accuracy.csv", index=False)
    F.to_csv(out / "bench_feedback.csv", index=False)
    if rend is not None:
        rend.to_csv(out / "bench_render.csv", index=False)
    with pd.ExcelWriter(out / "benchmark_tables.xlsx") as xw:
        for k, v in list(D.items()) + list(S.items()) + [("feedback_validation", F)] + list(SE.items()):
            if isinstance(v, pd.DataFrame) and len(v):
                (v.rename_axis("metric").reset_index() if k.endswith("sensitivity") else v).to_excel(xw, sheet_name=k[:31], index=False)
    figs = figures(img, mesh, rend, S, D, out)
    lp = bool(use_lpips and "lpips" in img and img["lpips"].notna().any())
    meta = {"version": ALGO_VERSION, "images": list(imgs), "meshes": list(meshes), "subsets": subsets, "error_levels": LEVELS,
            "image_nuisance_levels": NUIS_LEVELS, "mesh_nuisance_levels": [lv for lv in LEVELS if lv > 0], "repeats": repeats, "lpips": lp,
            "render_compositions": sorted(rend.reference.unique().tolist()) if rend is not None else [],
            "blender_version": (det or {}).get("blender_version")}
    (out / "benchmark_report_en.md").write_text(report_md(S, D, F, SE, det, rt, meta), encoding="utf-8")
    (out / "benchmark_summary.json").write_text(json.dumps({"meta": meta, "determinism": det, "runtime": rt,
                                                            "mesh_noise_floor": S.get("mesh_noise_floor")}, indent=2, default=str), encoding="utf-8")
    return {"summary": S, "discrimination": D, "feedback": F, "sensitivity": SE, "render": rend, "determinism": det, "runtime": rt,
            "figures": figs, "out": out}
