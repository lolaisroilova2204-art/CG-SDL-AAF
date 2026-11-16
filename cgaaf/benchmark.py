"""Controlled-perturbation benchmark (technical validation, no student data).

A. Image channel : reference renders x 4 nuisance + 9 error families x severity levels
B. Mesh channel  : reference meshes (primitives + Blender assets) x 4 nuisance + 5 error families
C. Blender       : 6 compositions, render-level errors and nuisances, screenshot vs re-render
                   pipeline, determinism (bench_render.py)
D. Robustness    : application UI in screenshots, auxiliary floors
E. Discrimination: AUC / Cohen d of nuisance vs error for the framework and every baseline
F. Runtime
All results are written to CSV / XLSX, publication figures and benchmark_report_en.md.
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
from .blender import MAKE_SCENE, find_blender, run_blender, run_script
from .image_metrics import compare_images, load_image
from .mesh_metrics import compare_meshes, load_mesh, prepare_reference, torus
from .scoring import composite, score

LEVELS = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
IMG_FAMILIES = ["translation", "rotation", "scale", "hue", "brightness", "blur", "noise", "jpeg", "occlusion"]
IMG_NUISANCE = ["n_jpeg", "n_noise", "n_shift", "n_resample"]   # changes a teacher would not penalise
NUIS_LEVELS = [1 / 3, 2 / 3, 1.0]
MESH_INVARIANT = ["rigid_rotation", "uniform_scale", "retriangulation", "micro_noise"]
MESH_ERRORS = ["anisotropic_scale", "vertex_noise", "hole", "part_removal", "extra_part"]
# (column, label, sign): sign orients the value so that larger = more similar (for AUC)
IMG_METHODS = [("score", "Framework score", 1), ("composite", "Framework composite P (unclipped)", 1), ("psnr", "PSNR", 1), ("ssim_plain", "SSIM (unweighted)", 1), ("ssim", "SSIM (fg-weighted)", 1),
               ("msssim", "MS-SSIM", 1), ("lpips", "LPIPS", -1), ("deltaE00", "ΔE00", -1), ("color_hist", "Colour histogram", 1), ("edge_f1", "Edge F1", 1)]
MESH_METHODS = [("score", "Framework score", 1), ("composite", "Framework composite P (unclipped)", 1), ("cd_noalign", "Chamfer, no alignment", -1), ("cd", "Chamfer, PCA+ICP", -1),
                ("hd", "Hausdorff", -1), ("h95", "Hausdorff-95", -1), ("fscore", "F-score@0.1", 1)]
IMG_METRICS = [("score", "Score"), ("psnr", "PSNR"), ("ssim_plain", "SSIM (unweighted)"), ("ssim", "SSIM (weighted)"), ("msssim", "MS-SSIM"),
               ("lpips", "LPIPS"), ("deltaE00", "ΔE00"), ("color_hist", "Hist. BC"), ("edge_f1", "Edge F1")]
MESH_METRICS = [("score", "Score"), ("cd_noalign", "CD (no align)"), ("cd", "CD"), ("hd", "HD"), ("h95", "H95"), ("fscore", "F-score"), ("boundary_edges", "Boundary edges")]


# ------------------------------------------------------------------ image perturbations
def _bg(img):
    a = np.asarray(img, float)
    return tuple(int(v) for v in np.mean([a[:4, :4].reshape(-1, 3).mean(0), a[:4, -4:].reshape(-1, 3).mean(0),
                                          a[-4:, :4].reshape(-1, 3).mean(0), a[-4:, -4:].reshape(-1, 3).mean(0)], 0))


def perturb_image(img: Image.Image, fam: str, lv: float, rng) -> Image.Image:
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
        a[max(0, cy - s // 2):cy + s // 2, max(0, cx - s // 2):cx + s // 2] = bg
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


def builtin_images():
    from .demo import scene
    return {"scene_A": scene(), "scene_B": scene(dx=-70, dy=15)}


# ------------------------------------------------------------------ runners
def run_image_bench(refs: dict, repeats=3, use_lpips=True, log=print):
    rows = []
    for name, ref in refs.items():
        R = np.asarray(ref.convert("RGB"), float)
        for fam in IMG_NUISANCE + IMG_FAMILIES:
            nuis = fam in IMG_NUISANCE
            for lv in (NUIS_LEVELS if nuis else LEVELS):
                for rep in range(repeats if fam in ("noise", "occlusion", "n_noise") else 1):
                    rng = np.random.default_rng(1000 * rep + int(lv * 100))
                    t0 = time.perf_counter()
                    res = compare_images(R, np.asarray(perturb_image(ref, fam, lv, rng), float), trim=False, use_lpips=use_lpips)
                    dt_ = time.perf_counter() - t0
                    rows.append({"reference": name, "family": fam, "kind": "nuisance" if nuis else ("identity" if lv == 0 else "error"),
                                 "level": lv, "repeat": rep, **res.raw, "score": score({"type": "image"}, res.m),
                                 "composite": composite({"type": "image"}, res.m), "seconds": dt_})
        log(f"image bench: {name} ✓")
    return pd.DataFrame(rows)


def run_mesh_bench(refs: dict, repeats=3, log=print):
    rows = []
    for name, m in refs.items():
        ref_pts = prepare_reference(m, "pca_icp")
        for fam in MESH_INVARIANT + MESH_ERRORS:
            levels = [0.0, 0.5, 1.0] if fam == "retriangulation" else LEVELS
            for lv in levels:
                for rep in range(repeats):
                    rng = np.random.default_rng(7919 * rep + int(lv * 100))
                    t0 = time.perf_counter()
                    res = compare_meshes(perturb_mesh(m, fam, lv, rng), ref_pts, "pca_icp")
                    dt_ = time.perf_counter() - t0
                    rows.append({"reference": name, "family": fam, "kind": "nuisance" if fam in MESH_INVARIANT else ("identity" if lv == 0 else "error"),
                                 "level": lv, "repeat": rep, **res.raw, **{"m_" + k: v for k, v in res.m.items()},
                                 "score": score({"type": "mesh"}, res.m), "composite": composite({"type": "mesh"}, res.m), "seconds": dt_})
        log(f"mesh bench: {name} ✓")
    return pd.DataFrame(rows)


JITTERS = [(0, 0, 1.0), (5, 0, 1.0), (-5, 3, 1.0), (10, -3, 1.0), (-10, 5, 1.05), (15, 0, 0.95), (-15, -5, 1.1), (20, 8, 1.0), (-20, 0, 1.15), (25, -6, 0.9)]
ERR_SCENES = {"correct": (0.0, 0.35, 1), "colour_error": (0.45, 0.35, 1), "proportion_error": (0.0, 0.20, 1), "missing_object": (0.0, 0.35, 0)}


def run_blender_bench(work: Path, res=256, samples=16, blender=None, log=print):
    """Simulated screenshots (identical scene, jittered viewpoint) vs canonical re-render."""
    b = blender or find_blender()
    if b is None:
        log("Blender not found — part C skipped")
        return None, None
    work.mkdir(parents=True, exist_ok=True)
    for k, (hue, minor, cube) in ERR_SCENES.items():
        run_script(MAKE_SCENE, [work / f"{k}.blend", hue, minor, cube], blender=b)
    # determinism
    run_blender(work / "correct.blend", work / "ref.png", work / "ref.obj", res=res, samples=samples, blender=b)
    run_blender(work / "correct.blend", work / "ref2.png", work / "ref2.obj", res=res, samples=samples, blender=b)
    a1, a2 = np.asarray(Image.open(work / "ref.png").convert("RGB"), int), np.asarray(Image.open(work / "ref2.png").convert("RGB"), int)
    det = {"max_abs_pixel_diff": int(np.abs(a1 - a2).max()), "identical_obj": (work / "ref.obj").read_bytes() == (work / "ref2.obj").read_bytes()}
    ref_png = load_image(work / "ref.png")
    ref_pts = prepare_reference(load_mesh(work / "ref.obj"))
    rows = []
    for k in ERR_SCENES:
        t0 = time.perf_counter()
        run_blender(work / f"{k}.blend", work / f"{k}_canon.png", work / f"{k}.obj", res=res, samples=samples, blender=b)
        rt = time.perf_counter() - t0
        ir = compare_images(ref_png, load_image(work / f"{k}_canon.png"), trim=True, use_lpips=False)
        mr = compare_meshes(work / f"{k}.obj", ref_pts)
        rows.append({"scene": k, "mode": "re-render (canonical)", "jitter": "0/0/1.00", "score_image": score({"type": "image"}, ir.m),
                     "score_mesh": score({"type": "mesh"}, mr.m), "ssim": ir.raw["ssim"], "render_seconds": rt})
        for j, (az, el, df) in enumerate(JITTERS[1:]):
            p = work / f"{k}_shot{j}.png"
            run_blender(work / f"{k}.blend", p, None, res=res, samples=samples, blender=b, jitter=(az, el, df))
            ir = compare_images(ref_png, load_image(p), trim=True, use_lpips=False)
            rows.append({"scene": k, "mode": "screenshot (jittered view)", "jitter": f"{az}/{el}/{df:.2f}", "score_image": score({"type": "image"}, ir.m),
                         "score_mesh": np.nan, "ssim": ir.raw["ssim"], "render_seconds": np.nan})
        log(f"blender bench: {k} ✓")
    return pd.DataFrame(rows), det


def run_robustness_bench(imgs: dict, meshes: dict, log=print):
    """(i) screenshots containing application UI, with and without automatic viewport cropping;
    (ii) meshes with an auxiliary floor of varying size, with and without helper-component removal."""
    from .mesh_metrics import load_mesh as _lm
    from .ui_crop import mock_ui
    import tempfile
    rows = []
    for name, ref in imgs.items():
        R = np.asarray(ref.convert("RGB"), float)
        for lay in range(3):
            for seed in range(3):
                ui = np.asarray(mock_ui(ref, lay, seed), float)
                for crop in (None, "auto"):
                    r = compare_images(R, ui, trim=False, use_lpips=False, crop=crop)
                    rows.append({"test": "UI screenshot", "reference": name, "variant": f"layout {lay}", "method": "auto crop" if crop else "no crop",
                                 "score": score({"type": "image"}, r.m)})
    tmp = Path(tempfile.mkdtemp())
    floor = lambda s: trimesh.creation.box(extents=[s, s, 0.02]).apply_translation([0, 0, -0.45])
    for name, m in meshes.items():
        ext = float(np.max(m.extents))
        trimesh.util.concatenate([m, floor(4 * ext)]).export(tmp / "ref.obj")
        for drop in (False, True):
            ref = prepare_reference(_lm(tmp / "ref.obj", drop_planes=drop))
            for fs in (0, 3, 4, 6):
                p = tmp / f"s{fs}.obj"
                (trimesh.util.concatenate([m, floor(fs * ext)]) if fs else m).export(p)
                r = compare_meshes(_lm(p, drop_planes=drop), ref)
                rows.append({"test": "auxiliary floor", "reference": name, "variant": f"floor {fs}× (ref 4×)" if fs else "no floor (ref 4×)",
                             "method": "helper removal" if drop else "no removal", "score": score({"type": "mesh"}, r.m)})
    log("robustness bench ✓")
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ analysis & figures
def _plt():
    from .reports import _plt as p
    return p()


def summarise(img: pd.DataFrame, mesh: pd.DataFrame):
    out = {}
    if img is not None and len(img):
        mono = []
        ie = img[img.family.isin(IMG_FAMILIES)]
        for fam, g in ie.groupby("family"):
            m = g.groupby("level")["score"].mean()
            mono.append({"family": fam, "rho_level_score": spearmanr(g["level"], g["score"]).statistic, "score_L0": m.iloc[0], "score_L0.4": m.get(0.4, np.nan),
                         "score_L1": m.iloc[-1], "monotone_means": bool((np.diff(m.values) <= 1e-6).all())})
        out["image_monotonicity"] = pd.DataFrame(mono)
        out["image_nuisance"] = img[img.kind == "nuisance"].groupby("family")["score"].agg(["mean", "std", "min"]).reset_index()
        sens = {}
        for col, lab in IMG_METRICS:
            if col in ie and ie[col].notna().any():
                sens[lab] = {fam: abs(spearmanr(g["level"], g[col]).statistic) for fam, g in ie.groupby("family") if g[col].std() > 0}
        out["image_sensitivity"] = pd.DataFrame(sens).T
    if mesh is not None and len(mesh):
        inv = []
        for fam, g in mesh[mesh.kind == "nuisance"].groupby("family"):
            gg = g[g.level > 0]
            inv.append({"family": fam, "n": len(gg), "score_mean": gg.score.mean(), "score_min": gg.score.min(), "cd_mean": gg.cd.mean(), "cd_noalign_mean": gg.cd_noalign.mean(), "fscore_mean": gg.fscore.mean()})
        out["mesh_invariance"] = pd.DataFrame(inv)
        mono = []
        me = mesh[mesh.family.isin(MESH_ERRORS)]
        for fam, g in me.groupby("family"):
            m = g.groupby("level")["score"].mean()
            mono.append({"family": fam, "rho_level_score": spearmanr(g["level"], g["score"]).statistic, "score_L0": m.iloc[0], "score_L0.4": m.get(0.4, np.nan),
                         "score_L1": m.iloc[-1], "monotone_means": bool((np.diff(m.values) <= 1e-6).all())})
        out["mesh_monotonicity"] = pd.DataFrame(mono)
        sens = {}
        for col, lab in MESH_METRICS:
            sens[lab] = {fam: abs(spearmanr(g["level"], g[col]).statistic) for fam, g in me.groupby("family") if g[col].std() > 0}
        out["mesh_sensitivity"] = pd.DataFrame(sens).T
        noise = mesh[(mesh.kind == "identity") | ((mesh.kind == "nuisance") & (mesh.level == 0))]
        out["mesh_noise_floor"] = {"cd_mean": float(noise.cd.mean()), "h95_mean": float(noise.h95.mean())}
    return out


def discrimination(img, mesh, rend):
    """Framework vs baselines: AUC of nuisance vs error (E)."""
    from .discrimination import compare_methods, detectability, min_detectable
    D = {}
    if img is not None and len(img):
        D["image_auc"] = compare_methods(img, IMG_METHODS, min_level=0.2)
        D["image_detectability"] = detectability(img, IMG_METHODS)
        D["image_min_severity"] = min_detectable(D["image_detectability"])
    if mesh is not None and len(mesh):
        D["mesh_auc"] = compare_methods(mesh, MESH_METHODS, min_level=0.2)
        D["mesh_detectability"] = detectability(mesh, MESH_METHODS)
        D["mesh_min_severity"] = min_detectable(D["mesh_detectability"])
    if rend is not None and len(rend):
        parts = []
        for pipe, g in rend.groupby("pipeline"):
            t = compare_methods(g, IMG_METHODS, min_level=0.0)
            t.insert(0, "pipeline", pipe)
            parts.append(t)
        D["render_auc"] = pd.concat(parts, ignore_index=True)
        D["render_scores"] = rend.groupby(["pipeline", "kind", "family"])["score"].agg(["mean", "std", "min", "max", "count"]).reset_index()
    return D


def figures(img, mesh, rend, S, D, out: Path):
    plt = _plt()
    figs = {}

    def save(fig, name):
        fig.savefig(out / f"{name}.png"); fig.savefig(out / f"{name}.pdf"); plt.close(fig)
        figs[name] = out / f"{name}.png"

    if img is not None and len(img):
        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        for fam, g in img[img.family.isin(IMG_FAMILIES)].groupby("family"):
            m = g.groupby("level")["score"].mean()
            ax.plot(m.index, m.values, marker="o", ms=3, lw=1.2, label=fam)
        ax.set(xlabel="Error severity", ylabel="Framework score", ylim=(0, 102), title="Image channel: score vs error severity")
        ax.legend(frameon=False, fontsize=6.5, ncol=2)
        save(fig, "fig_bench_image_curves")
    if mesh is not None and len(mesh):
        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        for fam, g in mesh.groupby("family"):
            m = g.groupby("level")["score"].mean()
            ax.plot(m.index, m.values, marker="o", ms=3, lw=1.2, ls="--" if fam in MESH_INVARIANT else "-", label=fam)
        ax.set(xlabel="Perturbation severity", ylabel="Framework score", ylim=(0, 102), title="Mesh channel (dashed = nuisance)")
        ax.legend(frameon=False, fontsize=6.5, ncol=2)
        save(fig, "fig_bench_mesh_curves")
    for key in ("image_sensitivity", "mesh_sensitivity"):
        if key in S and len(S[key]):
            M = S[key]
            fig, ax = plt.subplots(figsize=(0.62 * M.shape[1] + 1.8, 0.34 * M.shape[0] + 1.0))
            im = ax.imshow(M.values.astype(float), cmap="Blues", vmin=0, vmax=1)
            ax.set(xticks=range(M.shape[1]), yticks=range(M.shape[0]), xticklabels=M.columns, yticklabels=M.index)
            plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    v = M.values[i, j]
                    if np.isfinite(v):
                        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color="white" if v > 0.6 else "black")
            ax.set_title("|Spearman ρ| between severity and metric")
            fig.colorbar(im, ax=ax, fraction=0.03)
            save(fig, f"fig_bench_{key}")
    for key, title in (("image_detectability", "Image: nuisance vs error at each severity"), ("mesh_detectability", "Mesh: nuisance vs error at each severity")):
        if key in D and len(D[key]):
            fig, ax = plt.subplots(figsize=(4.8, 3.3))
            for lab, g in D[key].groupby("method", sort=False):
                fw = lab.startswith("Framework")
                ax.plot(g["level"], g["AUC"], marker="o", ms=3 if not fw else 4, lw=2.2 if fw else 1, color="#2b59d9" if fw else None,
                        ls="--" if "composite" in lab else "-", alpha=1 if fw else 0.75, label=lab, zorder=3 if fw else 2)
            ax.axhline(0.5, color="#98a2b3", lw=0.8, ls=":")
            ax.set(xlabel="Error severity", ylabel="AUC (nuisance vs error)", ylim=(0.3, 1.02), title=title)
            ax.legend(frameon=False, fontsize=6.3, ncol=2)
            save(fig, f"fig_bench_{key}")
    if "render_auc" in D:
        R = D["render_auc"]
        meths = [m for m in R["method"].unique()]
        x = np.arange(len(meths))
        fig, ax = plt.subplots(figsize=(5.4, 3.2))
        for k, (pipe, col) in enumerate([("screenshot", "#98a2b3"), ("re-render", "#2b59d9")]):
            g = R[R.pipeline == pipe].set_index("method").reindex(meths)
            ax.bar(x + (k - 0.5) * 0.38, g["AUC"], 0.38, color=col, label={"screenshot": "screenshot (jittered viewpoint)", "re-render": "Blender re-render"}[pipe])
        ax.axhline(0.5, color="#667085", lw=0.8, ls=":")
        ax.set(xticks=x, ylim=(0, 1.05), ylabel="AUC (nuisance vs error)", title="Render-level errors: re-rendering vs screenshots")
        ax.set_xticklabels(meths, rotation=35, ha="right", fontsize=7)
        ax.legend(frameon=False, fontsize=7, loc="lower right")
        save(fig, "fig_bench_render_auc")
        fig, ax = plt.subplots(figsize=(5.0, 3.0))
        fams = ["nuisance"] + [f for f in ["hue", "proportion", "lighting", "missing_object", "extra_object"] if f in set(rend.family)]
        for k, (pipe, col) in enumerate([("screenshot", "#98a2b3"), ("re-render", "#2b59d9")]):
            for i, f in enumerate(fams):
                v = rend[(rend.pipeline == pipe) & ((rend.kind == "nuisance") if f == "nuisance" else (rend.family == f))]["score"]
                ax.scatter(np.full(len(v), i + (k - 0.5) * 0.3) + np.random.default_rng(i + 7 * k).normal(0, 0.03, len(v)), v, s=9, color=col,
                           label=({"screenshot": "screenshot (jittered viewpoint)", "re-render": "Blender re-render"}[pipe] if i == 0 else None))
        ax.set(xticks=range(len(fams)), ylim=(0, 102), ylabel="Framework score", title="Correct scenes (nuisance) vs scenes with errors")
        ax.set_xticklabels([f.replace("_", "\n") for f in fams], fontsize=7)
        ax.legend(frameon=False, fontsize=7)
        save(fig, "fig_bench_viewpoint")
    return figs


def robustness_table(rb):
    return rb.groupby(["test", "method"])["score"].agg(["mean", "std", "min", "max", "count"]).reset_index()


def report_md(S, D, det, runtime, meta):
    L = [f"# Controlled-perturbation benchmark — {ALGO_VERSION}", "", f"References: {meta}", ""]
    blocks = [("image_auc", "E1. Image channel — framework vs baselines (AUC, nuisance vs errors with severity ≥ 0.2)", D),
              ("image_min_severity", "E2. Image channel — smallest severity detected with AUC ≥ 0.9", D),
              ("mesh_auc", "E3. Mesh channel — framework vs baselines (AUC)", D), ("mesh_min_severity", "E4. Mesh channel — smallest severity detected", D),
              ("render_auc", "E5. Render-level errors — re-render vs screenshot pipeline (AUC)", D), ("render_scores", "C1. Render-level scores", D),
              ("image_monotonicity", "A1. Image channel — monotonicity", S), ("image_nuisance", "A2. Image channel — scores under nuisance", S),
              ("mesh_invariance", "B1. Mesh channel — nuisance invariance", S), ("mesh_monotonicity", "B2. Mesh channel — monotonicity", S),
              ("image_sensitivity", "A3. Image metric sensitivity |ρ|", S), ("mesh_sensitivity", "B3. Mesh metric sensitivity |ρ|", S),
              ("robustness", "D. Robustness: UI screenshots and auxiliary objects", S)]
    for key, title, src in blocks:
        if key in src and len(src[key]):
            L += [f"## {title}", "", src[key].round(3).to_markdown(index=key.endswith("sensitivity")), ""]
    L += [f"Determinism: {det}", "", "## Runtime (s per assessment)", "", pd.DataFrame([runtime]).round(3).to_markdown(index=False), "",
          "## Results paragraph (EN)", "", paragraph(S, D, det, runtime), ""]
    return "\n".join(L)


def _get(T, method, col="AUC"):
    r = T[T["method"] == method]
    return float(r[col].iloc[0]) if len(r) else np.nan


def paragraph(S, D, det, rt):
    s = []
    if "image_auc" in D:
        T = D["image_auc"]
        best_base = T[~T.method.str.startswith("Framework")].iloc[0]
        s.append(f"In the image channel the framework score separated nuisance variation from errors of severity ≥ 0.2 with AUC = {_get(T, 'Framework score'):.3f} "
                 f"({T.loc[T.method == 'Framework score', 'AUC 95% CI'].iloc[0]}), compared with {_get(T, 'PSNR'):.3f} for PSNR and {_get(T, 'SSIM (unweighted)'):.3f} for unweighted SSIM; "
                 f"the strongest single baseline was {best_base['method']} (AUC = {best_base['AUC']:.3f}).")
    if "image_monotonicity" in S:
        im = S["image_monotonicity"]
        s.append(f"The score decreased monotonically with severity for {int(im.monotone_means.sum())} of {len(im)} error families (median Spearman ρ = {im.rho_level_score.median():.2f}).")
    if "mesh_invariance" in S:
        iv = S["mesh_invariance"]
        s.append(f"The mesh channel was invariant to rigid rotation, uniform scaling, re-triangulation and micro-noise (mean score {iv.score_mean.mean():.1f}, minimum {iv.score_min.min():.1f}; "
                 f"mean Chamfer distance {iv.cd_mean.mean():.3f} versus {iv.cd_noalign_mean.mean():.3f} without pose alignment).")
    if "mesh_auc" in D:
        T = D["mesh_auc"]
        s.append(f"Against nuisance variation the framework reached AUC = {_get(T, 'Framework score'):.3f} for geometric errors, versus {_get(T, 'Chamfer, no alignment'):.3f} for Chamfer distance without alignment.")
    if "render_auc" in D:
        R = D["render_auc"]
        fr = R[(R.method == "Framework score") & (R.pipeline == "re-render")]["AUC"]
        fs = R[(R.method == "Framework score") & (R.pipeline == "screenshot")]["AUC"]
        rs = D["render_scores"]
        nshot = rs[(rs.pipeline == "screenshot") & (rs.kind == "nuisance")]
        nrr = rs[(rs.pipeline == "re-render") & (rs.kind == "nuisance")]
        if len(fr) and len(fs):
            s.append(f"For render-level errors in six Blender compositions, standardised re-rendering raised the framework's AUC from {float(fs.iloc[0]):.3f} (screenshots from jittered viewpoints) "
                     f"to {float(fr.iloc[0]):.3f}; correct scenes scored {nrr['mean'].mean():.1f} on average when re-rendered with different sample counts and resolutions, "
                     f"but {nshot['mean'].mean():.1f} as viewpoint-jittered screenshots.")
    if det:
        s.append(f"Re-rendering was deterministic (maximum pixel difference {det['max_abs_pixel_diff']}, identical mesh export: {det['identical_obj']}).")
    if "robustness" in S:
        R = S["robustness"].set_index(["test", "method"])["mean"]
        try:
            s.append(f"Screenshots that contained the application interface scored {R[('UI screenshot', 'no crop')]:.1f} on average and {R[('UI screenshot', 'auto crop')]:.1f} after automatic viewport cropping; "
                     f"an auxiliary floor reduced the mesh score to {R[('auxiliary floor', 'no removal')]:.1f} without and {R[('auxiliary floor', 'helper removal')]:.1f} with helper-component removal.")
        except KeyError:
            pass
    s.append(f"Mean assessment time was {rt.get('image_s', np.nan):.2f} s per image and {rt.get('mesh_s', np.nan):.2f} s per mesh on a CPU"
             + (f"; one Blender re-render took {rt['blender_render_s']:.2f} s." if "blender_render_s" in rt else "."))
    return " ".join(s)


def run_benchmark(out, ref_images_dir=None, ref_meshes_dir=None, blender=True, repeats=3, use_lpips=True, log=print):
    from .bench_render import run_render_bench
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rend, det, assets = run_render_bench(out / "_render", use_lpips=use_lpips, log=log) if blender else (None, None, {})
    imgs = builtin_images()
    for name in ("suzanne", "still_life", "chair", "mug", "desk_lamp"):
        p = out / "_render" / name / "ref.png"
        if p.exists():
            imgs["render_" + name] = Image.open(p).convert("RGB")
    if ref_images_dir:
        for p in sorted(Path(ref_images_dir).glob("*")):
            if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
                imgs[p.stem] = Image.open(p).convert("RGB")
    meshes = builtin_meshes()
    for name in ("suzanne", "chair", "mug", "desk_lamp"):
        if name in assets and Path(assets[name]).exists():
            meshes["asset_" + name] = load_mesh(assets[name])
    if ref_meshes_dir:
        for p in sorted(Path(ref_meshes_dir).glob("*")):
            if p.suffix.lower() in (".obj", ".stl", ".ply", ".off", ".glb"):
                meshes[p.stem] = load_mesh(p)
    img = run_image_bench(imgs, repeats, use_lpips, log)
    mesh = run_mesh_bench(meshes, repeats, log)
    rb = run_robustness_bench(imgs, meshes, log)
    S = summarise(img, mesh)
    S["robustness"] = robustness_table(rb)
    D = discrimination(img, mesh, rend)
    rt = {"image_s": float(img.seconds.mean()), "mesh_s": float(mesh.seconds.mean())}
    if det:
        rt["blender_render_s"] = det.get("render_seconds")
    img.to_csv(out / "bench_image.csv", index=False)
    mesh.to_csv(out / "bench_mesh.csv", index=False)
    rb.to_csv(out / "bench_robustness.csv", index=False)
    if rend is not None:
        rend.to_csv(out / "bench_render.csv", index=False)
    with pd.ExcelWriter(out / "benchmark_tables.xlsx") as xw:
        for k, v in list(D.items()) + list(S.items()):
            if isinstance(v, pd.DataFrame):
                (v.rename_axis("metric").reset_index() if k.endswith("sensitivity") else v).to_excel(xw, sheet_name=k[:31], index=False)
    figs = figures(img, mesh, rend, S, D, out)
    lp = bool(use_lpips and img["lpips"].notna().any())
    meta = {"images": list(imgs), "meshes": list(meshes), "error_levels": LEVELS, "nuisance_levels": NUIS_LEVELS, "repeats": repeats, "lpips": lp,
            "render_compositions": sorted(rend.reference.unique().tolist()) if rend is not None else []}
    (out / "benchmark_report_en.md").write_text(report_md(S, D, det, rt, meta), encoding="utf-8")
    (out / "benchmark_summary.json").write_text(json.dumps({"meta": meta, "determinism": det, "runtime": rt,
                                                            "paragraph": paragraph(S, D, det, rt)}, indent=2, default=str), encoding="utf-8")
    return {"summary": S, "discrimination": D, "render": rend, "determinism": det, "runtime": rt, "figures": figs, "out": out}
