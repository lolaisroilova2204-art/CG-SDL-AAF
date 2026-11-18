"""Render-level benchmark with Blender (part C of the technical validation).

Twelve parametric compositions are generated with bpy: six development compositions (torus + cube,
Suzanne on a plinth, still life with a procedural texture, chair, mug, desk lamp) that existed when
the defaults were fixed, and six held-out compositions created afterwards (table + vase, bottle +
glass, snowman, rocket, robot, kettle). For each composition:

  reference   canonical render, 32 samples, 256 px (helper objects hidden)
  re-render pipeline (the scene file is submitted and re-rendered canonically)
      nuisance  8 / 16 / 64 samples at 256 / 384 / 192 px; three sets of student-like render settings
                (EEVEE + AgX + Full HD; own camera + Filmic + 128 samples; transparent film + 50 % +
                denoising); the same objects tessellated with half / double the segment counts
      errors    hue shift 0.04 / 0.10 / 0.25 · main object z-scale 0.9 / 0.75 / 0.5 · secondary object
                missing · light energy x0.6 / x0.35 / x0.15 · an extra object
  screenshot pipeline (an image is submitted): the correct and the erroneous scenes rendered from
      jittered viewpoints with the floor visible, (a) with Cycles ("render screenshot") and (b) with the
      Workbench engine in solid shading ("viewport screenshot", what the 3D viewport shows)
  resolution check: the re-render pipeline repeated at 512 px
plus determinism (two identical re-renders) and runtime. The main objects (and their re-tessellated
versions) are exported as OBJ for the mesh benchmark.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .blender import find_blender, run_blender, run_script
from .image_metrics import compare_images, load_image
from .scoring import composite, score

SCENE2 = Path(__file__).with_name("blender_make_scene2.py")
COMPOSITIONS = {0: "torus_cube", 1: "suzanne", 2: "still_life", 3: "chair", 4: "mug", 5: "desk_lamp",
                6: "table_vase", 7: "bottle_glass", 8: "snowman", 9: "rocket", 10: "robot", 11: "kettle"}
HELD_OUT = {6, 7, 8, 9, 10, 11}
ERRORS = [("hue", 1 / 3, dict(hue=0.04)), ("hue", 2 / 3, dict(hue=0.10)), ("hue", 1.0, dict(hue=0.25)),
          ("proportion", 1 / 3, dict(zs=0.9)), ("proportion", 2 / 3, dict(zs=0.75)), ("proportion", 1.0, dict(zs=0.5)),
          ("missing_object", 1.0, dict(drop=1)), ("lighting", 1 / 3, dict(lf=0.6)), ("lighting", 2 / 3, dict(lf=0.35)),
          ("lighting", 1.0, dict(lf=0.15)), ("extra_object", 1.0, dict(extra=1))]
NUIS_RENDER = [(8, 256), (16, 384), (64, 192)]
SETTINGS = {1: "EEVEE + AgX + 1920x1080", 2: "own camera + Filmic + 128 samples", 3: "transparent film + 50 % + denoising"}
TESS = {1: "half segment counts", 2: "double segment counts"}
JITTERS = [(8, 3, 1.0), (-8, -3, 1.05), (15, 0, 0.95), (-15, 5, 1.1), (22, -5, 0.92), (-22, 6, 1.12)]


def _scene(path, comp, hue=0.0, zs=1.0, drop=0, lf=1.0, extra=0, mesh_out=None, tess=0, settings=0, blender=None):
    args = [path, comp, hue, zs, drop, lf, extra, mesh_out or "-", tess, settings]
    run_script(SCENE2, args, blender=blender)


def _metrics(ref, img_path, use_lpips):
    r = compare_images(ref, load_image(img_path), trim=True, use_lpips=use_lpips, baselines=True)
    out = dict(r.raw)
    out["score"] = score({"type": "image"}, r.m)
    out["composite"] = composite({"type": "image"}, r.m)
    return out


def run_render_bench(work: Path, use_lpips=True, res=256, blender=None, log=print, compositions=None, hires=512):
    b = blender or find_blender()
    if b is None:
        log("Blender not found — render benchmark skipped")
        return None, None, {}
    work.mkdir(parents=True, exist_ok=True)
    rows, assets, det = [], {}, None
    rng = np.random.default_rng(11)
    for comp in (compositions or list(COMPOSITIONS)):
        name = COMPOSITIONS[comp]
        subset = "held-out" if comp in HELD_OUT else "development"
        cdir = work / name
        cdir.mkdir(exist_ok=True)
        _scene(cdir / "ref.blend", comp, mesh_out=cdir / f"{name}.obj", blender=b)
        assets[name] = {"ref": cdir / f"{name}.obj"}
        for tv in TESS:
            _scene(cdir / f"tess{tv}.blend", comp, mesh_out=cdir / f"{name}_tess{tv}.obj", tess=tv, blender=b)
            assets[name][f"tess{tv}"] = cdir / f"{name}_tess{tv}.obj"
        for sv in SETTINGS:
            _scene(cdir / f"settings{sv}.blend", comp, settings=sv, blender=b)
        t0 = time.perf_counter()
        r0 = run_blender(cdir / "ref.blend", cdir / "ref.png", None, res=res, samples=32, blender=b)
        t_render = time.perf_counter() - t0
        ref = load_image(cdir / "ref.png")
        if det is None:  # determinism on the first composition
            run_blender(cdir / "ref.blend", cdir / "ref_again.png", cdir / "a.obj", res=res, samples=32, blender=b)
            run_blender(cdir / "ref.blend", cdir / "ref_again2.png", cdir / "b.obj", res=res, samples=32, blender=b)
            x1 = np.asarray(Image.open(cdir / "ref.png").convert("RGB"), int)
            x2 = np.asarray(Image.open(cdir / "ref_again.png").convert("RGB"), int)
            det = {"max_abs_pixel_diff": int(np.abs(x1 - x2).max()), "identical_obj": (cdir / "a.obj").read_bytes() == (cdir / "b.obj").read_bytes(),
                   "render_seconds": round(t_render, 2), "blender_version": r0.get("blender_version")}

        def add(p, pipeline, kind, family, level, ref_img=ref):
            rows.append({"reference": name, "subset": subset, "pipeline": pipeline, "kind": kind, "family": family, "level": level,
                         **_metrics(ref_img, p, use_lpips)})

        # ---- re-render pipeline: nuisance
        for k, (spp, rr) in enumerate(NUIS_RENDER):
            p = cdir / f"nuis_rr{k}.png"
            run_blender(cdir / "ref.blend", p, None, res=rr, samples=spp, blender=b)
            add(p, "re-render", "nuisance", "samples / resolution", 0.0)
        for sv in SETTINGS:
            p = cdir / f"nuis_settings{sv}.png"
            run_blender(cdir / f"settings{sv}.blend", p, None, res=res, samples=16, blender=b)
            add(p, "re-render", "nuisance", "student render settings", 0.0)
        for tv in TESS:
            p = cdir / f"nuis_tess{tv}.png"
            run_blender(cdir / f"tess{tv}.blend", p, None, res=res, samples=16, blender=b)
            add(p, "re-render", "nuisance", "re-tessellation", 0.0)
        # ---- screenshot pipelines: nuisance (correct scene, other viewpoints, floor visible)
        for k, jit in enumerate(JITTERS):
            p = cdir / f"nuis_shot{k}.png"
            run_blender(cdir / "ref.blend", p, None, res=res, samples=16, blender=b, jitter=jit, helpers="show")
            add(p, "screenshot", "nuisance", "viewpoint", 0.0)
            p = cdir / f"nuis_view{k}.png"
            run_blender(cdir / "ref.blend", p, None, res=res, samples=1, engine="BLENDER_WORKBENCH", blender=b, jitter=jit, helpers="show")
            add(p, "viewport", "nuisance", "viewpoint", 0.0)
        # ---- errors in all pipelines
        err_blends = []
        for k, (fam, lv, kw) in enumerate(ERRORS):
            bl = cdir / f"err{k}.blend"
            _scene(bl, comp, blender=b, **kw)
            err_blends.append((bl, fam, lv))
            p = cdir / f"err{k}_rr.png"
            run_blender(bl, p, None, res=res, samples=16, blender=b)
            add(p, "re-render", "error", fam, lv)
            jit = JITTERS[int(rng.integers(len(JITTERS)))]
            p = cdir / f"err{k}_shot.png"
            run_blender(bl, p, None, res=res, samples=16, blender=b, jitter=jit, helpers="show")
            add(p, "screenshot", "error", fam, lv)
            p = cdir / f"err{k}_view.png"
            run_blender(bl, p, None, res=res, samples=1, engine="BLENDER_WORKBENCH", blender=b, jitter=jit, helpers="show")
            add(p, "viewport", "error", fam, lv)
        # ---- resolution check: the re-render pipeline at a higher resolution
        if hires:
            run_blender(cdir / "ref.blend", cdir / "ref_hi.png", None, res=hires, samples=32, blender=b)
            ref_hi = load_image(cdir / "ref_hi.png")
            for k, spp in enumerate((8, 64)):
                p = cdir / f"hi_nuis_rr{k}.png"
                run_blender(cdir / "ref.blend", p, None, res=hires, samples=spp, blender=b)
                add(p, f"re-render {hires}px", "nuisance", "samples / resolution", 0.0, ref_hi)
            for sv in SETTINGS:
                p = cdir / f"hi_nuis_settings{sv}.png"
                run_blender(cdir / f"settings{sv}.blend", p, None, res=hires, samples=16, blender=b)
                add(p, f"re-render {hires}px", "nuisance", "student render settings", 0.0, ref_hi)
            for tv in TESS:
                p = cdir / f"hi_nuis_tess{tv}.png"
                run_blender(cdir / f"tess{tv}.blend", p, None, res=hires, samples=16, blender=b)
                add(p, f"re-render {hires}px", "nuisance", "re-tessellation", 0.0, ref_hi)
            for k, (bl, fam, lv) in enumerate(err_blends):
                p = cdir / f"hi_err{k}_rr.png"
                run_blender(bl, p, None, res=hires, samples=16, blender=b)
                add(p, f"re-render {hires}px", "error", fam, lv, ref_hi)
        log(f"render bench: {name} ✓")
    return pd.DataFrame(rows), det, assets
