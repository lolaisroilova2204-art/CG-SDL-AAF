"""Render-level benchmark with Blender (part C of the technical validation).

Six parametric compositions (torus+cube, Suzanne on a plinth, still life with a procedural
texture, chair, mug, desk lamp) are generated with bpy. For each composition:
  reference       canonical render, 32 samples, 256 px
  nuisance        re-render pipeline: 8 / 16 / 64 samples at 256 / 384 / 192 px (sampling noise, resolution)
                  screenshot pipeline: the correct scene from 6 jittered viewpoints
  errors          hue shift 0.04 / 0.10 / 0.25 · main object z-scale 0.9 / 0.75 / 0.5 · secondary object
                  missing · light energy ×0.6 / ×0.35 / ×0.15 · an extra object
                  each rendered canonically (re-render pipeline) and from one jittered viewpoint (screenshot)
plus determinism (two identical re-renders) and runtime. The main objects are exported as OBJ
for the mesh benchmark (realistic assets: Suzanne, chair, mug, lamp).
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
COMPOSITIONS = {0: "torus_cube", 1: "suzanne", 2: "still_life", 3: "chair", 4: "mug", 5: "desk_lamp"}
ERRORS = [("hue", 1 / 3, dict(hue=0.04)), ("hue", 2 / 3, dict(hue=0.10)), ("hue", 1.0, dict(hue=0.25)),
          ("proportion", 1 / 3, dict(zs=0.9)), ("proportion", 2 / 3, dict(zs=0.75)), ("proportion", 1.0, dict(zs=0.5)),
          ("missing_object", 1.0, dict(drop=1)), ("lighting", 1 / 3, dict(lf=0.6)), ("lighting", 2 / 3, dict(lf=0.35)),
          ("lighting", 1.0, dict(lf=0.15)), ("extra_object", 1.0, dict(extra=1))]
NUIS_RENDER = [(8, 256), (16, 384), (64, 192)]
JITTERS = [(8, 3, 1.0), (-8, -3, 1.05), (15, 0, 0.95), (-15, 5, 1.1), (22, -5, 0.92), (-22, 6, 1.12)]


def _scene(path, comp, hue=0.0, zs=1.0, drop=0, lf=1.0, extra=0, mesh_out=None, blender=None):
    args = [path, comp, hue, zs, drop, lf, extra] + ([mesh_out] if mesh_out else [])
    run_script(SCENE2, args, blender=blender)


def _metrics(ref, img_path, use_lpips):
    r = compare_images(ref, load_image(img_path), trim=True, use_lpips=use_lpips)
    out = dict(r.raw)
    out["score"] = score({"type": "image"}, r.m)
    out["composite"] = composite({"type": "image"}, r.m)
    return out


def run_render_bench(work: Path, use_lpips=True, res=256, blender=None, log=print, compositions=None):
    b = blender or find_blender()
    if b is None:
        log("Blender not found — render benchmark skipped")
        return None, None, {}
    work.mkdir(parents=True, exist_ok=True)
    rows, assets, det = [], {}, None
    rng = np.random.default_rng(11)
    for comp in (compositions or list(COMPOSITIONS)):
        name = COMPOSITIONS[comp]
        cdir = work / name
        cdir.mkdir(exist_ok=True)
        _scene(cdir / "ref.blend", comp, mesh_out=cdir / f"{name}.obj", blender=b)
        assets[name] = cdir / f"{name}.obj"
        t0 = time.perf_counter()
        run_blender(cdir / "ref.blend", cdir / "ref.png", None, res=res, samples=32, blender=b)
        t_render = time.perf_counter() - t0
        ref = load_image(cdir / "ref.png")
        if det is None:  # determinism on the first composition
            run_blender(cdir / "ref.blend", cdir / "ref_again.png", cdir / "a.obj", res=res, samples=32, blender=b)
            run_blender(cdir / "ref.blend", cdir / "ref_again2.png", cdir / "b.obj", res=res, samples=32, blender=b)
            x1 = np.asarray(Image.open(cdir / "ref.png").convert("RGB"), int)
            x2 = np.asarray(Image.open(cdir / "ref_again.png").convert("RGB"), int)
            det = {"max_abs_pixel_diff": int(np.abs(x1 - x2).max()), "identical_obj": (cdir / "a.obj").read_bytes() == (cdir / "b.obj").read_bytes(),
                   "render_seconds": round(t_render, 2)}
        for k, (spp, rr) in enumerate(NUIS_RENDER):
            p = cdir / f"nuis_rr{k}.png"
            run_blender(cdir / "ref.blend", p, None, res=rr, samples=spp, blender=b)
            rows.append({"reference": name, "pipeline": "re-render", "kind": "nuisance", "family": f"samples {spp} / {rr}px", "level": 0.0, **_metrics(ref, p, use_lpips)})
        for k, jit in enumerate(JITTERS):
            p = cdir / f"nuis_shot{k}.png"
            run_blender(cdir / "ref.blend", p, None, res=res, samples=16, blender=b, jitter=jit)
            rows.append({"reference": name, "pipeline": "screenshot", "kind": "nuisance", "family": "viewpoint", "level": 0.0, **_metrics(ref, p, use_lpips)})
        for k, (fam, lv, kw) in enumerate(ERRORS):
            bl = cdir / f"err{k}.blend"
            _scene(bl, comp, blender=b, **kw)
            p = cdir / f"err{k}_rr.png"
            run_blender(bl, p, None, res=res, samples=16, blender=b)
            rows.append({"reference": name, "pipeline": "re-render", "kind": "error", "family": fam, "level": lv, **_metrics(ref, p, use_lpips)})
            jit = JITTERS[int(rng.integers(len(JITTERS)))]
            p = cdir / f"err{k}_shot.png"
            run_blender(bl, p, None, res=res, samples=16, blender=b, jitter=jit)
            rows.append({"reference": name, "pipeline": "screenshot", "kind": "error", "family": fam, "level": lv, **_metrics(ref, p, use_lpips)})
        log(f"render bench: {name} ✓")
    return pd.DataFrame(rows), det, assets
