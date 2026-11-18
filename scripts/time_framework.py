"""Time the framework metrics alone (no baselines) on the renders and meshes of a benchmark run.

Usage: python scripts/time_framework.py <benchmark_out>/_render
The benchmark CSVs record the time per case with all baselines (FLIP, SNCD, D2, ...) computed;
this script reports the cost of an ordinary assessment.
"""
import glob
import sys
import time

import numpy as np
import trimesh
from PIL import Image

from cgaaf.image_metrics import compare_images
from cgaaf.mesh_metrics import compare_meshes, prepare_reference, topology

root = sys.argv[1] if len(sys.argv) > 1 else "benchmark_out/_render"
ti, tm = [], []
for c in sorted(glob.glob(root + "/*")):
    ref = sorted(glob.glob(c + "/ref*.png"))
    if not ref:
        continue
    R = np.asarray(Image.open(ref[0]).convert("RGB"), float)
    for f in sorted(glob.glob(c + "/err*_rr.png"))[:6]:
        S = np.asarray(Image.open(f).convert("RGB"), float)
        t0 = time.perf_counter()
        compare_images(R, S, trim=True, use_lpips=False, baselines=False)
        ti.append(time.perf_counter() - t0)
rng = np.random.default_rng(1)
for o in sorted(glob.glob(root + "/*/*.obj"))[:24]:
    m = trimesh.load(o, force="mesh")
    if m.is_empty:
        continue
    ref, rt = prepare_reference(m, "pca_icp"), topology(m)
    stu = m.copy()
    stu.apply_transform(trimesh.transformations.random_rotation_matrix(rng.random(3)))
    t0 = time.perf_counter()
    compare_meshes(stu, ref, "pca_icp", ref_topo=rt, baselines=False)
    tm.append(time.perf_counter() - t0)
print(f"images: {len(ti)} pairs, {np.mean(ti):.3f} s per assessment")
print(f"meshes: {len(tm)} meshes, {np.mean(tm):.3f} s per assessment")
