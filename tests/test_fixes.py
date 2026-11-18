"""Tests for the review fixes: UI cropping, helper objects, animation, scales, size, independence."""
import subprocess
import textwrap

import numpy as np
import pytest
import trimesh
from PIL import Image

from cgaaf.archive import GRADE5_MIDPOINTS, rescale
from cgaaf.blender import find_blender
from cgaaf.course import Course
from cgaaf.demo import scene
from cgaaf.image_metrics import compare_images
from cgaaf.mesh_metrics import compare_meshes, load_mesh, prepare_reference, torus
from cgaaf.reports import rated
from cgaaf.scoring import score
from cgaaf.stats import agreement
from cgaaf.ui_crop import detect_viewport, mock_ui


@pytest.mark.parametrize("layout", [0, 1, 2])
def test_ui_crop_recovers_viewport(layout):
    r = scene()
    ui = mock_ui(r, layout, seed=layout)
    W, H = r.size
    left, top = [0, 46, 70][layout], [34, 30, 60][layout]
    assert detect_viewport(ui) == (left + 2, top + 28, left + 2 + W, top + 28 + H)
    a = np.asarray(r, float)
    assert score({"type": "image"}, compare_images(a, np.asarray(ui, float), trim=False, use_lpips=False, crop="auto").m) > 98
    assert score({"type": "image"}, compare_images(a, np.asarray(ui, float), trim=False, use_lpips=False).m) < 20


def test_ui_crop_leaves_clean_renders():
    for im in (scene(), scene(noise=10), scene(hue=2.0)):
        assert detect_viewport(im) is None


def test_floor_components_dropped(tmp_path):
    floor = lambda s: trimesh.creation.box(extents=[s, s, 0.02]).apply_translation([0, 0, -0.4])
    trimesh.util.concatenate([torus(), floor(4)]).export(tmp_path / "ref.obj")
    ref = prepare_reference(load_mesh(tmp_path / "ref.obj", drop_planes=True))
    for s in (0, 3, 6):
        p = tmp_path / f"s{s}.obj"
        (trimesh.util.concatenate([torus(), floor(s)]) if s else torus()).export(p)
        assert score({"type": "mesh"}, compare_meshes(load_mesh(p, drop_planes=True), ref).m) > 99


def test_size_check_optional():
    ref, s = prepare_reference(torus(), return_scale=True)
    big = compare_meshes(torus(R=2, r=0.7), ref, ref_scale=s)
    assert big.raw["size_ratio"] == pytest.approx(2, rel=0.02) and big.m["size"] < 0.01
    assert score({"type": "mesh"}, compare_meshes(torus(R=2, r=0.7), ref).m) > 99      # size not checked
    assert score({"type": "mesh"}, big.m) < 85                                          # size checked


def test_expert_rescaling():
    assert rescale(2, "linear", 2, 5) == 0 and rescale(5, "linear", 2, 5) == 100
    assert rescale(3, "grade5") == GRADE5_MIDPOINTS[3]
    with pytest.raises(ValueError):
        rescale(7, "grade5")


def test_one_per_student_and_cluster_bootstrap():
    import pandas as pd
    df = pd.DataFrame({"student": ["a", "a", "b", "c", "d", "e", "f", "g", "h", "i"], "task_id": ["t"] * 10, "attempt": [1, 2, 1, 1, 1, 1, 1, 1, 1, 1],
                       "expert_mean": np.linspace(30, 90, 10), "score": np.linspace(32, 88, 10), "score_current": np.linspace(32, 88, 10)})
    d = rated(df)
    assert len(d) == 9 and d[d.student == "a"].attempt.iloc[0] == 2
    A = agreement(d["auto"], d["expert_mean"], clusters=d["student"].values)
    assert A["bootstrap"].startswith("cluster") and A["icc21_ci"][0] <= A["icc21"] <= A["icc21_ci"][1]


def test_multiple_screenshots_best_match(tmp_path):
    c = Course.create(tmp_path / "c", "t")
    scene().save(tmp_path / "ref.png")
    c.save_task({"id": "T", "type": "image", "trim": False}, ref_image=tmp_path / "ref.png")
    d = c.root / "submissions" / "S1" / "T" / "1"
    d.mkdir(parents=True)
    scene(dx=150, cube=False).save(d / "a_wrong_view.png")
    mock_ui(scene(dx=5), 1).save(d / "b_good_view_with_ui.png")
    r = c.evaluate_attempt(d, use_lpips=False)
    assert r["image"]["n_images"] == 2 and r["image"]["file"] == "b_good_view_with_ui.png" and r["image"]["crop"] and r["score"] > 85
    assert r["env"]["numpy"] and r["env"]["algo"] == "CG-SDL-AAF 1.4"


@pytest.mark.skipif(find_blender() is None, reason="Blender not available")
def test_animation_and_helpers_blender(tmp_path):
    mk = textwrap.dedent('''
        import bpy, sys, math
        a = sys.argv[sys.argv.index("--") + 1:]
        out, spin, floor = a[0], float(a[1]), float(a[2])
        bpy.ops.wm.read_factory_settings(use_empty=True)
        sc = bpy.context.scene; sc.frame_start, sc.frame_end = 1, 24
        if floor > 0:
            bpy.ops.mesh.primitive_plane_add(size=floor, location=(0, 0, -0.6))
        bpy.ops.mesh.primitive_torus_add(major_radius=1, minor_radius=0.35)
        t = bpy.context.object
        t.keyframe_insert("rotation_euler", frame=1); t.rotation_euler = (math.radians(90) * spin, 0, 0); t.keyframe_insert("rotation_euler", frame=24)
        bpy.ops.wm.save_as_mainfile(filepath=out)
        print("AAF_OK")
    ''')
    (tmp_path / "mk.py").write_text(mk)
    from cgaaf.blender import run_script
    for name, spin, floor in [("ref", 1, 4), ("good", 1, 9), ("still", 0, 4)]:
        run_script(tmp_path / "mk.py", [tmp_path / f"{name}.blend", spin, floor])
    c = Course.create(tmp_path / "c", "t")
    c.save_task({"id": "A", "type": "both", "frames": 3, "render": {"mode": "canonical", "res": 128, "engine": "CYCLES", "samples": 4}}, ref_blend=tmp_path / "ref.blend")
    assert len(c.task("A")["reference"]["images"]) == 3
    good = c.submit("S1", "A", [tmp_path / "good.blend"], {"goal": "g", "minutes": 1, "pred": 50}, use_lpips=False)
    still = c.submit("S2", "A", [tmp_path / "still.blend"], {"goal": "g", "minutes": 1, "pred": 50}, use_lpips=False)
    assert good["score"] > 97 and good["image"]["frames"] == 3
    assert still["image"]["frame_scores"][0] > 97 and min(still["image"]["frame_scores"]) < 50 and still["score"] < good["score"] - 15
