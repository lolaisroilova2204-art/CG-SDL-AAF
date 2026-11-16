import os
from pathlib import Path

import pytest

from cgaaf.blender import find_blender
from cgaaf.course import Course
from cgaaf.demo import make_demo, scene
from cgaaf.mesh_metrics import torus
from cgaaf.reports import analytics_report, validation_report


def test_demo_pipeline(tmp_path):
    c = make_demo(tmp_path / "c", n_students=10, log=lambda s: None)
    df = c.dataset()
    assert len(df) >= 20 and df["score"].between(0, 100).all() and df["expert_mean"].notna().sum() >= 20
    rep, out = validation_report(c, "T1_render")
    assert rep["n_rated"] >= 10 and (out / "fig_scatter_auto_vs_expert.png").exists() and (out / "report_en.md").exists()
    T, out2 = analytics_report(c)
    assert len(T) == 2


def test_batch_mode_without_srl_data(tmp_path):
    c = Course.create(tmp_path / "c", "t")
    scene().save(tmp_path / "ref.png")
    c.save_task({"id": "T1", "type": "image", "trim": False}, ref_image=tmp_path / "ref.png")
    d = c.root / "submissions" / "S001" / "T1" / "1"
    d.mkdir(parents=True)
    scene(dx=10).save(d / "work.png")
    assert len(c.pending()) == 1
    r = c.evaluate_attempt(d, use_lpips=False)
    assert r["image"]["source"] == "screenshot" and r["score"] > 70 and not c.pending()


@pytest.mark.skipif(find_blender() is None, reason="Blender not available")
def test_blend_rerender_pipeline(tmp_path):
    import subprocess, sys, textwrap
    mk = textwrap.dedent(f"""
        import bpy, sys
        bpy.ops.wm.read_factory_settings(use_empty=True)
        r = float(sys.argv[-2]); out = sys.argv[-1]
        bpy.ops.mesh.primitive_torus_add(major_radius=1, minor_radius=r)
        bpy.ops.wm.save_as_mainfile(filepath=out)
    """)
    (tmp_path / "mk.py").write_text(mk)
    b = find_blender()
    for name, r in [("ref.blend", 0.35), ("good.blend", 0.34), ("bad.blend", 0.15)]:
        cmd = ([b[1], "-b", "--factory-startup", "--python", str(tmp_path / "mk.py"), "--"] if b[0] == "exe" else [b[1], str(tmp_path / "mk.py"), "--"]) + [str(r), str(tmp_path / name)]
        subprocess.run(cmd, check=True, capture_output=True)
    c = Course.create(tmp_path / "c", "t")
    c.save_task({"id": "T1", "type": "both", "render": {"mode": "canonical", "res": 256, "engine": "CYCLES", "samples": 4}}, ref_blend=tmp_path / "ref.blend")
    good = c.submit("S001", "T1", [tmp_path / "good.blend"], {"goal": "g", "minutes": 10, "pred": 80}, use_lpips=False)
    bad = c.submit("S002", "T1", [tmp_path / "bad.blend"], {"goal": "g", "minutes": 10, "pred": 80}, use_lpips=False)
    assert good["image"]["source"] == "render" and good["mesh"]["source"] == "blender_export"
    assert good["score"] > bad["score"] + 20
