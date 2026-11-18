import pandas as pd
import pytest

from cgaaf.archive import import_mapping, make_mapping_template
from cgaaf.benchmark import builtin_images, builtin_meshes, run_image_bench, run_mesh_bench, summarise
from cgaaf.course import Course
from cgaaf.demo import scene
from cgaaf.mesh_metrics import torus
from cgaaf.paper import make_paper_bundle


def test_archive_roundtrip_and_paper(tmp_path):
    arch = tmp_path / "archive"
    for i in range(12):
        d = arch / f"S{i:03d}"
        d.mkdir(parents=True)
        scene(dx=i * 8, hue=i * 0.08).save(d / "render.png")
        torus(r=0.35 - 0.01 * i).export(d / "model.obj")
    (arch / "S000" / "old.max").write_bytes(b"x")
    df = make_mapping_template(arch, tmp_path / "map.xlsx")
    assert (df["status"].str.startswith("CONVERT")).sum() == 1
    df = df[df.status == "ok"].copy()
    df["task_id"] = df["files"].map(lambda f: "T_img" if f.endswith(".png") else "T_mesh")
    df["expert1"] = [5 - (int(s[1:]) % 4) for s in df.student_id]  # 5-point scale
    df["expert2"] = df["expert1"]
    df.to_excel(tmp_path / "map.xlsx", index=False)
    c = Course.create(tmp_path / "c", "retro")
    scene().save(tmp_path / "ref.png"); torus().export(tmp_path / "ref.obj")
    c.save_task({"id": "T_img", "type": "image", "trim": False}, ref_image=tmp_path / "ref.png")
    c.save_task({"id": "T_mesh", "type": "mesh"}, ref_mesh=tmp_path / "ref.obj")
    s = import_mapping(c, tmp_path / "map.xlsx", expert_max=5, log=lambda m: None)
    assert s["attempts"] == 24 and s["experts"] == 24
    for d in c.pending():
        try:
            c.evaluate_attempt(d, use_lpips=False)
        except Exception as e:
            raise AssertionError(f"{d}: {[f.name for f in d.iterdir()]} {e}")
    ds = c.dataset()
    assert ds["retrospective"].all() and ds["expert_mean"].max() == 100 and ds["img_psnr"].notna().sum() == 12
    out = make_paper_bundle(c, None, tmp_path / "paper", log=lambda m: None)
    x = pd.read_excel(out / "paper_tables.xlsx", sheet_name=None)
    assert "T1_dataset" in x and "T4_agreement" in x and "T5_baselines" in x


def test_benchmark_monotone_and_invariant():
    img = run_image_bench({"A": builtin_images()["scene_A"]}, repeats=1, use_lpips=False, log=lambda s: None)
    mesh = run_mesh_bench({"torus": builtin_meshes()["torus"]}, repeats=1, log=lambda s: None)
    S = summarise(img, mesh)
    assert (S["mesh_invariance"]["score_min"] > 95).all()
    assert (S["mesh_monotonicity"]["rho_level_score"] < -0.7).all()
    im = S["image_monotonicity"]
    comp = im[im["mode"] == "composition"].set_index("family")
    assert comp.loc[["translation", "scale", "brightness", "noise"], "rho_level_score"].lt(-0.9).all()
    obj = im[im["mode"] == "object"].set_index("family")
    assert "translation" not in obj.index  # placement is not assessed in object mode
    # object mode: a render-like object on a uniform background may be moved and resized in the frame
    from PIL import Image, ImageDraw
    obj = Image.new("RGB", (256, 256), (60, 60, 64))
    dr = ImageDraw.Draw(obj)
    dr.ellipse((78, 70, 178, 190), fill=(180, 90, 40)); dr.rectangle((100, 150, 160, 200), fill=(40, 120, 190))
    im2 = run_image_bench({"obj": obj}, repeats=1, use_lpips=False, log=lambda s: None, modes=("object",))
    nu = im2[(im2.kind == "nuisance") & (im2.family == "n_reframe")]
    assert len(nu) == 3 and nu["score"].min() > 90
