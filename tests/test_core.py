import numpy as np
import pytest
from scipy import stats as ss

from cgaaf.demo import scene
from cgaaf.image_metrics import compare_images
from cgaaf.mesh_metrics import compare_meshes, prepare_reference, topology, torus
from cgaaf.scoring import calibrate, score
from cgaaf.stats import agreement, icc


def arr(img):
    return np.asarray(img, dtype=np.float64)


def test_identical_image_scores_100():
    a = arr(scene())
    r = compare_images(a, a, trim=False, use_lpips=False)
    assert r.raw["ssim"] == pytest.approx(1, abs=1e-6) and r.raw["edge_f1"] == 1
    assert score({"type": "image"}, r.m) == pytest.approx(100)


def test_image_degradations_ordered():
    ref = arr(scene())
    s_small = score({"type": "image"}, compare_images(ref, arr(scene(dx=6)), trim=False, use_lpips=False).m)
    s_color = score({"type": "image"}, compare_images(ref, arr(scene(hue=2.0)), trim=False, use_lpips=False).m)
    s_big = score({"type": "image"}, compare_images(ref, arr(scene(dx=120, cube=False, blur=4)), trim=False, use_lpips=False).m)
    assert s_small > s_color > s_big
    blank = np.full_like(ref, 200.0)
    assert score({"type": "image"}, compare_images(ref, blank, trim=False, use_lpips=False).m) < 20


def test_mesh_invariance_and_topology():
    ref = prepare_reference(torus())
    same = compare_meshes(torus(nu=40, nv=20, rot=1.1, scale=3.0), ref)
    assert same.raw["cd"] < 0.035 and same.raw["fscore"] > 0.99 and same.raw["watertight"]
    thin = compare_meshes(torus(r=0.2), ref)
    assert thin.raw["cd"] > 2 * same.raw["cd"]
    t = topology(torus(hole=True))
    assert t["boundary_edges"] > 0 and not t["watertight"]


def test_icc_against_reference_values():
    # Shrout & Fleiss (1979) example data, 6 targets x 4 judges: ICC(2,1)=0.29, ICC(3,1)=0.71
    M = np.array([[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]], float)
    r = icc(M)
    assert r["icc21"] == pytest.approx(0.29, abs=0.01) and r["icc31"] == pytest.approx(0.71, abs=0.01)


def test_agreement_matches_scipy():
    rng = np.random.default_rng(0)
    e = rng.uniform(20, 95, 40)
    a = e + rng.normal(0, 5, 40)
    A = agreement(a, e, boot=False)
    assert A["pearson_r"] == pytest.approx(ss.pearsonr(a, e).statistic)
    assert A["mae"] == pytest.approx(np.abs(a - e).mean())


def test_calibration_recovers_weights():
    rng = np.random.default_rng(1)
    M = [{"a": x, "b": y} for x, y in rng.random((60, 2))]
    y = [10 + 50 * m["a"] + 30 * m["b"] for m in M]
    c = calibrate(M, y, ["a", "b"])
    assert c["b"]["a"] == pytest.approx(50, abs=1e-6) and c["b"]["b"] == pytest.approx(30, abs=1e-6) and c["mae_loo"] < 1e-6
