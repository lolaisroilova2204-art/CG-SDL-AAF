"""Tests for the technical-validation (variant 1) components."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import mannwhitneyu

from cgaaf.blender import find_blender
from cgaaf.discrimination import auc, cohen_d, compare_methods


def test_auc_matches_mann_whitney():
    rng = np.random.default_rng(0)
    a, b = rng.normal(1, 1, 40), rng.normal(0, 1, 50)
    a[:5] = b[:5]  # ties
    U = mannwhitneyu(a, b).statistic
    assert auc(a, b) == pytest.approx(U / (len(a) * len(b)))
    assert auc([1, 1], [1, 1]) == 0.5 and auc([2, 3], [0, 1]) == 1.0


def test_compare_methods_orientation_and_ci():
    df = pd.DataFrame({"reference": ["r1"] * 6 + ["r2"] * 6, "kind": ["nuisance"] * 3 + ["error"] * 3 + ["nuisance"] * 3 + ["error"] * 3,
                       "level": [0, 0, 0, .5, .7, 1] * 2, "sim": [1, .99, .98, .5, .4, .3] * 2, "dist": [0, .01, .02, .5, .6, .7] * 2})
    T = compare_methods(df, [("sim", "similarity", 1), ("dist", "distance", -1)], B=200)
    assert (T["AUC"] == 1.0).all() and cohen_d([1, .99, .98], [.5, .4, .3]) > 2


@pytest.mark.skipif(find_blender() is None, reason="Blender not available")
def test_render_bench_one_composition(tmp_path):
    from cgaaf.bench_render import run_render_bench
    df, det, assets = run_render_bench(tmp_path, use_lpips=False, res=128, log=lambda s: None, compositions=[1], hires=0)
    assert det["max_abs_pixel_diff"] == 0 and det["identical_obj"] and assets["suzanne"]["ref"].exists()
    assert assets["suzanne"]["tess1"].exists() and assets["suzanne"]["tess2"].exists()
    rr = df[df.pipeline == "re-render"]
    assert rr[rr.kind == "nuisance"]["score"].min() > rr[rr.kind == "error"]["score"].median()
    # canonical re-rendering neutralises student render settings (engine, view transform, camera, film)
    assert rr[(rr.kind == "nuisance") & (rr.family == "student render settings")]["score"].min() > 90
    assert set(df.pipeline) == {"re-render", "screenshot", "viewport"}


def test_paired_auc_difference():
    rng = np.random.default_rng(1)
    rows = []
    for r in range(6):
        for k in range(20):
            rows.append({"reference": f"r{r}", "kind": "nuisance", "level": 0, "good": rng.normal(1, .1), "weak": rng.normal(.6, .3)})
            rows.append({"reference": f"r{r}", "kind": "error", "level": .5, "good": rng.normal(0, .1), "weak": rng.normal(.4, .3)})
    T = compare_methods(pd.DataFrame(rows), [("good", "Framework score", 1), ("weak", "weak", 1)], B=300).set_index("method")
    assert T.loc["weak", "differs"] and T.loc["weak", "dAUC vs framework"] < 0 and T.loc["weak", "LOOO max"] <= 1
