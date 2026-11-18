"""v1.4 feedback corrections found by the feedback validation of the benchmark."""
from cgaaf.feedback import feedback_keys

BASE = {"msssim": 0.9, "lpips": None, "color": 0.9, "deltaE00": 1.0, "edge_f1": 0.9, "edge_p": 0.9, "edge_r": 0.9}


def test_no_contours_count_as_missing():
    raw = {**BASE, "edge_f1": 0.0, "edge_p": 0.0, "edge_r": 0.0}  # strong blur: no contour detected at all
    keys = feedback_keys(image_raw=raw, worst=(4, 0.9), image_score=50)
    assert "fb_edge_missing" in keys and "fb_edge_extra" not in keys


def test_penalised_image_is_never_silent():
    keys = feedback_keys(image_raw=BASE, worst=(2, 0.85), image_score=80)
    assert keys == ["fb_region"]
    assert feedback_keys(image_raw=BASE, worst=(2, 0.85), image_score=95) == []


def test_penalised_mesh_is_never_silent():
    raw = {"cd": 0.05, "h95": 0.1, "precision": 0.9, "recall": 0.9, "topo_relative": True,
           "excess_boundary": 0, "excess_nonmanifold": 0, "excess_degenerate": 0}
    m = {"cd": 0.85, "h95": 0.85, "fscore": 0.9, "topo": 1.0}
    assert feedback_keys(mesh_raw=raw, mesh_m=m, mesh_score=80) == ["fb_mesh_shape"]
    assert feedback_keys(mesh_raw=raw, mesh_m=m, mesh_score=95) == []
