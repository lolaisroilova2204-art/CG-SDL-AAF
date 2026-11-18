"""Rule-based formative feedback (Uzbek / English)."""
from .i18n import region_name, tr
from .scoring import score as _score

TOP_GRADE = 86  # scores below this value are penalised (grade 4 or lower); such a result is never left unexplained


def _channel_score(part: dict, m: dict, kind: str):
    """Channel score for the 'no silent penalty' rule: taken from the result if present, else with default weights."""
    if part.get("score") is not None:
        return part["score"]
    return _score({"type": kind}, m) if m else None


def feedback_items(result: dict):
    """Language-neutral list of (key, args)."""
    F = []
    img, mesh, m = result.get("image"), result.get("mesh"), result.get("m", {})
    if img:
        r = img["raw"]
        n0 = len(F)
        if r["msssim"] < 0.75:
            F.append(("fb_shape", []))
        if r.get("lpips") is not None and r["lpips"] > 0.35:
            F.append(("fb_perceptual", [round(r["lpips"], 3)]))
        if r["color"] < 0.7:
            F.append(("fb_color", [round(r["deltaE00"], 1)]))
        if r["edge_f1"] < 0.6:
            # v1.4: ties (typically no contour detected at all, recall = precision = 0) count as missing contours
            F.append(("fb_edge_missing", []) if r["edge_r"] <= r["edge_p"] else ("fb_edge_extra", []))
        w = img.get("worst")
        if w and w[1] < 0.8:
            F.append(("fb_region", ["#R%d" % w[0]]))
        if len(F) == n0 and w:  # v1.4: a penalised result always names at least the region with the largest difference
            s_img = _channel_score(img, m, "image")
            if s_img is not None and s_img < TOP_GRADE:
                F.append(("fb_region", ["#R%d" % w[0]]))
        if len(F) == n0:
            F.append(("fb_good_img", []))
        if img.get("frames", 1) > 1:
            F.append(("fb_frame", [img["frames"], img["worst_frame"] + 1, round(min(img["frame_scores"]))]))
        if img.get("n_images", 1) > 1:
            F.append(("fb_multi", [img["n_images"]]))
        if img.get("crop") and not img.get("manual_crop"):
            F.append(("fb_crop", []))
        if img.get("source") == "screenshot" and result.get("blend_possible"):
            F.append(("fb_source_shot", []))
    if mesh:
        r = mesh["raw"]
        n0 = len(F)
        if m.get("cd", 1) < 0.7:
            F.append(("fb_mesh_shape", [round(r["cd"], 3)]))
        if m.get("h95", 1) < 0.7:
            F.append(("fb_mesh_outlier", [round(r["h95"], 3)]))
        if r["precision"] < r["recall"] - 0.1:
            F.append(("fb_mesh_extra", [round(100 * r["precision"])]))
        elif r["recall"] < r["precision"] - 0.1:
            F.append(("fb_mesh_missing", [round(100 * r["recall"])]))
        if "size" in m and m["size"] < 0.7:
            F.append(("fb_size", [round(r["size_ratio"], 2)]))
        rel = bool(r.get("topo_relative"))  # v1.4: defects counted only in excess of the reference
        nb = r["excess_boundary"] if rel else r["boundary_edges"]
        nn = r["excess_nonmanifold"] if rel else r["nonmanifold_edges"]
        nd = r["excess_degenerate"] if rel else r["degenerate_faces"]
        if nb:
            F.append(("fb_open", [int(round(nb))]))
        if nn:
            F.append(("fb_nonmanifold", [int(round(nn))]))
        if nd:
            F.append(("fb_degenerate", [int(round(nd))]))
        if len(F) == n0:  # v1.4: a penalised result always receives at least the overall-shape message
            s_mesh = _channel_score(mesh, m, "mesh")
            if s_mesh is not None and s_mesh < TOP_GRADE:
                F.append(("fb_mesh_shape", [round(r["cd"], 3)]))
        if len(F) == n0:
            F.append(("fb_good_mesh", []))
    pred = (result.get("plan") or {}).get("pred")
    if pred is not None:
        s = round(result["score"])
        d = s - pred
        F.append(("fb_cal" if abs(d) <= 10 else "fb_over" if d < 0 else "fb_under", [pred, s]))
    return F


PROBLEM_KEYS = {"fb_shape", "fb_perceptual", "fb_color", "fb_edge_missing", "fb_edge_extra", "fb_region", "fb_mesh_shape",
                "fb_mesh_outlier", "fb_mesh_extra", "fb_mesh_missing", "fb_size", "fb_open", "fb_nonmanifold", "fb_degenerate"}


def feedback_keys(image_raw=None, worst=None, mesh_raw=None, mesh_m=None, image_score=None, mesh_score=None):
    """Keys of the problem messages that the rules would issue (used to validate the feedback)."""
    res = {"m": mesh_m or {}}
    if image_raw is not None:
        res["image"] = {"raw": image_raw, "worst": worst, "score": image_score}
    if mesh_raw is not None:
        res["mesh"] = {"raw": mesh_raw, "score": mesh_score}
    return sorted({k for k, _ in feedback_items(res)} & PROBLEM_KEYS)


def render_feedback(items, lang="uz"):
    out = []
    for k, args in items:
        args = [region_name(int(a[2:]), lang) if isinstance(a, str) and a.startswith("#R") else a for a in args]
        out.append(tr(k, lang, *args))
    return out
