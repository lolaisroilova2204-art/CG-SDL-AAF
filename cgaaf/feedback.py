"""Rule-based formative feedback (Uzbek / English)."""
from .i18n import region_name, tr


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
            F.append(("fb_edge_missing", []) if r["edge_r"] < r["edge_p"] else ("fb_edge_extra", []))
        w = img.get("worst")
        if w and w[1] < 0.8:
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
        if r["boundary_edges"]:
            F.append(("fb_open", [r["boundary_edges"]]))
        if r["nonmanifold_edges"]:
            F.append(("fb_nonmanifold", [r["nonmanifold_edges"]]))
        if r["degenerate_faces"]:
            F.append(("fb_degenerate", [r["degenerate_faces"]]))
        if len(F) == n0:
            F.append(("fb_good_mesh", []))
    pred = (result.get("plan") or {}).get("pred")
    if pred is not None:
        s = round(result["score"])
        d = s - pred
        F.append(("fb_cal" if abs(d) <= 10 else "fb_over" if d < 0 else "fb_under", [pred, s]))
    return F


def render_feedback(items, lang="uz"):
    out = []
    for k, args in items:
        args = [region_name(int(a[2:]), lang) if isinstance(a, str) and a.startswith("#R") else a for a in args]
        out.append(tr(k, lang, *args))
    return out
