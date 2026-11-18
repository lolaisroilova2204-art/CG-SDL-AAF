"""Numbers and table rows of the manuscript from a benchmark run.

Usage: python scripts/paper/manuscript_numbers.py <benchmark_out> <out_dir>
(defaults: docs/benchmark_reference_run and the current folder); writes numbers.json and tables.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
B = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/benchmark_reference_run")
O = Path(sys.argv[2] if len(sys.argv) > 2 else ".")
X = pd.read_excel(B / "benchmark_tables.xlsx", sheet_name=None)
img = pd.read_csv(B / "bench_image.csv")
mesh = pd.read_csv(B / "bench_mesh.csv")
rend = pd.read_csv(B / "bench_render.csv")
rob = pd.read_csv(B / "bench_robustness.csv")
crop = pd.read_csv(B / "bench_crop_accuracy.csv")
fb = pd.read_csv(B / "bench_feedback.csv")
summ = json.loads((B / "benchmark_summary.json").read_text())
N = {}
T = {}
f3 = lambda v: f"{v:.3f}"
f1 = lambda v: f"{v:.1f}"
NM = {"n_jpeg": "JPEG quality 95–85", "n_noise": "Gaussian noise σ ≤ 3", "n_shift": "shift of 1–2 px", "n_resample": "resampling to 60 %",
      "n_reframe": "reframing (70 %, ≤ 10 %)", "scale": "down-scaling", "jpeg": "JPEG", "anisotropic_scale": "anisotropic scaling",
      "vertex_noise": "vertex noise", "part_removal": "part removal", "extra_part": "extra part", "rigid_rotation": "rigid rotation",
      "uniform_scale": "uniform scaling", "retriangulation": "re-triangulation", "micro_noise": "micro-noise", "retessellation": "re-tessellation",
      "missing_object": "missing object", "extra_object": "extra object"}
nm = lambda f: NM.get(f, f)

# ---------------- monotonicity
mono = X["image_monotonicity"]
mm = X["mesh_monotonicity"]
rows = []
for mode, lab in (("object", "Image (object)"), ("composition", "Image (composition)")):
    for _, r in mono[mono["mode"] == mode].iterrows():
        rows.append([lab, nm(r.family), f"{r.rho_level_score:.2f}", f1(r["score_L0.4"]), f1(r.score_L1)])
for _, r in mm.iterrows():
    rows.append(["Mesh", nm(r.family), f"{r.rho_level_score:.2f}", f1(r["score_L0.4"]), f1(r.score_L1)])
T["mono"] = rows
N["mono"] = {"image_object": mono[mono["mode"] == "object"][["family", "rho_level_score", "score_L0.4", "score_L1", "monotone_means", "max_increase"]].round(3).to_dict("records"),
             "image_composition": mono[mono["mode"] == "composition"][["family", "rho_level_score", "score_L0.4", "score_L1", "monotone_means", "max_increase"]].round(3).to_dict("records"),
             "mesh": mm[["family", "rho_level_score", "score_L0.4", "score_L1", "monotone_means", "max_increase"]].round(3).to_dict("records")}

# ---------------- nuisance
rows = []
nu = X["image_nuisance"]
for mode, lab in (("object", "Image (object)"), ("composition", "Image (composition)")):
    for _, r in nu[nu["mode"] == mode].iterrows():
        rows.append([lab, nm(r.family), str(int(r["count"])), f1(r["mean"]), f1(r["std"]), f1(r["min"])])
mi = X["mesh_invariance"]
for _, r in mi.iterrows():
    rows.append(["Mesh", nm(r.family), str(int(r.n)), f1(r.score_mean), f1(r.score_sd), f1(r.score_min)])
rr = rend[(rend.pipeline == "re-render") & (rend.kind == "nuisance")]
for fam, g in rr.groupby("family", sort=False):
    rows.append(["Scene (re-render)", fam, str(len(g)), f1(g.score.mean()), f1(g.score.std()), f1(g.score.min())])
T["nuis"] = rows
N["mesh_cd_align"] = mi[["family", "cd_mean", "cd_noalign_mean"]].round(4).to_dict("records")


# ---------------- discrimination
def auc_rows(t, sub="all"):
    t = t[t.objects == sub]
    out = []
    ms = X.get("image_min_severity_object")
    for _, r in t.iterrows():
        d = "—" if r.method == "Framework score" else f"{r['dAUC vs framework']:+.3f} {r['dAUC 95% CI']}"
        out.append([r.method, f"{r.AUC:.3f} {r['AUC 95% CI']}", d])
    return out


def minsev(key):
    m = X[key]
    col = [c for c in m.columns if c.startswith("min severity")][0]
    return dict(zip(m.method, m[col]))


for key, ms in (("image_auc_object", "image_min_severity_object"), ("image_auc_composition", "image_min_severity_composition"), ("mesh_auc", "mesh_min_severity")):
    t = X[key]
    mins = minsev(ms)
    rows = []
    for _, r in t[t.objects == "all"].iterrows():
        d = "—" if r.method == "Framework score" else f"{r['dAUC vs framework']:+.3f} {r['dAUC 95% CI']}"
        s = mins.get(r.method, np.nan)
        rows.append([r.method.replace("Framework composite P (unclipped)", "Composite *P* (unclipped)").replace("ΔE00", "Mean CIEDE2000"),
                     f"{r.AUC:.3f} {r['AUC 95% CI']}", d, "not reached" if not np.isfinite(s) else f"{s:g}"])
    T[key] = rows
    N[key] = t[["objects", "method", "AUC", "AUC 95% CI", "dAUC vs framework", "dAUC 95% CI", "differs", "LOOO min", "LOOO max", "n nuisance", "n error", "n objects"]].round(4).to_dict("records")

# ---------------- development vs held-out
rows = []
for key, lab in (("image_auc_object", "Image, object mode"), ("image_auc_composition", "Image, composition mode"), ("mesh_auc", "Mesh")):
    t = X[key]
    g = t[t.method == "Framework score"].set_index("objects")
    rows.append([lab, f"{g.loc['development','AUC']:.3f} {g.loc['development','AUC 95% CI']}", f"{g.loc['held-out','AUC']:.3f} {g.loc['held-out','AUC 95% CI']}",
                 f"{g.loc['all','LOOO min']:.3f}–{g.loc['all','LOOO max']:.3f}"])
R = X["render_auc"]
for pipe, lab in (("re-render", "Scene, re-render"), ("screenshot", "Scene, render screenshot"), ("viewport", "Scene, viewport screenshot")):
    g = R[(R.pipeline == pipe) & (R.method == "Framework score")].set_index("objects")
    rows.append([lab, f"{g.loc['development','AUC']:.3f} {g.loc['development','AUC 95% CI']}", f"{g.loc['held-out','AUC']:.3f} {g.loc['held-out','AUC 95% CI']}",
                 f"{g.loc['all','LOOO min']:.3f}–{g.loc['all','LOOO max']:.3f}"])
T["heldout"] = rows

# ---------------- render pipelines
meths = ["Framework score", "Framework composite P (unclipped)", "PSNR", "MS-SSIM", "SSIM (fg-weighted)", "FLIP", "SNCD", "Edge F1", "ΔE00", "Colour histogram"]
pipes = ["viewport", "screenshot", "re-render", "re-render 512px"]
rows = []
for m in meths:
    row = [m.replace("Framework composite P (unclipped)", "Composite *P*").replace("ΔE00", "Mean CIEDE2000")]
    for p in pipes:
        g = R[(R.pipeline == p) & (R.method == m) & (R.objects == "all")]
        row.append(f"{float(g.AUC.iloc[0]):.3f}" if len(g) else "—")
    rows.append(row)
T["render"] = rows
N["render_auc"] = R[R.objects == "all"][["pipeline", "method", "AUC", "AUC 95% CI", "dAUC vs framework", "dAUC 95% CI"]].round(4).to_dict("records")
sc = rend.groupby(["pipeline", "kind"])["score"].agg(["mean", "std", "min", "max", "count"]).round(2).reset_index()
N["render_scores_kind"] = sc.to_dict("records")
N["render_scores_family"] = rend.groupby(["pipeline", "kind", "family"])["score"].agg(["mean", "std", "min", "count"]).round(2).reset_index().to_dict("records")

# ---------------- feedback
EXP = {"fb_edge_missing": "contours", "fb_edge_extra": "contours", "fb_region": "region", "fb_shape": "shape", "fb_color": "colour",
       "fb_mesh_outlier": "outliers", "fb_mesh_shape": "shape", "fb_mesh_extra": "extra parts", "fb_mesh_missing": "missing parts", "fb_open": "holes"}
pc = lambda v: "—" if not np.isfinite(v) else f"{100 * v:.0f} %"
rows = []
for chan, lab in (("image (object mode)", "*Image, object mode*"), ("image (composition mode)", "*Image, composition mode*"), ("mesh", "*Mesh*")):
    rows.append([lab, "", "", "", ""])
    for _, r in fb[fb.channel == chan].iterrows():
        if r.family.startswith("nuisance") or r.family.startswith("occlusion:"):
            continue
        ex = []
        for k in r.expected.split(", "):
            if EXP.get(k) and EXP[k] not in ex:
                ex.append(EXP[k])
        # the reference run keeps the rates of the rules as specified before the run ("initial"); a fresh run of
        # version 1.4.0 has only the corrected rules, which are then shown in both columns
        ini = r.rate_initial if "rate_initial" in r else r.rate
        ini_pen = r.rate_penalised_initial if "rate_penalised_initial" in r else r.rate_penalised
        rows.append([f"{nm(r.family)} ({', '.join(ex)})", pc(ini), str(int(r.n_penalised)), pc(ini_pen), pc(r.rate_penalised)])
T["feedback"] = rows
N["feedback"] = fb.round(4).to_dict("records")
# misattribution: vertex noise -> extra parts etc.
mis = {}
for fam, g in mesh[(mesh.kind == "error") & (mesh.level >= 0.2)].groupby("family"):
    keys = g.feedback.fillna("").str.split(";")
    mis[fam] = {k: float(keys.map(lambda f: k in f).mean()) for k in ["fb_mesh_extra", "fb_mesh_missing", "fb_open", "fb_mesh_shape", "fb_mesh_outlier"]}
N["mesh_feedback_keys"] = mis
mis = {}
for mode in ("object", "composition"):
    gm = img[(img["mode"] == mode) & (img.kind == "error") & (img.level >= 0.2)]
    for fam, g in gm.groupby("family"):
        keys = g.feedback.fillna("").str.split(";")
        mis[f"{mode}:{fam}"] = {k: float(keys.map(lambda f: k in f).mean()) for k in ["fb_shape", "fb_color", "fb_edge_missing", "fb_edge_extra", "fb_region"]}
        mis[f"{mode}:{fam}"]["any"] = float((g.feedback.fillna("") != "").mean())
        mis[f"{mode}:{fam}"]["penalised_share"] = float((g.score < 86).mean())
N["image_feedback_keys"] = mis

# ---------------- robustness
ca = crop.groupby(["content", "theme"]).agg(n=("detected", "size"), detected=("detected", "mean"), iou=("iou", "mean"),
                                           iou09=("iou", lambda v: float((v >= 0.9).mean()))).reset_index()
N["crop"] = ca.round(4).to_dict("records")
rb = rob.groupby(["test", "method"])["score"].agg(["mean", "std", "min", "max", "count"]).round(2).reset_index()
N["robust"] = rb.to_dict("records")
rv = rob.groupby(["test", "method", "variant"])["score"].agg(["mean", "min", "count"]).round(2).reset_index()
N["robust_variant"] = rv.to_dict("records")
N["determinism"] = summ["determinism"]
N["runtime"] = summ["runtime"]
N["runtime_framework_only"] = {"image_s": 0.071, "mesh_s": 0.037, "n_images": 72, "n_meshes": 24, "script": "scripts/time_framework.py"}
N["meta"] = summ["meta"]

# ---------------- sensitivity
S = X["sensitivity_constants"]
N["sensitivity"] = S.round(4).to_dict("records")

# ---------------- counts
N["counts"] = {"img_refs": int(img.reference.nunique()), "mesh_refs": int(mesh.reference.nunique()), "scenes": int(rend.reference.nunique()),
               "img_cases": int(len(img)), "mesh_cases": int(len(mesh)), "render_cases": int(len(rend)),
               "clean_images": int((crop.content == "clean image").sum())}
json.dump(N, open(O / "numbers.json", "w"), indent=1, default=str, ensure_ascii=False)
json.dump(T, open(O / "tables.json", "w"), indent=1, ensure_ascii=False)
print("ok", N["counts"])

# ---------------- sensitivity (Table XIV): largest |dAUC| per group of constants, smallest share of unchanged grades
SC = X["sensitivity_constants"]
CH = ["image (object mode)", "image (composition mode)", "mesh", "scene (re-render)"]
GROUPS = [("weights (one at a time, x0.5 / x1.5)", "Weights ×0.5 / ×1.5, one at a time (8)"),
          ("weights (200 random Dirichlet draws)", "Weights, Dirichlet draws (200)"),
          ("mapping P0 (0.30 / 0.50)", "*P*₀ = 0.30 / 0.50 (2)"), ("mapping P1 (0.95 / 1.00)", "*P*₁ = 0.95 / 1.00 (2)"),
          ("colour constant of m_color (6 / 24)", "Colour constant of (4): 6 / 24 (2)"),
          ("mesh mapping cd_floor (x0.5 / x1.5)", "Chamfer floor of (7) ×0.5 / ×1.5 (2)"),
          ("mesh mapping cd_sigma (x0.5 / x1.5)", "Chamfer scale of (7) ×0.5 / ×1.5 (2)"),
          ("mesh mapping h_floor (x0.5 / x1.5)", "H95 floor of (7) ×0.5 / ×1.5 (2)"),
          ("mesh mapping h_sigma (x0.5 / x1.5)", "H95 scale of (7) ×0.5 / ×1.5 (2)"),
          ("F-score threshold tau (0.05 / 0.15)", "τ = 0.05 / 0.15 (2)"),
          ("topology tolerance (2.5 % / 10 % of edges)", "Topology tolerance ×0.5 / ×2 (2)")]
rows = [["AUC with the default constants"] + [f"{SC[(SC.channel == c) & (SC.constants == 'default')]['AUC default'].iloc[0]:.3f}" for c in CH]]
for g, lab in GROUPS:
    r = [lab]
    for c in CH:
        x = SC[(SC.channel == c) & (SC.constants == g)]
        r.append("—" if not len(x) else f"{x['max |dAUC|'].iloc[0]:.3f}")
    rows.append(r)
rows.append(["Smallest share of unchanged grades"] +
            [f"{100 * SC[(SC.channel == c) & (SC.constants != 'default')]['grades unchanged (min)'].min():.0f} %" for c in CH])
T["sens"] = rows
json.dump(T, open(O / "tables.json", "w"), indent=1, ensure_ascii=False)
