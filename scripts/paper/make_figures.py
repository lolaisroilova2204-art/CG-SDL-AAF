"""Figures 2-7 of the manuscript from a benchmark run.

Usage: python scripts/paper/make_figures.py <benchmark_out> <figure_dir>
(defaults: docs/benchmark_reference_run and docs/paper_figures)
"""
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
B = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/benchmark_reference_run")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "docs/paper_figures")
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 7, "axes.titlesize": 7.5,
                     "axes.labelsize": 7, "legend.fontsize": 6.2, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "axes.linewidth": 0.6, "lines.linewidth": 1.0, "savefig.dpi": 400})
MK = ["o", "s", "^", "v", "D", "P", "X", "*", "<", ">"]
LS = ["-", "--", "-.", ":"]
COL = plt.get_cmap("tab10").colors
img = pd.read_csv(B / "bench_image.csv")
mesh = pd.read_csv(B / "bench_mesh.csv")
rend = pd.read_csv(B / "bench_render.csv")
X = pd.read_excel(B / "benchmark_tables.xlsx", sheet_name=None)
NAMES = {"n_reframe": "reframing", "scale": "down-scaling", "jpeg": "JPEG", "greyscale": "greyscale"}


STYLE_ORDER = ["translation", "rotation", "scale", "hue", "brightness", "blur", "noise", "jpeg", "occlusion", "greyscale"]


def curves(ax, df, fams, title, ylab=True):
    for fam in fams:
        k = STYLE_ORDER.index(fam)  # the same marker, line and colour for a family in both panels
        g = df[(df.family == fam) & df.kind.isin(["error", "identity"])]
        m = g.groupby("level")["score"].mean()
        ax.plot(m.index, m.values, marker=MK[k % 10], ms=2.8, ls=LS[k % 4], color=COL[k % 10], label=NAMES.get(fam, fam))
    ax.set(ylim=(0, 103), xlim=(-0.02, 1.02), title=title)
    ax.set_xlabel("Error severity")
    if ylab:
        ax.set_ylabel("Framework score")
    ax.grid(alpha=0.25, lw=0.4)


# Fig. 2: image channel, both modes (stacked panels)
fig, axs = plt.subplots(2, 1, figsize=(3.3, 4.6))
ERR_OBJ = ["rotation", "hue", "brightness", "blur", "noise", "jpeg", "occlusion", "greyscale"]
ERR_COMP = ["translation", "rotation", "scale", "hue", "brightness", "blur", "noise", "jpeg", "occlusion", "greyscale"]
curves(axs[0], img[img["mode"] == "object"], ERR_OBJ, "(a) Object mode (placement not assessed)")
curves(axs[1], img[img["mode"] == "composition"], ERR_COMP, "(b) Composition mode (frame assessed)")
h, l = axs[1].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.005))
fig.tight_layout(rect=(0, 0.085, 1, 1), h_pad=0.8)
fig.savefig(OUT / "fig2_image_curves.png"); plt.close(fig)

# Fig. 3: mesh channel
fig, ax = plt.subplots(figsize=(3.3, 2.75))
ERRS = ["anisotropic_scale", "vertex_noise", "hole", "part_removal", "extra_part"]
NUIS = ["rigid_rotation", "uniform_scale", "retriangulation", "micro_noise"]
for k, fam in enumerate(ERRS + NUIS):
    g = mesh[mesh.family == fam]
    m = g.groupby("level")["score"].mean()
    ax.plot(m.index, m.values, marker=MK[k % 10], ms=2.8, ls="--" if fam in NUIS else "-", color=COL[k % 10],
            label=fam.replace("_", " ") + (" (nuisance)" if fam in NUIS else ""))
ax.set(ylim=(0, 103), xlim=(-0.02, 1.02), xlabel="Perturbation severity", ylabel="Framework score")
ax.grid(alpha=0.25, lw=0.4)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
fig.tight_layout()
fig.savefig(OUT / "fig3_mesh_curves.png", bbox_inches="tight"); plt.close(fig)


def detect(ax, D, methods, title):
    for k, (lab, show) in enumerate(methods):
        g = D[D.method == lab].sort_values("level")
        fw = lab.startswith("Framework score")
        ax.plot(g["level"], g["AUC"], marker=MK[k % 10], ms=2.8 if not fw else 3.4, lw=1.8 if fw else 0.9,
                ls="-" if fw else LS[(k + 1) % 4], color="black" if fw else COL[k % 10], label=show)
    ax.axhline(0.5, color="#888", lw=0.6, ls=":")
    ax.axhline(0.9, color="#bbb", lw=0.5, ls="--")
    ax.set(ylim=(0.35, 1.02), xlim=(0.08, 1.02), xlabel="Error severity", ylabel="AUC (nuisance vs error)", title=title)
    ax.grid(alpha=0.25, lw=0.4)


IM = [("Framework score", "Framework score"), ("Framework composite P (unclipped)", "Composite P"), ("PSNR", "PSNR"),
      ("MS-SSIM", "MS-SSIM"), ("FLIP", "FLIP"), ("Edge F1", "Edge F1"), ("ΔE00", "Mean ΔE00"), ("SNCD", "SNCD")]
fig, ax = plt.subplots(figsize=(3.3, 2.7))
detect(ax, X["image_detectability_object"], IM, "")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)
fig.tight_layout(); fig.savefig(OUT / "fig4_image_detectability.png", bbox_inches="tight"); plt.close(fig)

MM = [("Framework score", "Framework score"), ("Framework composite P (unclipped)", "Composite P"), ("Chamfer, PCA+ICP", "Chamfer, PCA+ICP"),
      ("Hausdorff-95", "H95"), ("F-score@0.1", "F-score"), ("D2 shape distribution", "D2"), ("Chamfer, ICP only", "Chamfer, ICP only"),
      ("Chamfer, no alignment", "Chamfer, no alignment")]
fig, ax = plt.subplots(figsize=(3.3, 2.7))
detect(ax, X["mesh_detectability"], MM, "")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)
fig.tight_layout(); fig.savefig(OUT / "fig5_mesh_detectability.png", bbox_inches="tight"); plt.close(fig)

# Fig. 6: scene-level AUC by pipeline (horizontal grouped bars)
R = X["render_auc"]
R = R[R.objects == "all"]
meths = ["Framework score", "PSNR", "MS-SSIM", "FLIP", "Edge F1", "ΔE00", "Colour histogram", "SSIM (fg-weighted)"]
pipes = [("viewport", "viewport screenshot", "#d9d9d9", "////"), ("screenshot", "render screenshot", "#9e9e9e", ""),
         ("re-render", "canonical re-render", "#2f4b7c", "")]
fig, ax = plt.subplots(figsize=(3.3, 3.2))
y = np.arange(len(meths))[::-1]
hgt = 0.26
for k, (p, lab, c, hatch) in enumerate(pipes):
    g = R[R.pipeline == p].set_index("method").reindex(meths)
    ax.barh(y + (1 - k) * hgt, g["AUC"], hgt, color=c, hatch=hatch, edgecolor="black", lw=0.3, label=lab)
ax.axvline(0.5, color="#555", lw=0.6, ls=":")
ax.set(yticks=y, xlim=(0, 1.02), xlabel="AUC (correct vs erroneous scenes)")
ax.set_yticklabels([m.replace("Colour histogram", "Colour hist.").replace("SSIM (fg-weighted)", "SSIM (fg-w.)").replace("ΔE00", "Mean ΔE00") for m in meths])
ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.14), ncol=3, frameon=False, fontsize=5.8)
fig.tight_layout(); fig.savefig(OUT / "fig6_render_auc.png", bbox_inches="tight"); plt.close(fig)

# Fig. 7: framework scores of correct and erroneous scenes by pipeline
fams = ["nuisance", "hue", "proportion", "lighting", "missing_object", "extra_object"]
fig, ax = plt.subplots(figsize=(3.3, 2.5))
rng = np.random.default_rng(3)
for k, (p, lab, c, hatch) in enumerate(pipes):
    for i, f in enumerate(fams):
        v = rend[(rend.pipeline == p) & ((rend.kind == "nuisance") if f == "nuisance" else (rend.family == f))]["score"]
        xs = np.full(len(v), i + (k - 1) * 0.27) + rng.normal(0, 0.03, len(v))
        ax.scatter(xs, v, s=3.5, color={"viewport": "#bdbdbd", "screenshot": "#636363", "re-render": "#2f4b7c"}[p],
                   marker=["^", "s", "o"][k], lw=0, label=lab if i == 0 else None)
ax.set(xticks=range(len(fams)), ylim=(-2, 103), ylabel="Framework score")
ax.set_xticklabels(["correct\n(nuisance)", "hue\nshift", "propor-\ntion", "light-\ning", "missing\nobject", "extra\nobject"], fontsize=6.0)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False, markerscale=2, fontsize=5.8)
ax.grid(alpha=0.25, lw=0.4, axis="y")
fig.tight_layout(); fig.savefig(OUT / "fig7_viewpoint.png", bbox_inches="tight"); plt.close(fig)
print("figures written")
