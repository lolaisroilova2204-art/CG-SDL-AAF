"""Draw the framework architecture figure (Fig. 1 of the manuscript)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "docs"
fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.set(xlim=(0, 100), ylim=(-7, 68)); ax.axis("off")
C = {"srl": "#e8eefc", "in": "#f2f4f7", "img": "#e6f4ec", "mesh": "#fdf1dc", "out": "#f3e8fd"}


def box(x, y, w, h, title, lines, col):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=col, ec="#475467", lw=.8))
    ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=7.6, weight="bold")
    ax.text(x + w / 2, y + h - 5.6, "\n".join(lines), ha="center", va="top", fontsize=6.2, linespacing=1.35)


def arrow(a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8, lw=.8, color="#475467"))


box(1, 44, 30, 16, "Forethought", ["goal · planned time", "expected score (self-prediction)"], C["srl"])
box(35, 44, 30, 16, "Performance", ["submission: .blend / .fbx /", "screenshot / OBJ·STL·PLY"], C["srl"])
box(69, 44, 30, 16, "Self-reflection", ["formative feedback,", "calibration of self-assessment"], C["srl"])
box(35, 26, 30, 13, "Blender headless", ["canonical camera · fixed engine/seed", "render.png + export.obj"], C["in"])
box(1, 4, 31, 19, "Perceptual image metrics", ["object / composition mode", "foreground-weighted SSIM, MS-SSIM", "LPIPS (optional)", "CIEDE2000 colour · edge F1"], C["img"])
box(35, 4, 30, 19, "Geometric mesh metrics", ["seeded area sampling (n=4096)", "PCA pose + ICP (Kabsch)", "Chamfer · Hausdorff · H95", "F-score@τ · topology vs ref."], C["mesh"])
box(68, 4, 31, 19, "Scoring & validation", ["weighted model → NNLS calibration", "rule-based bilingual feedback", "benchmark · sensitivity", "dataset CSV/XLSX, figures"], C["out"])
arrow((31.6, 52), (34.4, 52)); arrow((65.6, 52), (68.4, 52))
arrow((50, 43.4), (50, 39.6)); arrow((44, 25.4), (22, 23.6)); arrow((50, 25.4), (50, 23.6))
arrow((65.6, 13), (67.4, 13))
ax.add_patch(FancyArrowPatch((16, 3.4), (83, 3.4), arrowstyle="-|>", mutation_scale=8, lw=.8, color="#475467", connectionstyle="arc3,rad=0.12"))
ax.add_patch(FancyArrowPatch((83.5, 23.6), (84, 43.4), arrowstyle="-|>", mutation_scale=8, lw=.8, color="#2b59d9", connectionstyle="arc3,rad=0"))
ax.text(85.5, 33.5, "score +\nfeedback", fontsize=6.2, color="#2b59d9", va="center")
ax.add_patch(FancyArrowPatch((84, 60.8), (16, 60.8), arrowstyle="-|>", mutation_scale=8, lw=.8, ls="--", color="#98a2b3", connectionstyle="arc3,rad=0.12"))
ax.text(50, 67.6, "next attempt (SRL cycle)", ha="center", fontsize=6.2, color="#667085")
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig_architecture.{ext}", dpi=300, bbox_inches="tight")
print("saved", OUT / "fig_architecture.png")
