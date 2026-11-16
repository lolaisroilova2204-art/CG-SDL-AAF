"""One command -> every table, figure and results paragraph needed for the manuscript
(Variant 2: controlled benchmark + retrospective validation on archived submissions)."""
from __future__ import annotations

import datetime as dt
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from . import ALGO_VERSION, __version__
from .reports import results_paragraph, validation_report


def dataset_table(course):
    df = course.dataset()
    tasks = {t["id"]: t for t in course.tasks()}
    rows = []
    for tid, g in df.groupby("task_id"):
        t = tasks.get(tid, {})
        rows.append({"assignment": tid, "title": t.get("title_en") or t.get("title_uz") or tid, "type": t.get("type"),
                     "students": g["student"].nunique(), "attempts": len(g), "expert-rated": int(g["expert_mean"].notna().sum()),
                     "two raters": int((g["expert1"].notna() & g["expert2"].notna()).sum()),
                     "screenshots": int((g["image_source"] == "screenshot").sum()), "Blender re-renders": int((g["image_source"] == "render").sum()),
                     "mesh files / exports": int(g["mesh_source"].notna().sum()), "retrospective": bool(g["retrospective"].all())})
    T = pd.DataFrame(rows)
    if len(T):
        tot = {c: (T[c].sum() if T[c].dtype != object and c not in ("retrospective",) else "") for c in T.columns}
        tot.update({"assignment": "Total", "title": "", "type": "", "students": df["student"].nunique(), "retrospective": bool(df["retrospective"].all())})
        T = pd.concat([T, pd.DataFrame([tot])], ignore_index=True)
    return T


def make_paper_bundle(course=None, bench_dir=None, out=None, log=print):
    """course=None -> technical-validation paper (benchmark only)."""
    if course is None and not bench_dir:
        raise ValueError("give a course folder and/or a benchmark folder")
    base = course.root / "results" if course is not None else Path(bench_dir).parent
    out = Path(out or base / ("paper_" + dt.datetime.now().strftime("%Y%m%d_%H%M")))
    figd = out / "figures"
    figd.mkdir(parents=True, exist_ok=True)
    tables, paras, figs = {}, [], []
    if course is not None:
        tables["T1_dataset"] = dataset_table(course)
    # ---- validation per task + pooled
    agree_rows, base_tabs, abl_tabs = [], [], []
    for tid in ([t["id"] for t in course.tasks()] + [None] if course is not None else []):
        rep, vdir = validation_report(course, tid, "stored", out / "_validation" / (tid or "pooled"), lang="en")
        if "agreement" not in rep:
            log(f"{tid or 'pooled'}: {rep.get('error')}")
            continue
        A = rep["agreement"]
        agree_rows.append({"assignment": tid or "Pooled", "n": A["n"], "Pearson r [95% CI]": f"{A['pearson_r']:.3f} [{A['pearson_lo']:.3f}, {A['pearson_hi']:.3f}]",
                           "Spearman ρ": round(A["spearman_rho"], 3), "ICC(2,1) [95% CI]": f"{A['icc21']:.3f}" + (f" [{A['icc21_ci'][0]:.3f}, {A['icc21_ci'][1]:.3f}]" if "icc21_ci" in A else ""),
                           "MAE": round(A["mae"], 2), "RMSE": round(A["rmse"], 2), "BA bias [LoA]": f"{A['ba_bias']:.2f} [{A['ba_lo']:.2f}, {A['ba_hi']:.2f}]",
                           "QWK": round(A["qwk"], 3), "exact grade %": round(100 * A["exact_grade"], 1),
                           "inter-rater ICC(2,1)": round(rep["inter_rater"]["icc21"], 3) if "inter_rater" in rep else None})
        if rep.get("baselines"):
            base_tabs.append(pd.DataFrame(rep["baselines"]).assign(assignment=tid))
        if rep.get("ablation"):
            abl_tabs.append(pd.DataFrame(rep["ablation"]).assign(assignment=tid))
        for k, f in (rep.get("figures") or {}).items():
            dst = figd / f"validation_{tid or 'pooled'}_{Path(f).stem}.png"
            shutil.copy2(f, dst)
            pdf = Path(f).with_suffix(".pdf")
            if pdf.exists():
                shutil.copy2(pdf, dst.with_suffix(".pdf"))
            figs.append(dst.name)
        paras.append(f"**{tid or 'Pooled'}.** " + results_paragraph(rep))
    if agree_rows:
        tables["T4_agreement"] = pd.DataFrame(agree_rows)
    if base_tabs:
        tables["T5_baselines"] = pd.concat(base_tabs, ignore_index=True)
    if abl_tabs:
        tables["T6_ablation"] = pd.concat(abl_tabs, ignore_index=True)
    # ---- screenshot vs re-render on real data
    df = course.dataset() if course is not None else None
    d = df[df["expert_mean"].notna()] if df is not None and len(df) else None
    if d is not None and len(d) and d["image_source"].nunique() > 1:
        from .stats import agreement
        rows = []
        for s, g in d.groupby("image_source"):
            if len(g) >= 4:
                A = agreement(g["score"], g["expert_mean"], boot=False)
                rows.append({"image source": s, "n": A["n"], "r": round(A["pearson_r"], 3), "ICC(2,1)": round(A["icc21"], 3), "MAE": round(A["mae"], 2)})
        if rows:
            tables["T7_source_real_data"] = pd.DataFrame(rows)
    # ---- benchmark
    bench_par = None
    if bench_dir and Path(bench_dir, "benchmark_tables.xlsx").exists():
        bx = pd.read_excel(Path(bench_dir) / "benchmark_tables.xlsx", sheet_name=None)
        names = {"image_monotonicity": "B2a_image_monotonicity", "image_nuisance": "B2b_image_nuisance", "mesh_invariance": "B3a_mesh_invariance",
                 "mesh_monotonicity": "B3b_mesh_monotonicity", "image_sensitivity": "B2c_image_sensitivity", "mesh_sensitivity": "B3c_mesh_sensitivity",
                 "image_auc": "B4a_image_auc", "image_min_severity": "B4b_image_min_severity", "mesh_auc": "B4c_mesh_auc", "mesh_min_severity": "B4d_mesh_min_severity",
                 "render_auc": "B5a_render_auc", "render_scores": "B5b_render_scores", "robustness": "B6_robustness",
                 "image_detectability": "X_image_detectability", "mesh_detectability": "X_mesh_detectability"}
        for k, v in bx.items():
            tables[names.get(k, "bench_" + k)] = v
        js = json.loads(Path(bench_dir, "benchmark_summary.json").read_text())
        bench_par = js.get("paragraph")
        for f in sorted(Path(bench_dir).glob("fig_bench_*")):
            shutil.copy2(f, figd / f.name)
            if f.suffix == ".png":
                figs.append(f.name)
    arch = Path(__file__).resolve().parents[1] / "docs" / "fig_architecture.png"
    if arch.exists():
        shutil.copy2(arch, figd / "fig_architecture.png")
    # ---- write
    with pd.ExcelWriter(out / "paper_tables.xlsx") as xw:
        for k, v in sorted(tables.items()):
            v.to_excel(xw, sheet_name=k[:31], index=False)
    title = course.meta.get("name") if course is not None else "technical validation (controlled benchmark)"
    md = [f"# Manuscript results bundle — {title}", "",
          f"Software: CG-SDL-AAF {__version__} ({ALGO_VERSION}) · generated {dt.datetime.now():%Y-%m-%d %H:%M}"
          + (" · ⚠ SYNTHETIC COURSE — not for publication" if course is not None and course.meta.get("synthetic") else ""), ""]
    if bench_dir and Path(bench_dir, "benchmark_summary.json").exists():
        meta = json.loads(Path(bench_dir, "benchmark_summary.json").read_text()).get("meta", {})
        md += [f"Benchmark references — images: {', '.join(meta.get('images', []))}; meshes: {', '.join(meta.get('meshes', []))}; "
               f"Blender compositions: {', '.join(meta.get('render_compositions', []))}; LPIPS: {meta.get('lpips')}.", ""]
    caps = {"T1_dataset": "Table. Archived dataset used for retrospective validation",
            "B2a_image_monotonicity": "Table. Image channel — monotonicity of the score over 9 error families",
            "B2b_image_nuisance": "Table. Image channel — scores under nuisance variation (JPEG, noise, 1–2 px shift, resampling)",
            "B2c_image_sensitivity": "Table. Image channel — |Spearman ρ| between severity and each metric",
            "B3a_mesh_invariance": "Table. Mesh channel — invariance to rotation, scaling, re-triangulation, micro-noise",
            "B3b_mesh_monotonicity": "Table. Mesh channel — monotonicity over 5 geometric error families",
            "B3c_mesh_sensitivity": "Table. Mesh channel — |Spearman ρ| between severity and each metric",
            "B4a_image_auc": "Table. Image channel — framework vs baselines: AUC of nuisance vs errors (severity ≥ 0.2)",
            "B4b_image_min_severity": "Table. Image channel — smallest error severity detected with AUC ≥ 0.9",
            "B4c_mesh_auc": "Table. Mesh channel — framework vs baselines: AUC of nuisance vs errors (severity ≥ 0.2)",
            "B4d_mesh_min_severity": "Table. Mesh channel — smallest error severity detected with AUC ≥ 0.9",
            "B5a_render_auc": "Table. Render-level errors in six Blender compositions — AUC by pipeline (screenshot vs re-render)",
            "B5b_render_scores": "Table. Render-level scores by pipeline, kind and family",
            "B6_robustness": "Table. Robustness to application UI in screenshots and to auxiliary objects",
            "T2d_bench_image_sensitivity": "Table II(d). Benchmark — |Spearman ρ| between severity and each image metric",
            "T2e_bench_mesh_sensitivity": "Table II(e). Benchmark — |Spearman ρ| between severity and each mesh metric",
            "T3_bench_viewpoint": "Table III. Screenshots with viewpoint variation vs standardised Blender re-rendering",
            "T3b_bench_robustness": "Table III(b). Robustness to application UI in screenshots and to auxiliary objects",
            "T4_agreement": "Table IV. Agreement between automated and expert scores", "T5_baselines": "Table V. Comparison with single-metric baselines (LOOCV)",
            "T6_ablation": "Table VI. Ablation of the metric set (LOOCV)", "T7_source_real_data": "Table VII. Agreement by image source on archived data"}
    for k in sorted(tables):
        if k.startswith("X_"):
            continue  # long detectability series: in paper_tables.xlsx and the figures only
        md += [f"## {caps.get(k, k)}", "", tables[k].round(3).to_markdown(index=False), ""]
    if bench_par:
        md += ["## Results paragraph — benchmark", "", bench_par, ""]
    if paras:
        md += ["## Results paragraphs — retrospective validation", ""] + paras
    md += ["", "## Figures", ""] + [f"* figures/{f}" for f in figs + ["fig_architecture.png"]]
    (out / "paper_results_en.md").write_text("\n".join(md), encoding="utf-8")
    log(f"paper bundle → {out}")
    return out
