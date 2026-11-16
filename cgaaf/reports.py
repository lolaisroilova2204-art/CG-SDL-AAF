"""Validation, ablation and learning-analytics reports with publication figures."""
from __future__ import annotations

import datetime as dt
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import ALGO_VERSION
from .scoring import IMG_KEYS, MESH_KEYS, calibrate, grade5, keys_for
from .stats import agreement, fmt_p, icc, paired_gain, qwk

M_LABEL = {"ssim": "SSIM", "msssim": "MS-SSIM", "lpips": "LPIPS", "color": "Colour", "edge": "Edge F1",
           "cd": "Chamfer", "h95": "H95", "fscore": "F-score", "topo": "Topology"}
RAW_COLS = [("img_ssim", "SSIM", +1), ("img_msssim", "MS-SSIM", +1), ("img_lpips", "LPIPS distance", -1), ("img_deltaE00", "ΔE00", -1),
            ("img_color_hist", "Colour hist. BC", +1), ("img_edge_f1", "Edge F1", +1), ("mesh_cd", "Chamfer distance", -1),
            ("mesh_hd", "Hausdorff distance", -1), ("mesh_h95", "Hausdorff-95", -1), ("mesh_fscore", "F-score@0.1", +1), ("m_topo", "Topology score", +1)]


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight"})
    return plt


def _save(fig, out, name):
    fig.savefig(out / f"{name}.png")
    fig.savefig(out / f"{name}.pdf")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return out / f"{name}.png"


def rated(df, model="stored", one_per_student=True):
    """Expert-rated attempts. one_per_student: keep only the latest rated attempt of each
    student on each assignment, so observations are independent within an assignment."""
    d = df[np.isfinite(df["expert_mean"].astype(float))].copy()
    if one_per_student and len(d):
        d = d.sort_values("attempt").groupby(["student", "task_id"], as_index=False).tail(1)
    d["auto"] = d["score_current" if model == "current" else "score"].astype(float)
    return d


def ablation(d, keys):
    """LOOCV NNLS models: each metric alone, all metrics, all-minus-one."""
    keys = [k for k in keys if "m_" + k in d and d["m_" + k].notna().all()]
    M = d[["m_" + k for k in keys]].rename(columns=lambda c: c[2:]).to_dict("records")
    y = d["expert_mean"].astype(float).values
    rows = []
    for name, ks in [("all", keys)] + [(k + " only", [k]) for k in keys] + [("all − " + k, [x for x in keys if x != k]) for k in keys if len(keys) > 1]:
        c = calibrate(M, y, ks)
        rows.append({"model": name, "metrics": "+".join(M_LABEL[k] for k in ks), "r_loo": c["r_loo"], "mae_loo": c["mae_loo"], "n": c["n"]})
    return pd.DataFrame(rows)


BASELINES = {"image": [("img_psnr", "PSNR"), ("img_ssim_plain", "SSIM (unweighted)"), ("img_msssim", "MS-SSIM"), ("img_lpips", "LPIPS"), ("img_deltaE00", "ΔE00")],
             "mesh": [("mesh_cd_noalign", "Chamfer (no alignment)"), ("mesh_cd", "Chamfer (PCA+ICP)"), ("mesh_h95", "Hausdorff-95")]}


def _loo_linear(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    out = np.empty(len(y))
    for i in range(len(y)):
        m = np.arange(len(y)) != i
        b, a = np.polyfit(x[m], y[m], 1)
        out[i] = np.clip(a + b * x[i], 0, 100)
    return out


def _paired_boot_mae_diff(p1, p2, y, B=2000, seed=1):
    rng = np.random.default_rng(seed)
    p1, p2, y = map(lambda v: np.asarray(v, float), (p1, p2, y))
    d = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        d.append(np.abs(p1[i] - y[i]).mean() - np.abs(p2[i] - y[i]).mean())
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def baseline_comparison(d, task_type):
    """Framework (default and LOOCV-calibrated) vs single-metric baselines (LOOCV linear)."""
    y = d["expert_mean"].astype(float).values
    keys = [k for k in keys_for(task_type) if "m_" + k in d and d["m_" + k].notna().all()]
    M = d[["m_" + k for k in keys]].rename(columns=lambda c: c[2:]).to_dict("records")
    fw = np.array(calibrate(M, y, keys)["loo_pred"])
    preds = {"Framework (default weights)": d["score_default"].astype(float).values, "Framework (calibrated, LOOCV)": fw}
    kinds = ["image", "mesh"] if task_type == "both" else [task_type]
    for kd in kinds:
        for col, lab in BASELINES[kd]:
            if col in d and d[col].notna().all() and d[col].std() > 0:
                preds[lab + " (LOOCV linear)"] = _loo_linear(d[col].values, y)
    rows = []
    for name, p in preds.items():
        rec = {"method": name, "n": len(y), "r": float(np.corrcoef(p, y)[0, 1]) if np.std(p) > 0 else np.nan, "MAE": float(np.abs(p - y).mean()),
               "RMSE": float(np.sqrt(((p - y) ** 2).mean())), "ICC(2,1)": float(icc(np.c_[p, y])["icc21"])}
        if name != "Framework (calibrated, LOOCV)":
            lo, hi = _paired_boot_mae_diff(fw, p, y)
            rec["ΔMAE (calibrated framework − method) [95% CI]"] = f"[{lo:.2f}, {hi:.2f}]"
        rows.append(rec)
    return pd.DataFrame(rows)


def validation_report(course, task_id=None, model="stored", out=None, lang="en", one_per_student=True):
    plt = _plt()
    df = course.dataset(anonymise=True)
    if task_id:
        df = df[df["task_id"] == task_id]
    out = Path(out or course.root / "results" / ("validation_" + (task_id or "all") + "_" + dt.datetime.now().strftime("%Y%m%d_%H%M")))
    out.mkdir(parents=True, exist_ok=True)
    d = rated(df, model, one_per_student)
    rep = {"version": ALGO_VERSION, "task": task_id or "all", "model": model, "n_attempts": int(len(df)), "n_rated": int(len(d)),
           "n_students": int(d["student"].nunique()) if len(d) else 0, "one_per_student": one_per_student}
    if len(d) < 4:
        rep["error"] = "need >= 4 expert-rated attempts"
        (out / "report.json").write_text(json.dumps(rep, indent=2))
        return rep, out
    A = agreement(d["auto"], d["expert_mean"], clusters=d["student"].values)
    rep["agreement"] = A
    both = d[np.isfinite(d["expert1"].astype(float)) & np.isfinite(d["expert2"].astype(float))]
    if len(both) >= 4:
        e1, e2 = both["expert1"].astype(float).values, both["expert2"].astype(float).values
        rep["inter_rater"] = {"n": int(len(both)), "icc21": float(icc(np.c_[e1, e2])["icc21"]), "pearson_r": float(np.corrcoef(e1, e2)[0, 1]),
                              "mae": float(np.abs(e1 - e2).mean()), "qwk": float(qwk([grade5(x) for x in e1], [grade5(x) for x in e2])[0])}
    from scipy.stats import spearmanr
    pm = []
    for col, label, sign in RAW_COLS:
        if col in d and d[col].notna().sum() >= 4:
            x = d[[col, "expert_mean"]].dropna().astype(float)
            r = spearmanr(x[col], x["expert_mean"])
            pm.append({"metric": label, "n": int(len(x)), "rho": float(r.statistic), "p": float(r.pvalue), "expected_sign": "+" if sign > 0 else "−"})
    rep["per_metric"] = pm
    if task_id and len(d) >= 10:
        ttype = d["task_type"].iloc[0]
        ab = ablation(d, keys_for(ttype))
        rep["ablation"] = ab.to_dict("records")
        ab.to_csv(out / "ablation.csv", index=False)
        bc = baseline_comparison(d, ttype)
        rep["baselines"] = bc.to_dict("records")
        bc.to_csv(out / "baselines.csv", index=False)
    # ---------------- figures
    figs = {}
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    ax.scatter(d["expert_mean"], d["auto"], s=16, alpha=.75, color="#2b59d9", edgecolor="none")
    ax.plot([0, 100], [0, 100], "--", color="#888", lw=1)
    ax.set(xlim=(0, 100), ylim=(0, 100), xlabel="Expert score", ylabel="Automated score",
           title=f"r = {A['pearson_r']:.3f}, ICC(2,1) = {A['icc21']:.3f}")
    figs["scatter"] = _save(fig, out, "fig_scatter_auto_vs_expert")
    m_ = (d["auto"] + d["expert_mean"]) / 2
    df_ = d["auto"] - d["expert_mean"]
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    ax.scatter(m_, df_, s=16, alpha=.75, color="#2b59d9", edgecolor="none")
    for y, ls, lab in [(A["ba_bias"], "-", f"bias {A['ba_bias']:.2f}"), (A["ba_hi"], "--", f"+1.96 SD {A['ba_hi']:.1f}"), (A["ba_lo"], "--", f"−1.96 SD {A['ba_lo']:.1f}")]:
        ax.axhline(y, ls=ls, color="#d2433f" if ls == "--" else "#333", lw=1)
        ax.text(100, y, lab, ha="right", va="bottom", fontsize=7)
    ax.set(xlim=(0, 100), xlabel="Mean of automated and expert", ylabel="Automated − expert", title="Bland–Altman")
    figs["ba"] = _save(fig, out, "fig_bland_altman")
    if pm:
        fig, ax = plt.subplots(figsize=(4.2, 0.28 * len(pm) + 0.8))
        lab = [p["metric"] for p in pm][::-1]
        val = [p["rho"] for p in pm][::-1]
        ax.barh(lab, val, color=["#1d9a5b" if v >= 0 else "#d2433f" for v in val])
        ax.axvline(0, color="#333", lw=.8)
        ax.set(xlim=(-1, 1), xlabel="Spearman ρ with expert score")
        figs["per_metric"] = _save(fig, out, "fig_per_metric_correlation")
    if rep.get("ablation"):
        ab = pd.DataFrame(rep["ablation"])
        fig, ax = plt.subplots(figsize=(4.4, 0.26 * len(ab) + 0.8))
        ax.barh(ab["model"][::-1], ab["mae_loo"][::-1], color="#2b59d9")
        ax.set(xlabel="LOOCV MAE (points)", title="Ablation (NNLS, leave-one-out)")
        figs["ablation"] = _save(fig, out, "fig_ablation")
    C = np.array(A["confusion"])
    fig, ax = plt.subplots(figsize=(2.8, 2.6))
    ax.imshow(C, cmap="Blues")
    for i, j in itertools.product(range(4), range(4)):
        ax.text(j, i, C[i, j], ha="center", va="center", color="white" if C[i, j] > C.max() / 2 else "black")
    ax.set(xticks=range(4), yticks=range(4), xticklabels=[2, 3, 4, 5], yticklabels=[2, 3, 4, 5], xlabel="Automated grade", ylabel="Expert grade",
           title=f"QWK = {A['qwk']:.3f}")
    figs["confusion"] = _save(fig, out, "fig_grade_confusion")
    rep["figures"] = {k: str(v) for k, v in figs.items()}
    with pd.ExcelWriter(out / "tables.xlsx") as xw:
        pd.DataFrame([{k: v for k, v in A.items() if k != "confusion"}]).T.rename(columns={0: "value"}).to_excel(xw, sheet_name="agreement")
        pd.DataFrame(pm).to_excel(xw, sheet_name="per_metric", index=False)
        if rep.get("ablation"):
            pd.DataFrame(rep["ablation"]).to_excel(xw, sheet_name="ablation", index=False)
        if rep.get("baselines"):
            pd.DataFrame(rep["baselines"]).to_excel(xw, sheet_name="baselines", index=False)
        d.to_excel(xw, sheet_name="rated_attempts", index=False)
    (out / "report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=float))
    (out / f"report_{lang}.md").write_text(markdown(rep, lang), encoding="utf-8")
    return rep, out


def results_paragraph(rep):
    A = rep["agreement"]
    r_ = abs(A["pearson_r"])
    strength = "strongly" if r_ >= 0.7 else "moderately" if r_ >= 0.5 else "weakly"
    s = (f"Across {A['n']} expert-rated submissions from {rep.get('n_students', '?')} students, the automated scores correlated {strength} with the expert scores "
         f"(Pearson r = {A['pearson_r']:.3f}, 95% CI [{A['pearson_lo']:.3f}, {A['pearson_hi']:.3f}], p {fmt_p(A['pearson_p'])}; "
         f"Spearman ρ = {A['spearman_rho']:.3f}). Absolute agreement was ICC(2,1) = {A['icc21']:.3f}"
         + (f" (95% {'student-cluster ' if A.get('bootstrap', '').startswith('cluster') else ''}bootstrap CI [{A['icc21_ci'][0]:.3f}, {A['icc21_ci'][1]:.3f}])" if "icc21_ci" in A else "")
         + f", with MAE = {A['mae']:.2f} and RMSE = {A['rmse']:.2f} points. The Bland–Altman bias was {A['ba_bias']:.2f} points "
         f"(limits of agreement {A['ba_lo']:.2f} to {A['ba_hi']:.2f}). On the five-point grade scale the quadratic weighted κ was "
         f"{A['qwk']:.3f} and exact agreement {100 * A['exact_grade']:.1f}%.")
    if "inter_rater" in rep:
        I = rep["inter_rater"]
        s += f" For reference, the two experts agreed with ICC(2,1) = {I['icc21']:.3f} (MAE = {I['mae']:.2f}, n = {I['n']})."
    if rep.get("ablation"):
        best = min(rep["ablation"], key=lambda r: r["mae_loo"])
        s += f" In the leave-one-out ablation the lowest error was obtained by the '{best['model']}' model (MAE = {best['mae_loo']:.2f}, r = {best['r_loo']:.3f})."
    return s


def markdown(rep, lang="en"):
    uz = lang == "uz"
    A = rep["agreement"]
    L = [f"# {'Validatsiya hisoboti' if uz else 'Validation report'} — {rep['task']}", "",
         f"{'Algoritm versiyasi' if uz else 'Algorithm version'}: {rep['version']} · {'ball manbasi' if uz else 'score source'}: {rep['model']} · "
         f"{'baholangan urinishlar' if uz else 'rated attempts'}: {rep['n_rated']} / {rep['n_attempts']}", "",
         f"## {'Avtomatik va ekspert bahosi mosligi' if uz else 'Automated vs expert agreement'}", "", "| | |", "|---|---|"]
    L += [f"| Pearson r [95% CI] | {A['pearson_r']:.3f} [{A['pearson_lo']:.3f}, {A['pearson_hi']:.3f}], p {fmt_p(A['pearson_p'])} |",
          f"| Spearman ρ | {A['spearman_rho']:.3f}, p {fmt_p(A['spearman_p'])} |", f"| MAE / RMSE | {A['mae']:.2f} / {A['rmse']:.2f} |",
          f"| ICC(2,1) / ICC(3,1) | {A['icc21']:.3f} / {A['icc31']:.3f} |", f"| Bland–Altman bias [LoA] | {A['ba_bias']:.2f} [{A['ba_lo']:.2f}, {A['ba_hi']:.2f}] |",
          f"| QWK / exact grade | {A['qwk']:.3f} / {100 * A['exact_grade']:.1f}% |", ""]
    if "inter_rater" in rep:
        I = rep["inter_rater"]
        L += [f"## {'Ekspertlararo ishonchlilik' if uz else 'Inter-rater reliability'}", "",
              f"n = {I['n']}, ICC(2,1) = {I['icc21']:.3f}, r = {I['pearson_r']:.3f}, MAE = {I['mae']:.2f}, QWK = {I['qwk']:.3f}", ""]
    if rep.get("per_metric"):
        L += [f"## {'Metrikalar va ekspert bahosi (Spearman ρ)' if uz else 'Metrics vs expert score (Spearman ρ)'}", "", "| Metric | n | ρ | p | sign |", "|---|---|---|---|---|"]
        L += [f"| {p['metric']} | {p['n']} | {p['rho']:.3f} | {fmt_p(p['p'])} | {p['expected_sign']} |" for p in rep["per_metric"]] + [""]
    if rep.get("ablation"):
        L += ["## Ablation (LOOCV)", "", "| model | r | MAE |", "|---|---|---|"]
        L += [f"| {a['model']} | {a['r_loo']:.3f} | {a['mae_loo']:.2f} |" for a in rep["ablation"]] + [""]
    if rep.get("baselines"):
        L += ["## Baselines (LOOCV)", "", "| method | r | MAE | ICC(2,1) | ΔMAE [95% CI] |", "|---|---|---|---|---|"]
        L += [f"| {b['method']} | {b['r']:.3f} | {b['MAE']:.2f} | {b['ICC(2,1)']:.3f} | {b.get('ΔMAE (calibrated framework − method) [95% CI]', '—')} |" for b in rep["baselines"]] + [""]
    L += ["## Results paragraph (EN, for the manuscript)", "", results_paragraph(rep), ""]
    return "\n".join(L)


def analytics_report(course, out=None):
    """Learning analytics of the SRL cycle per task."""
    plt = _plt()
    df = course.dataset(anonymise=True)
    out = Path(out or course.root / "results" / ("analytics_" + dt.datetime.now().strftime("%Y%m%d_%H%M")))
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for tid, g in df.groupby("task_id"):
        per = g.sort_values("attempt").groupby("student")
        first, best = per["score"].first(), per["score"].max()
        pg = paired_gain(first.values, best.values)
        multi = [x for _, x in per if len(x) > 1 and x["self_pred"].notna().all()]
        cal_f = [abs(x.iloc[0]["self_pred"] - x.iloc[0]["score"]) for x in multi]
        cal_l = [abs(x.iloc[-1]["self_pred"] - x.iloc[-1]["score"]) for x in multi]
        cg = paired_gain(cal_f, cal_l) if len(multi) > 1 else {}
        rows.append({"task_id": tid, "students": int(per.ngroups), "attempts": int(len(g)), "mean_attempts": float(len(g) / per.ngroups),
                     "first_mean": pg.get("mean_first", float(first.mean())), "best_mean": pg.get("mean_best", float(best.mean())), "gain": pg.get("gain"),
                     "t": pg.get("t"), "p": pg.get("p"), "dz": pg.get("dz"), "wilcoxon_p": pg.get("wilcoxon_p"),
                     "on_time_rate": float(g.groupby("student")["on_time"].first().astype(float).mean()) if g["on_time"].notna().any() else np.nan,
                     "reflection_rate": float((g["reflection_words"] > 0).mean()), "calib_err_first": cg.get("mean_first"), "calib_err_last": cg.get("mean_best"),
                     "calib_p": cg.get("p"), "calib_n": cg.get("n", 0)})
    T = pd.DataFrame(rows)
    T.to_csv(out / "analytics.csv", index=False)
    if len(T):
        fig, ax = plt.subplots(figsize=(max(3.5, 1.1 * len(T)), 3))
        x = np.arange(len(T))
        ax.bar(x - 0.18, T["first_mean"], 0.36, label="First attempt", color="#98a2b3")
        ax.bar(x + 0.18, T["best_mean"], 0.36, label="Best attempt", color="#2b59d9")
        ax.set(xticks=x, xticklabels=T["task_id"], ylim=(0, 100), ylabel="Mean score")
        ax.legend(frameon=False, fontsize=8)
        _save(fig, out, "fig_learning_gain")
        bests = df.groupby(["task_id", "student"])["score"].max()
        cnt = [int((bests.map(grade5) == g).sum()) for g in (2, 3, 4, 5)]
        fig, ax = plt.subplots(figsize=(3, 2.6))
        ax.bar(["2", "3", "4", "5"], cnt, color="#2b59d9")
        ax.set(xlabel="Grade (best attempt)", ylabel="n")
        _save(fig, out, "fig_grade_distribution")
    return T, out
