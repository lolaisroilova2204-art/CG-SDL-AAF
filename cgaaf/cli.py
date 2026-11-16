"""Command-line interface:  python -m cgaaf <command> ...   (or `cgaaf` after pip install)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import ALGO_VERSION, __version__
from .feedback import render_feedback
from .i18n import tr


def _course(p):
    from .course import Course
    return Course(p)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cgaaf", description=f"CG-SDL-AAF {__version__} — automated assessment of self-directed learning in Computer Graphics / "
                                                           "Kompyuter grafikasida mustaqil ta'limni avtomatik baholash")
    ap.add_argument("--lang", choices=["uz", "en"], default=None, help="interface language / interfeys tili")
    sp = ap.add_subparsers(dest="cmd", required=True)

    p = sp.add_parser("init", help="create a course folder / kurs papkasini yaratish")
    p.add_argument("course"); p.add_argument("--name", default="Kompyuter grafikasi")

    p = sp.add_parser("add-task", help="add an assignment / topshiriq qo'shish")
    p.add_argument("course"); p.add_argument("--id", required=True); p.add_argument("--type", choices=["image", "mesh", "both"], required=True)
    p.add_argument("--title-uz", default=""); p.add_argument("--title-en", default=""); p.add_argument("--desc-uz", default=""); p.add_argument("--desc-en", default="")
    p.add_argument("--ref-image"); p.add_argument("--ref-mesh"); p.add_argument("--ref-blend")
    p.add_argument("--align", choices=["pca_icp", "icp", "none"], default="pca_icp"); p.add_argument("--no-trim", action="store_true")
    p.add_argument("--render-mode", choices=["canonical", "scene"], default="canonical"); p.add_argument("--res", type=int, default=512)
    p.add_argument("--engine", default="CYCLES"); p.add_argument("--samples", type=int, default=32)
    p.add_argument("--max-attempts", type=int, default=5); p.add_argument("--deadline", default="")
    p.add_argument("--objects", default="auto", help="auto | all | names:Torus*,Cube* | collection:Model")
    p.add_argument("--frames", type=int, default=1, help="animation frames to compare (1 = still)")
    p.add_argument("--no-ui-crop", action="store_true"); p.add_argument("--check-size", action="store_true"); p.add_argument("--render-helpers", action="store_true")

    p = sp.add_parser("submit", help="student submission with SRL plan / talaba topshirig'i")
    p.add_argument("course"); p.add_argument("--student", required=True); p.add_argument("--task", required=True); p.add_argument("files", nargs="+")
    p.add_argument("--goal", default=""); p.add_argument("--minutes", type=int); p.add_argument("--pred", type=int); p.add_argument("--no-lpips", action="store_true")

    p = sp.add_parser("evaluate", help="assess pending submissions / baholanmaganlarni baholash")
    p.add_argument("course"); p.add_argument("--force", action="store_true"); p.add_argument("--no-lpips", action="store_true")

    p = sp.add_parser("expert-template", help="blind expert rating sheet (xlsx)"); p.add_argument("course"); p.add_argument("--out", default=None); p.add_argument("--all-attempts", action="store_true")
    p = sp.add_parser("expert-import", help="import expert ratings (xlsx/csv)"); p.add_argument("course"); p.add_argument("file")

    p = sp.add_parser("validate", help="validation report + figures / validatsiya hisoboti")
    p.add_argument("course"); p.add_argument("--task"); p.add_argument("--model", choices=["stored", "current"], default="stored")
    p.add_argument("--all-attempts", action="store_true", help="use every rated attempt (default: latest rated attempt per student and task)")

    p = sp.add_parser("calibrate", help="NNLS calibration against experts / kalibrlash")
    p.add_argument("course"); p.add_argument("--task", required=True); p.add_argument("--apply", action="store_true")

    p = sp.add_parser("analytics", help="learning analytics / o'quv analitikasi"); p.add_argument("course")
    p = sp.add_parser("export", help="dataset CSV + XLSX"); p.add_argument("course"); p.add_argument("--names", action="store_true", help="keep names (default: anonymised)")
    p = sp.add_parser("render", help="re-render a .blend/.fbx with Blender"); p.add_argument("file"); p.add_argument("--png"); p.add_argument("--obj")
    p.add_argument("--mode", default="canonical"); p.add_argument("--res", type=int, default=512); p.add_argument("--samples", type=int, default=32)
    p = sp.add_parser("demo", help="synthetic sample course / sintetik namuna kurs"); p.add_argument("course"); p.add_argument("--students", type=int, default=24)
    p = sp.add_parser("archive-template", help="scan archived works -> mapping sheet / arxiv ishlari uchun jadval")
    p.add_argument("archive"); p.add_argument("--out", default="archive_mapping.xlsx")
    p = sp.add_parser("import-archive", help="import archived works + expert grades / arxivni import qilish")
    p.add_argument("course"); p.add_argument("mapping"); p.add_argument("--expert-max", type=float, default=100.0, help="max of the original grading scale (e.g. 5, 10, 100)")
    p.add_argument("--expert-min", type=float, default=0.0, help="min of the original scale (e.g. 2 for a 2-5 scale)")
    p.add_argument("--expert-scale", choices=["linear", "grade5"], default="linear", help="grade5: national 2-5 grades -> centre of 0-100 band")
    p.add_argument("--no-evaluate", action="store_true"); p.add_argument("--no-lpips", action="store_true")
    p = sp.add_parser("benchmark", help="controlled-perturbation benchmark / boshqariladigan benchmark")
    p.add_argument("out"); p.add_argument("--ref-images"); p.add_argument("--ref-meshes"); p.add_argument("--no-blender", action="store_true")
    p.add_argument("--repeats", type=int, default=3); p.add_argument("--no-lpips", action="store_true")
    p = sp.add_parser("paper", help="all manuscript tables/figures / maqola uchun barcha jadval va rasmlar")
    p.add_argument("course", nargs="?", default=None, help="course folder (omit for a benchmark-only technical paper)")
    p.add_argument("--benchmark"); p.add_argument("--out")
    sp.add_parser("gui", help="graphical interface / grafik interfeys")
    sp.add_parser("version")

    a = ap.parse_args(argv)
    L = a.lang or "uz"

    if a.cmd == "version":
        print(__version__, ALGO_VERSION); return 0
    if a.cmd == "gui":
        from .gui import run
        return run(L)
    if a.cmd == "init":
        from .course import Course
        Course.create(a.course, a.name, L); print(tr("saved", L), a.course); return 0
    if a.cmd == "demo":
        from .demo import make_demo
        print(tr("synthetic_warn", L))
        make_demo(a.course, a.students, lang=L); print(tr("done", L), a.course); return 0
    if a.cmd == "render":
        from .blender import run_blender
        run_blender(a.file, a.png, a.obj, a.mode, a.res, "CYCLES", a.samples); print(tr("done", L)); return 0

    if a.cmd == "archive-template":
        from .archive import make_mapping_template
        df = make_mapping_template(a.archive, a.out)
        print(len(df), "→", a.out, "| CONVERT:", int(df["status"].str.startswith("CONVERT").sum())); return 0
    if a.cmd == "benchmark":
        from .benchmark import run_benchmark
        r = run_benchmark(a.out, a.ref_images, a.ref_meshes, not a.no_blender, a.repeats, not a.no_lpips)
        print(json.loads((r["out"] / "benchmark_summary.json").read_text())["paragraph"]); print("→", r["out"]); return 0
    if a.cmd == "paper" and a.course is None:
        from .paper import make_paper_bundle
        make_paper_bundle(None, a.benchmark, a.out); return 0

    c = _course(a.course)
    L = a.lang or c.lang
    if a.cmd == "import-archive":
        from .archive import import_mapping
        s = import_mapping(c, a.mapping, a.expert_max, expert_min=a.expert_min, scale=a.expert_scale)
        for row, why in s["skipped"][:30]:
            print(f"  row {row}: {why}")
        if not a.no_evaluate:
            P = c.pending()
            for i, d in enumerate(P, 1):
                try:
                    r = c.evaluate_attempt(d, use_lpips=not a.no_lpips); print(f"[{i}/{len(P)}] {d.relative_to(c.root)}  {r['score']:.1f}")
                except Exception as e:
                    print(f"[{i}/{len(P)}] {d.relative_to(c.root)}  {tr('error', L)}: {e}", file=sys.stderr)
        return 0
    if a.cmd == "paper":
        from .paper import make_paper_bundle
        make_paper_bundle(c, a.benchmark, a.out); return 0
    if a.cmd == "add-task":
        t = {"id": a.id, "type": a.type, "title_uz": a.title_uz or a.id, "title_en": a.title_en or a.title_uz or a.id, "desc_uz": a.desc_uz, "desc_en": a.desc_en,
             "align": a.align, "trim": not a.no_trim, "max_attempts": a.max_attempts, "deadline": a.deadline,
             "render": {"mode": a.render_mode, "res": a.res, "engine": a.engine, "samples": a.samples},
             "objects": a.objects, "frames": a.frames, "ui_crop": None if a.no_ui_crop else "auto", "check_size": a.check_size, "render_helpers": a.render_helpers}
        c.save_task(t, a.ref_image, a.ref_mesh, a.ref_blend); print(tr("saved", L), a.id)
    elif a.cmd == "submit":
        try:
            r = c.submit(a.student, a.task, a.files, {"goal": a.goal, "minutes": a.minutes, "pred": a.pred}, use_lpips=not a.no_lpips)
        except RuntimeError as e:
            if "max_attempts" in str(e):
                print(tr("no_attempts", L)); return 2
            raise
        print(f"{tr('score', L)}: {r['score']:.1f}  {tr('grade', L)}: {r['grade']}")
        for s in render_feedback(r["feedback"], L):
            print(" •", s)
    elif a.cmd == "evaluate":
        P = c.pending(a.force)
        print(len(P), tr("pending", L))
        for i, d in enumerate(P, 1):
            try:
                r = c.evaluate_attempt(d, use_lpips=not a.no_lpips)
                print(f"[{i}/{len(P)}] {d.relative_to(c.root)}  {r['score']:.1f}")
            except Exception as e:
                print(f"[{i}/{len(P)}] {d.relative_to(c.root)}  {tr('error', L)}: {e}", file=sys.stderr)
    elif a.cmd == "expert-template":
        out = a.out or str(c.root / "results" / "expert_rating_sheet.xlsx")
        n = c.expert_template(out, latest_only=not a.all_attempts); print(n, "→", out)
    elif a.cmd == "expert-import":
        print(c.import_experts(a.file), tr("saved", L))
    elif a.cmd == "validate":
        from .reports import validation_report
        rep, out = validation_report(c, a.task, a.model, lang=L, one_per_student=not a.all_attempts)
        print(json.dumps({k: rep[k] for k in rep if k in ("n_rated", "agreement", "inter_rater", "error")}, indent=2, ensure_ascii=False, default=float)[:3000])
        print("→", out)
    elif a.cmd == "calibrate":
        from .scoring import calibrate, keys_for
        df = c.dataset()
        d = df[(df["task_id"] == a.task) & df["expert_mean"].notna()]
        t = c.task(a.task)
        M = [{k[2:]: row[k] for k in d.columns if k.startswith("m_") and row[k] == row[k]} for _, row in d.iterrows()]
        res = calibrate(M, d["expert_mean"].values, keys_for(t["type"]))
        print(json.dumps({k: v for k, v in res.items() if k != "loo_pred"}, indent=2, default=float))
        if a.apply:
            t["calibration"] = {k: res[k] for k in ("b0", "b", "n", "r_loo", "mae_loo")}
            c.save_task(t); print(tr("saved", L))
    elif a.cmd == "analytics":
        from .reports import analytics_report
        T, out = analytics_report(c); print(T.round(3).to_string()); print("→", out)
    elif a.cmd == "export":
        df = c.dataset(anonymise=not a.names)
        out = c.root / "results"
        df.to_csv(out / "dataset.csv", index=False, encoding="utf-8-sig"); df.to_excel(out / "dataset.xlsx", index=False)
        print(len(df), "→", out / "dataset.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
