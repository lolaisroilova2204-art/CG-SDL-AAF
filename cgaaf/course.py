"""Course folder: assignments, students, submissions, expert ratings.

course/
  course.json
  students.csv                     student_id,name,group,program
  experts.csv                      student_id,task_id,attempt,expert1,expert2,note
  tasks/<task_id>/task.json        + reference.png / reference.obj / reference.blend, ref_points.npy (cache)
  submissions/<student_id>/<task_id>/<attempt>/
        student files (PNG/JPG, OBJ/STL/..., .blend, .fbx)
        submission.json            plan + reflection entered by the student (SRL data)
        result.json                computed metrics, score, feedback (re-computable)
        aaf_norm.png, aaf_heatmap.png, aaf_render.png, aaf_export.obj, aaf_cloud.npy (generated)
  results/                         exports, figures, reports
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import ALGO_VERSION
from .blender import BLENDER_INPUT, run_blender
from .feedback import feedback_items
from .image_metrics import compare_images, heatmap
from .mesh_metrics import MESH_EXT, compare_meshes, load_mesh, prepare_reference, topology


def environment():
    """Library versions recorded in every result (reproducibility)."""
    import importlib
    out = {"python": __import__("platform").python_version(), "algo": ALGO_VERSION}
    from importlib.metadata import PackageNotFoundError, version
    for pkg in ("numpy", "scipy", "scikit-image", "trimesh", "Pillow", "torch", "lpips"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out
from .scoring import grade5, score

IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
DERIVED = {"submission.json", "result.json"}  # + every file whose name starts with "aaf_" (generated artefacts)


def _jload(p, default=None):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return default


def _jsave(p, obj):
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=float), encoding="utf-8")


def anon_code(sid: str) -> str:
    return "S-" + hashlib.sha1(("cgaaf:" + sid).encode()).hexdigest()[:5].upper()


class Course:
    def __init__(self, root):
        self.root = Path(root)
        if not (self.root / "course.json").exists():
            raise FileNotFoundError(f"{self.root} is not a CG-SDL-AAF course folder (course.json missing)")
        self.meta = _jload(self.root / "course.json", {})
        self.blender = None  # optional ('exe'|'bpy', path)

    # ------------------------------------------------------------ create / meta
    @classmethod
    def create(cls, root, name, lang="uz", synthetic=False):
        root = Path(root)
        for d in ["tasks", "submissions", "results"]:
            (root / d).mkdir(parents=True, exist_ok=True)
        _jsave(root / "course.json", {"name": name, "lang": lang, "created": dt.datetime.now().isoformat(timespec="seconds"),
                                       "version": ALGO_VERSION, "synthetic": synthetic})
        if not (root / "students.csv").exists():
            (root / "students.csv").write_text("student_id,name,group,program\n", encoding="utf-8")
        return cls(root)

    @property
    def lang(self):
        return self.meta.get("lang", "uz")

    # ------------------------------------------------------------ tasks
    def tasks(self):
        out = [t for p in sorted((self.root / "tasks").glob("*/task.json")) if (t := _jload(p))]
        return sorted(out, key=lambda t: (t.get("deadline") or "9999", t.get("created", "")))

    def task(self, tid):
        return _jload(self.root / "tasks" / tid / "task.json")

    def task_dir(self, tid):
        return self.root / "tasks" / tid

    TASK_DEFAULTS = {"objects": "auto", "frames": 1, "ui_crop": "auto", "check_size": False, "trim": True, "align": "pca_icp", "render_helpers": False}

    def save_task(self, task: dict, ref_image=None, ref_mesh=None, ref_blend=None):
        """Store a task; reference files are copied. A reference .blend/.fbx is re-rendered
        and exported with exactly the pipeline used for student .blend files."""
        tid = task["id"]
        d = self.task_dir(tid)
        d.mkdir(parents=True, exist_ok=True)
        for k, v in self.TASK_DEFAULTS.items():
            task.setdefault(k, v)
        task.setdefault("created", dt.datetime.now().isoformat(timespec="seconds"))
        task.setdefault("render", {"mode": "canonical", "res": 512, "engine": "CYCLES", "samples": 32})
        ref = task.setdefault("reference", {})
        if ref_blend:
            shutil.copy2(ref_blend, d / ("reference" + Path(ref_blend).suffix.lower()))
            for old in d.glob("reference_f*.png"):
                old.unlink()
            r = task["render"]
            out = run_blender(ref_blend, d / "reference.png" if task["type"] != "mesh" else None, d / "reference.obj" if task["type"] != "image" else None,
                              r["mode"], r["res"], r["engine"], r["samples"], blender=self.blender, obj_filter=task["objects"], frames=task["frames"],
                              helpers="show" if task["render_helpers"] else "hide")
            ref["from_blend"] = True
            ref["blender_version"] = out.get("blender_version")
            ref["objects"] = out.get("objects")
            if task["type"] != "mesh":
                ref["images"] = [Path(p).name for p in out["pngs"]]
                ref["image"] = ref["images"][0]
            if task["type"] != "image":
                ref["mesh"] = "reference.obj"
        if ref_image:
            dst = d / ("reference" + Path(ref_image).suffix.lower())
            if Path(ref_image).resolve() != dst.resolve():
                shutil.copy2(ref_image, dst)
            ref["image"], ref["images"] = dst.name, [dst.name]
            ref["from_blend"] = False
        if ref_mesh:
            dst = d / ("reference" + Path(ref_mesh).suffix.lower())
            if Path(ref_mesh).resolve() != dst.resolve():
                shutil.copy2(ref_mesh, dst)
            ref["mesh"] = dst.name
        (d / "ref_points.npy").unlink(missing_ok=True)
        _jsave(d / "task.json", task)
        return task

    def delete_task(self, tid):
        shutil.rmtree(self.task_dir(tid), ignore_errors=True)

    def ref_points(self, task, with_scale=False):
        d = self.task_dir(task["id"])
        cache, key = d / "ref_points.npy", d / "ref_points.key"
        k = f'{task["reference"]["mesh"]}|{task.get("align", "pca_icp")}|{task.get("objects", "auto")}|{ALGO_VERSION}'
        meta = _jload(key) if key.exists() else None
        if cache.exists() and isinstance(meta, dict) and meta.get("key") == k and "topo" in meta:
            P, s = np.load(cache), meta["scale"]
        else:
            m = load_mesh(d / task["reference"]["mesh"], drop_planes=task.get("objects", "auto") != "all")
            P, s = prepare_reference(m, task.get("align", "pca_icp"), return_scale=True)
            np.save(cache, P)
            _jsave(key, {"key": k, "scale": s, "topo": topology(m)})
        return (P, s) if with_scale else P

    def ref_topology(self, task):
        """Topology of the reference mesh: defects are penalised only in excess of it."""
        self.ref_points(task)
        meta = _jload(self.task_dir(task["id"]) / "ref_points.key")
        return meta.get("topo") if isinstance(meta, dict) else None

    # ------------------------------------------------------------ students
    def students(self):
        p = self.root / "students.csv"
        if not p.exists():
            return {}
        with open(p, encoding="utf-8-sig", newline="") as f:
            return {r["student_id"]: r for r in csv.DictReader(f) if r.get("student_id")}

    def save_student(self, sid, name, group="", program=""):
        st = self.students()
        st[sid] = {"student_id": sid, "name": name, "group": group, "program": program}
        with open(self.root / "students.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, ["student_id", "name", "group", "program"])
            w.writeheader()
            w.writerows(st.values())
        return st[sid]

    def next_student_id(self):
        ids = [int(s[1:]) for s in self.students() if s[:1] == "S" and s[1:].isdigit()]
        return "S%03d" % (max(ids, default=0) + 1)

    # ------------------------------------------------------------ submissions
    def attempt_dirs(self, sid=None, tid=None):
        base = self.root / "submissions"
        pat = f"{sid or '*'}/{tid or '*'}/*"
        return sorted([p for p in base.glob(pat) if p.is_dir() and p.name.isdigit()], key=lambda p: (p.parent.parent.name, p.parent.name, int(p.name)))

    def attempts(self, sid, tid):
        return [d for d in self.attempt_dirs(sid, tid)]

    def submit(self, sid, tid, files, plan=None, use_lpips=True, lang=None):
        task = self.task(tid)
        n = len(self.attempts(sid, tid)) + 1
        if n > int(task.get("max_attempts", 99)):
            raise RuntimeError("max_attempts")
        d = self.root / "submissions" / sid / tid / str(n)
        d.mkdir(parents=True, exist_ok=False)
        for f in files:
            shutil.copy2(f, d / Path(f).name)
        _jsave(d / "submission.json", {"student_id": sid, "task_id": tid, "attempt": n, "submitted_at": dt.datetime.now().isoformat(timespec="seconds"),
                                       "plan": plan or {}, "reflection": None, "files": [Path(f).name for f in files]})
        return self.evaluate_attempt(d, use_lpips=use_lpips)

    def save_reflection(self, attempt_dir, text):
        p = Path(attempt_dir) / "submission.json"
        s = _jload(p, {})
        s["reflection"] = {"text": text, "words": len(text.split()), "at": dt.datetime.now().isoformat(timespec="seconds")}
        _jsave(p, s)

    def pending(self, force=False):
        return [d for d in self.attempt_dirs() if force or not (d / "result.json").exists()]

    def _ensure_meta(self, d):
        p = d / "submission.json"
        s = _jload(p)
        if s is None:  # files dropped in by the teacher (batch mode)
            st = min((f.stat().st_mtime for f in d.iterdir() if f.is_file()), default=time.time())
            s = {"student_id": d.parent.parent.name, "task_id": d.parent.name, "attempt": int(d.name),
                 "submitted_at": dt.datetime.fromtimestamp(st).isoformat(timespec="seconds"), "plan": {}, "reflection": None}
            _jsave(p, s)
        return s

    def evaluate_attempt(self, d, use_lpips=True):
        d = Path(d)
        sub = self._ensure_meta(d)
        task = self.task(sub["task_id"])
        if task is None:
            raise RuntimeError(f"Unknown task {sub['task_id']}")
        task = {**self.TASK_DEFAULTS, **task}
        files = [f for f in d.iterdir() if f.is_file() and f.name not in DERIVED and not f.name.startswith("aaf_")]
        by = lambda exts: sorted(f for f in files if f.suffix.lower() in exts)
        blends, imgs, meshes = by(BLENDER_INPUT), by(IMG_EXT), by(MESH_EXT)
        t0 = time.time()
        res = {"version": ALGO_VERSION, "evaluated_at": dt.datetime.now().isoformat(timespec="seconds"), "task_type": task["type"], "m": {},
               "env": environment()}
        rset = task.get("render", {})
        tdir = self.task_dir(task["id"])
        ref_imgs = task["reference"].get("images") or ([task["reference"]["image"]] if task["reference"].get("image") else [])
        bl_out = None
        if blends:  # standardised re-rendering / export
            for old in d.glob("aaf_render*.png"):
                old.unlink()
            nfr = len(ref_imgs) if task["reference"].get("from_blend") and len(ref_imgs) > 1 else 1
            bl_out = run_blender(blends[0], d / "aaf_render.png" if task["type"] != "mesh" else None, d / "aaf_export.obj" if task["type"] != "image" else None,
                                 rset.get("mode", "canonical"), rset.get("res", 512), rset.get("engine", "CYCLES"), rset.get("samples", 32),
                                 blender=self.blender, obj_filter=task["objects"], frames=nfr, helpers="show" if task["render_helpers"] else "hide")
            res["env"]["blender"] = bl_out.get("blender_version")
            res["blender_objects"] = bl_out.get("objects")
        if task["type"] != "mesh":
            manual = sub.get("crop")  # {file: {x0,y0,x1,y1}} fractions, or one dict for all images
            crop_for = lambda f: (manual.get(Path(f).name) if isinstance(manual, dict) and Path(f).name in manual
                                  else manual if isinstance(manual, dict) and "x0" in manual else task.get("ui_crop", "auto"))
            if blends:
                pairs, kind = list(zip(ref_imgs, [Path(p) for p in bl_out["pngs"]])), "render"
                crop_fn = lambda f: None  # clean render, no UI
            elif imgs:
                pairs, kind, crop_fn = [(ref_imgs[0], f) for f in imgs], "screenshot", crop_for
            else:
                raise RuntimeError("no image / .blend in submission")
            results = []
            for rname, sfile in pairs:
                ir = compare_images(tdir / rname, sfile, trim=task.get("trim", True), use_lpips=use_lpips, crop=crop_fn(sfile),
                                    ref_crop=None if task["reference"].get("from_blend") else task.get("ui_crop", "auto"))
                results.append((ir, sfile))
            if kind == "render" and len(results) > 1:  # animation: mean over frames
                keys = results[0][0].m.keys()
                m = {k: float(np.mean([r.m[k] for r, _ in results])) for k in keys}
                per = [score({**task, "type": "image"}, r.m) for r, _ in results]
                ir, sfile = results[int(np.argmin(per))]
                res["image"] = {"raw": {**ir.raw, **{k: float(np.mean([r.raw[k] for r, _ in results if r.raw.get(k) is not None]))
                                                     for k in ("ssim", "msssim", "deltaE00", "color", "edge_f1", "psnr", "ssim_plain")}},
                                "frames": len(results), "frame_scores": [round(p, 2) for p in per], "worst_frame": int(np.argmin(per))}
            else:  # several screenshots: the best-matching one is assessed
                per = [score({**task, "type": "image"}, r.m) for r, _ in results]
                ir, sfile = results[int(np.argmax(per))]
                m = ir.m
                res["image"] = {"raw": ir.raw}
                if len(results) > 1:
                    res["image"].update({"n_images": len(results), "image_scores": [round(p, 2) for p in per]})
            Image.fromarray(ir.stu_norm.astype(np.uint8)).save(d / "aaf_norm.png")
            heatmap(ir, d / "aaf_heatmap.png")
            res["image"].update({"worst": list(ir.worst), "source": kind, "file": Path(sfile).name, "crop": ir.raw.get("crop_box"), "manual_crop": bool(manual)})
            res["m"].update(m)
            res["blend_possible"] = bool(task["reference"].get("from_blend"))
        if task["type"] != "image":
            src = d / "aaf_export.obj" if blends else (meshes[0] if meshes else None)
            if src is None:
                raise RuntimeError("no mesh / .blend in submission")
            ref_pts, ref_scale = self.ref_points(task, with_scale=True)
            mr = compare_meshes(load_mesh(src, drop_planes=task["objects"] != "all"), ref_pts, task.get("align", "pca_icp"),
                                ref_scale=ref_scale if task.get("check_size") else None, ref_topo=self.ref_topology(task))
            np.save(d / "aaf_cloud.npy", mr.aligned[:: max(1, len(mr.aligned) // 1024)].astype(np.float32))
            res["mesh"] = {"raw": mr.raw, "source": "blender_export" if blends else "file", "file": Path(src).name}
            res["m"].update(mr.m)
        res["score"] = round(score(task, res["m"]), 2)
        res["grade"] = grade5(res["score"])
        for part, kind in (("image", "image"), ("mesh", "mesh")):  # channel scores for the feedback rules
            if part in res:
                res[part]["score"] = round(score({**task, "type": kind}, res["m"]), 2)
        res["calibrated"] = bool(task.get("calibration"))
        res["plan"] = sub.get("plan") or {}
        res["feedback"] = feedback_items(res)
        res["seconds"] = round(time.time() - t0, 2)
        _jsave(d / "result.json", res)
        return res

    def set_crop(self, attempt_dir, crop, file=None):
        """Store a manual crop (fractions x0,y0,x1,y1) for one image or all images of an attempt."""
        p = Path(attempt_dir) / "submission.json"
        s = _jload(p, {})
        if crop is None:
            s.pop("crop", None)
        elif file:
            c = s.get("crop") if isinstance(s.get("crop"), dict) and "x0" not in s.get("crop", {}) else {}
            c[file] = crop
            s["crop"] = c
        else:
            s["crop"] = crop
        _jsave(p, s)

    def result(self, d):
        return _jload(Path(d) / "result.json")

    def submission(self, d):
        return _jload(Path(d) / "submission.json", {})

    # ------------------------------------------------------------ expert ratings
    def experts(self):
        p = self.root / "experts.csv"
        out = {}
        if p.exists():
            with open(p, encoding="utf-8-sig", newline="") as f:
                for r in csv.DictReader(f):
                    try:
                        out[(r["student_id"], r["task_id"], int(r["attempt"]))] = r
                    except (KeyError, ValueError):
                        continue
        return out

    def save_expert(self, sid, tid, attempt, e1=None, e2=None, note=""):
        ex = self.experts()
        ex[(sid, tid, int(attempt))] = {"student_id": sid, "task_id": tid, "attempt": int(attempt),
                                        "expert1": "" if e1 in (None, "") else e1, "expert2": "" if e2 in (None, "") else e2, "note": note or ""}
        self._write_experts(ex)

    def _write_experts(self, ex):
        with open(self.root / "experts.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, ["student_id", "task_id", "attempt", "expert1", "expert2", "note"])
            w.writeheader()
            for k in sorted(ex):
                w.writerow({c: ex[k].get(c, "") for c in ["student_id", "task_id", "attempt", "expert1", "expert2", "note"]})

    def import_experts(self, path):
        import pandas as pd
        df = pd.read_excel(path) if str(path).lower().endswith((".xlsx", ".xls")) else pd.read_csv(path)
        ex = self.experts()
        n = 0
        for _, r in df.iterrows():
            if str(r.get("expert1", "")).strip() in ("", "nan") and str(r.get("expert2", "")).strip() in ("", "nan"):
                continue
            k = (str(r["student_id"]), str(r["task_id"]), int(r["attempt"]))
            clean = lambda v: "" if str(v).strip() in ("", "nan") else float(v)
            ex[k] = {"student_id": k[0], "task_id": k[1], "attempt": k[2], "expert1": clean(r.get("expert1", "")),
                     "expert2": clean(r.get("expert2", "")), "note": "" if str(r.get("note", "")) == "nan" else str(r.get("note", ""))}
            n += 1
        self._write_experts(ex)
        return n

    def expert_template(self, path, latest_only=True):
        """Blind rating sheet: no automated scores."""
        import pandas as pd
        rows, ex = [], self.experts()
        groups = {}
        for d in self.attempt_dirs():
            groups.setdefault((d.parent.parent.name, d.parent.name), []).append(d)
        for (sid, tid), ds in sorted(groups.items()):
            for d in (ds[-1:] if latest_only else ds):
                k = (sid, tid, int(d.name))
                e = ex.get(k, {})
                rows.append({"code": anon_code(sid), "student_id": sid, "task_id": tid, "attempt": int(d.name),
                             "folder": str(d.relative_to(self.root)), "expert1": e.get("expert1", ""), "expert2": e.get("expert2", ""), "note": e.get("note", "")})
        pd.DataFrame(rows).to_excel(path, index=False)
        return len(rows)

    # ------------------------------------------------------------ dataset
    def dataset(self, anonymise=True, rescore=False):
        import pandas as pd
        st, ex, tasks = self.students(), self.experts(), {t["id"]: t for t in self.tasks()}
        rows = []
        for d in self.attempt_dirs():
            r, s = self.result(d), self.submission(d)
            if r is None:
                continue
            sid, tid, att = d.parent.parent.name, d.parent.name, int(d.name)
            t = tasks.get(tid, {})
            info = st.get(sid, {})
            e = ex.get((sid, tid, att), {})
            num = lambda v: float(v) if str(v).strip() not in ("", "nan", "None") else np.nan
            e1, e2 = num(e.get("expert1", "")), num(e.get("expert2", ""))
            plan, refl = s.get("plan") or {}, s.get("reflection") or {}
            sub_at = s.get("submitted_at", "")
            row = {"student": anon_code(sid) if anonymise else info.get("name", sid), "student_id": None if anonymise else sid,
                   "group": info.get("group", ""), "task_id": tid, "task_type": t.get("type"), "attempt": att, "submitted_at": sub_at,
                   "deadline": t.get("deadline"), "on_time": (sub_at[:10] <= t["deadline"]) if t.get("deadline") and sub_at else None,
                   "plan_goal_words": len(str(plan.get("goal", "")).split()) if plan.get("goal") else 0, "plan_minutes": plan.get("minutes"),
                   "self_pred": plan.get("pred"), "reflection_words": refl.get("words", 0),
                   "image_source": (r.get("image") or {}).get("source"), "mesh_source": (r.get("mesh") or {}).get("source")}
            for k, v in ((r.get("image") or {}).get("raw") or {}).items():
                row["img_" + k] = v
            for k, v in ((r.get("mesh") or {}).get("raw") or {}).items():
                row["mesh_" + k] = v
            for k, v in (r.get("m") or {}).items():
                row["m_" + k] = v
            row["score"] = r.get("score")
            row["score_current"] = round(score(t, r.get("m", {})), 2) if t else None
            row["score_default"] = round(score({k: v for k, v in t.items() if k != "calibration"}, r.get("m", {})), 2) if t else None
            row["retrospective"] = bool(s.get("retrospective", False))
            row["grade"] = r.get("grade")
            row["expert1"], row["expert2"] = e1, e2
            row["expert_mean"] = np.nanmean([e1, e2]) if np.isfinite([e1, e2]).any() else np.nan
            row["calibrated_model"] = bool(t.get("calibration"))
            row["version"] = r.get("version")
            rows.append(row)
        df = pd.DataFrame(rows)
        if anonymise and "student_id" in df:
            df = df.drop(columns=["student_id"])
        return df
