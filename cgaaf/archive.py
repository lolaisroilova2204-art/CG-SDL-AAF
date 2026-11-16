"""Retrospective import of archived submissions and previously assigned expert grades.

Step 1  make_mapping_template(archive_dir, out.xlsx)
        scans all submission files and writes a sheet to be completed by the teacher:
        student_id | name | group | task_id | attempt | files | expert1 | expert2 | note | status
Step 2  import_mapping(course, mapping.xlsx, expert_max=100)
        copies files into course/submissions/<student>/<task>/<attempt>/, stores students,
        rescales expert grades to 0-100 and marks the attempts as retrospective.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import shutil
from pathlib import Path

import pandas as pd

from .blender import BLENDER_INPUT
from .course import IMG_EXT
from .mesh_metrics import MESH_EXT

SUPPORTED = IMG_EXT | MESH_EXT | BLENDER_INPUT
NEEDS_CONVERSION = {".max": "export to FBX/OBJ in 3ds Max", ".ma": "export to FBX in Maya", ".mb": "export to FBX in Maya",
                    ".c4d": "export to FBX", ".3ds": "export to OBJ/FBX", ".skp": "export to OBJ"}
COLUMNS = ["student_id", "name", "group", "task_id", "attempt", "files", "expert1", "expert2", "note", "crop", "status"]


def _guess_student(p: Path, root: Path):
    rel = p.relative_to(root)
    if len(rel.parts) > 1:
        return rel.parts[0]
    stem = re.split(r"[_\-\s]+", p.stem)
    return stem[0] if stem else p.stem


def make_mapping_template(archive_dir, out_path):
    root = Path(archive_dir)
    rows = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in SUPPORTED:
            status = "ok"
        elif ext in NEEDS_CONVERSION:
            status = "CONVERT: " + NEEDS_CONVERSION[ext]
        else:
            continue
        rows.append({"student_id": _guess_student(p, root), "name": "", "group": "", "task_id": "", "attempt": 1,
                     "files": str(p.resolve()), "expert1": "", "expert2": "", "note": "", "crop": "", "status": status})
    df = pd.DataFrame(rows, columns=COLUMNS)
    df.to_excel(out_path, index=False)
    return df


def _num(v):
    s = str(v).strip().replace(",", ".")
    if s in ("", "nan", "None"):
        return None
    return float(s)


GRADE5_MIDPOINTS = {5: 93.0, 4: 78.0, 3: 62.5, 2: 27.5}  # centres of the score bands 86-100, 71-85, 55-70, 0-54


def rescale(v, scale="linear", vmin=0.0, vmax=100.0):
    """linear: 100*(v-min)/(max-min);  grade5: national 2-5 grade -> centre of its 0-100 band."""
    if v is None:
        return None
    if scale == "grade5":
        g = int(round(v))
        if g not in GRADE5_MIDPOINTS:
            raise ValueError(f"grade {v} is not on the 2-5 scale")
        return GRADE5_MIDPOINTS[g]
    return round(100.0 * (v - vmin) / (vmax - vmin), 2)


def import_mapping(course, mapping_path, expert_max=100.0, copy=True, log=print, expert_min=0.0, scale="linear"):
    """Rows with the same (student_id, task_id, attempt) are merged (e.g. a render + a mesh file).
    `files` may list several paths separated by ';'. Expert grades are rescaled to 0-100 with
    `scale` = 'linear' (expert_min..expert_max) or 'grade5' (2-5 national grades -> band centres).
    An optional `crop` column (x0,y0,x1,y1 in pixels) removes UI from screenshots. Returns a summary."""
    p = str(mapping_path)
    df = pd.read_excel(p) if p.lower().endswith((".xlsx", ".xls")) else pd.read_csv(p)
    df.columns = [str(c).strip().lower() for c in df.columns]
    miss = [c for c in ("student_id", "task_id", "files") if c not in df.columns]
    if miss:
        raise ValueError(f"mapping sheet lacks columns: {miss}")
    tasks = {t["id"] for t in course.tasks()}
    summary = {"rows": len(df), "attempts": 0, "files": 0, "experts": 0, "skipped": []}
    groups = {}
    for i, r in df.iterrows():
        sid, tid = str(r["student_id"]).strip(), str(r["task_id"]).strip()
        if sid in ("", "nan") or tid in ("", "nan"):
            summary["skipped"].append((i + 2, "student_id/task_id empty"))
            continue
        if tid not in tasks:
            summary["skipped"].append((i + 2, f"unknown task {tid}"))
            continue
        att = int(_num(r.get("attempt", 1)) or 1)
        g = groups.setdefault((sid, tid, att), {"files": [], "e1": None, "e2": None, "note": "", "name": "", "group": "", "crop": {}})
        for f in str(r["files"]).split(";"):
            f = f.strip()
            if not f or f == "nan":
                continue
            fp = Path(f)
            if not fp.exists():
                summary["skipped"].append((i + 2, f"file not found: {f}"))
                continue
            if fp.suffix.lower() not in SUPPORTED:
                summary["skipped"].append((i + 2, f"unsupported (convert first): {fp.name}"))
                continue
            g["files"].append(fp)
            cr = str(r.get("crop", "")).strip()
            if cr and cr != "nan":
                try:
                    x0, y0, x1, y1 = [float(v) for v in cr.replace(";", ",").split(",")]
                    from PIL import Image as _I
                    W, H = _I.open(fp).size
                    g["crop"][fp.name] = {"x0": x0 / W, "y0": y0 / H, "x1": x1 / W, "y1": y1 / H}
                except Exception:
                    summary["skipped"].append((i + 2, f"bad crop '{cr}' (use x0,y0,x1,y1 in pixels)"))
        for k, col in (("e1", "expert1"), ("e2", "expert2")):
            v = _num(r.get(col, ""))
            if v is not None:
                try:
                    g[k] = rescale(v, scale, float(expert_min), float(expert_max))
                except ValueError as e:
                    summary["skipped"].append((i + 2, str(e)))
        for k in ("note", "name", "group"):
            v = str(r.get(k, "")).strip()
            if v and v != "nan":
                g[k] = v
    students = course.students()
    for (sid, tid, att), g in sorted(groups.items()):
        if not g["files"]:
            continue
        if sid not in students or g["name"]:
            course.save_student(sid, g["name"] or students.get(sid, {}).get("name", sid), g["group"] or students.get(sid, {}).get("group", ""))
            students = course.students()
        d = course.root / "submissions" / sid / tid / str(att)
        d.mkdir(parents=True, exist_ok=True)
        for f in g["files"]:
            (shutil.copy2 if copy else shutil.move)(f, d / f.name)
            summary["files"] += 1
        mt = min(f.stat().st_mtime for f in g["files"])
        (d / "submission.json").write_text(json.dumps({
            "student_id": sid, "task_id": tid, "attempt": att, "retrospective": True,
            "submitted_at": dt.datetime.fromtimestamp(mt).isoformat(timespec="seconds"),
            "plan": {}, "reflection": None, "files": [f.name for f in g["files"]], **({"crop": g["crop"]} if g["crop"] else {})},
            ensure_ascii=False, indent=2), encoding="utf-8")
        (d / "result.json").unlink(missing_ok=True)
        summary["attempts"] += 1
        if g["e1"] is not None or g["e2"] is not None:
            course.save_expert(sid, tid, att, g["e1"] if g["e1"] is not None else "", g["e2"] if g["e2"] is not None else "", g["note"])
            summary["experts"] += 1
    course.meta["retrospective"] = True
    (course.root / "course.json").write_text(json.dumps(course.meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"imported {summary['attempts']} attempts, {summary['files']} files, {summary['experts']} expert ratings; skipped {len(summary['skipped'])}")
    return summary
