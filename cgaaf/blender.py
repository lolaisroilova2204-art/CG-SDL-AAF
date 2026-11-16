"""Locate Blender and run standardised re-rendering / mesh export headlessly."""
from __future__ import annotations

import glob
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).with_name("blender_script.py")
BLENDER_INPUT = {".blend", ".fbx"}


def find_blender(hint: str | None = None):
    """Return ('exe', path) for a Blender executable or ('bpy', python) for a
    Python with the bpy module, or None."""
    for c in [hint, os.environ.get("CGAAF_BLENDER")]:
        if c and Path(c).exists():
            return ("exe", c)
    py = os.environ.get("CGAAF_BPY_PYTHON")
    if py and Path(py).exists():
        return ("bpy", py)
    w = shutil.which("blender")
    if w:
        return ("exe", w)
    cands = []
    if platform.system() == "Windows":
        for base in [os.environ.get("ProgramFiles", r"C:\Program Files"), os.environ.get("ProgramFiles(x86)", "")]:
            cands += glob.glob(os.path.join(base, "Blender Foundation", "Blender*", "blender.exe"))
        cands += glob.glob(os.path.expandvars(r"%LOCALAPPDATA%\Programs\Blender Foundation\Blender*\blender.exe"))
    elif platform.system() == "Darwin":
        cands += ["/Applications/Blender.app/Contents/MacOS/Blender"]
    else:
        cands += glob.glob("/opt/blender*/blender") + glob.glob(os.path.expanduser("~/blender*/blender"))
    cands = [c for c in cands if Path(c).exists()]
    if cands:
        return ("exe", sorted(cands)[-1])
    try:
        import bpy  # noqa: F401
        return ("bpy", sys.executable)
    except Exception:
        return None


MAKE_SCENE = Path(__file__).with_name("blender_make_scene.py")


def run_script(script, args, blender=None, timeout=600):
    b = blender or find_blender()
    if b is None:
        raise RuntimeError("Blender not found")
    cmd = ([b[1], "-b", "--factory-startup", "--python", str(script), "--"] if b[0] == "exe" else [b[1], str(script), "--"]) + [str(a) for a in args]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if "AAF_OK" not in (p.stdout or "") + (p.stderr or ""):
        raise RuntimeError(((p.stdout or "") + (p.stderr or ""))[-1500:])


def run_blender(src, out_png=None, out_obj=None, mode="canonical", res=512, engine="CYCLES", samples=32,
                blender=None, timeout=900, jitter=(0.0, 0.0, 1.0), obj_filter="auto", frames=1, helpers="hide"):
    """Render `src` and/or export its geometry. Returns dict(png, obj, log)."""
    b = blender or find_blender()
    if b is None:
        raise RuntimeError("Blender not found (set CGAAF_BLENDER or install Blender).")
    args = ["--", str(Path(src).resolve()), str(out_png or "-"), str(out_obj or "-"), mode, str(res), engine, str(samples)] + [str(v) for v in jitter] + [obj_filter or "auto", str(int(frames)), helpers]
    if b[0] == "exe":
        cmd = [b[1], "-b", "--factory-startup", "--python", str(SCRIPT)] + args
    else:
        cmd = [b[1], str(SCRIPT)] + args
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    log = (p.stdout or "") + (p.stderr or "")
    if "AAF_OK" not in log:
        tail = "\n".join(log.strip().splitlines()[-15:])
        raise RuntimeError(f"Blender failed for {Path(src).name}:\n{tail}")
    ver = next((ln.split(" ", 1)[1].strip() for ln in log.splitlines() if ln.startswith("AAF_BLENDER_VERSION")), None)
    objs = next((ln.split(" ", 1)[1].strip().split(",") for ln in log.splitlines() if ln.startswith("AAF_OBJECTS ")), [])
    pngs = [out_png] if out_png and frames <= 1 else (sorted(Path(out_png).parent.glob(Path(out_png).stem + "_f*.png")) if out_png else [])
    return {"png": out_png, "pngs": [str(p) for p in pngs], "obj": out_obj, "log": log, "blender_version": ver, "objects": objs}
