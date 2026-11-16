"""PySide6 desktop interface (Uzbek / English)."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, QSettings, Qt, QThreadPool, Signal, Slot
from PySide6.QtGui import QFont, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QGridLayout,
                               QGroupBox, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit, QListWidget, QMainWindow,
                               QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem,
                               QTabWidget, QTextBrowser, QToolBar, QVBoxLayout, QWidget)

from . import ALGO_VERSION, __version__
from .blender import find_blender
from .course import Course
from .feedback import render_feedback
from .i18n import tr

POOL = QThreadPool.globalInstance()


class _Sig(QObject):
    done = Signal(object)
    fail = Signal(str)
    progress = Signal(int, str)


class _Bridge(QObject):
    """Lives in the GUI thread, so queued signals run callbacks there."""
    def __init__(self, on_done, on_fail, on_progress=None):
        super().__init__()
        self.cb = (on_done, on_fail, on_progress)

    @Slot(object)
    def done(self, v):
        _LIVE.discard(self)
        self.cb[0](v)

    @Slot(str)
    def fail(self, m):
        _LIVE.discard(self)
        self.cb[1](m)

    @Slot(int, str)
    def progress(self, i, m):
        if self.cb[2]:
            self.cb[2](i, m)


_LIVE = set()


class Job(QRunnable):
    def __init__(self, fn, *a, **k):
        super().__init__()
        self.fn, self.a, self.k, self.sig = fn, a, k, _Sig()

    def run(self):
        try:
            self.sig.done.emit(self.fn(*self.a, **self.k))
        except Exception as e:
            self.sig.fail.emit(f"{e}\n\n{traceback.format_exc(limit=3)}")


def run_bg(fn, on_done, on_fail, *a, progress=None, **k):
    j = Job(fn, *a, **k)
    br = _Bridge(on_done, on_fail, progress)
    _LIVE.add(br)
    j.sig.done.connect(br.done, Qt.QueuedConnection)
    j.sig.fail.connect(br.fail, Qt.QueuedConnection)
    if progress:
        j.sig.progress.connect(br.progress, Qt.QueuedConnection)
        j.k["_progress"] = j.sig.progress.emit
    j.setAutoDelete(False)
    br.job = j
    POOL.start(j)
    return j


def pix(path, size=240):
    p = QPixmap(str(path))
    return p.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation) if not p.isNull() else QPixmap()


def cloud_png(points_path, ref_points=None, size=3.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    P = np.load(points_path)
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_subplot(projection="3d")
    if ref_points is not None:
        R = ref_points[:: max(1, len(ref_points) // 1500)]
        ax.scatter(R[:, 0], R[:, 1], R[:, 2], s=1, c="#2b59d9", alpha=.35)
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=1.5, c="#d2433f")
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1))
    out = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    fig.savefig(out, dpi=90, bbox_inches="tight")
    plt.close(fig)
    return out


def table(headers):
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    t.horizontalHeader().setStretchLastSection(True)
    t.setEditTriggers(QTableWidget.NoEditTriggers)
    t.setSelectionBehavior(QTableWidget.SelectRows)
    return t


def fill(t, rows):
    t.setRowCount(len(rows))
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            t.setItem(i, j, QTableWidgetItem("" if v is None else str(v)))


def open_path(p):
    p = str(p)
    if sys.platform.startswith("win"):
        os.startfile(p)  # noqa
    elif sys.platform == "darwin":
        subprocess.Popen(["open", p])
    else:
        subprocess.Popen(["xdg-open", p])


# ================================================================== task dialog
class TaskDialog(QDialog):
    def __init__(self, win, task=None):
        super().__init__(win)
        L = win.L
        self.setWindowTitle(tr("edit_task" if task else "add_task", L))
        self.setMinimumWidth(780)
        t = task or {}
        f = QFormLayout(self)
        self.id = QLineEdit(t.get("id", ""))
        self.id.setEnabled(task is None)
        self.tuz, self.ten = QLineEdit(t.get("title_uz", "")), QLineEdit(t.get("title_en", ""))
        self.duz, self.den = QPlainTextEdit(t.get("desc_uz", "")), QPlainTextEdit(t.get("desc_en", ""))
        for w in (self.duz, self.den):
            w.setMaximumHeight(70)
        self.type = QComboBox()
        for k in ("image", "mesh", "both"):
            self.type.addItem(tr("type_" + k, L), k)
        self.type.setCurrentIndex(["image", "mesh", "both"].index(t.get("type", "image")))
        self.files = {}
        rows = []
        for key in ("ref_image", "ref_mesh", "ref_blend"):
            le, b = QLineEdit(), QPushButton(tr("browse", L))
            flt = {"ref_image": "Images (*.png *.jpg *.jpeg *.webp)", "ref_mesh": "Meshes (*.obj *.stl *.ply *.off *.glb *.gltf)", "ref_blend": "Blender/FBX (*.blend *.fbx)"}[key]
            b.clicked.connect(lambda _=False, le=le, flt=flt: le.setText(QFileDialog.getOpenFileName(self, "", "", flt)[0] or le.text()))
            h = QHBoxLayout(); h.addWidget(le); h.addWidget(b)
            self.files[key] = le
            rows.append((tr(key, L), h))
        if t.get("reference"):
            for k, v in t["reference"].items():
                if k in ("image", "mesh"):
                    self.files["ref_" + k].setPlaceholderText(f"({v})")
        self.align = QComboBox()
        for k, lab in (("pca_icp", "al_pca"), ("icp", "al_icp"), ("none", "al_none")):
            self.align.addItem(tr(lab, L), k)
        self.align.setCurrentIndex(["pca_icp", "icp", "none"].index(t.get("align", "pca_icp")))
        self.rmode = QComboBox()
        self.rmode.addItem(tr("rm_canonical", L), "canonical"); self.rmode.addItem(tr("rm_scene", L), "scene")
        self.rmode.setCurrentIndex(0 if t.get("render", {}).get("mode", "canonical") == "canonical" else 1)
        self.trim = QCheckBox(tr("trim", L)); self.trim.setChecked(t.get("trim", True))
        self.maxa = QSpinBox(); self.maxa.setRange(1, 50); self.maxa.setValue(int(t.get("max_attempts", 5)))
        self.objs = QComboBox()
        for k in ("auto", "all", "names", "collection"):
            self.objs.addItem(tr({"auto": "obj_auto", "all": "obj_all", "names": "obj_names", "collection": "obj_coll"}[k], L), k)
        ov = str(t.get("objects", "auto"))
        kind = "names" if ov.startswith("names:") else "collection" if ov.startswith("collection:") else ov if ov in ("auto", "all") else "auto"
        self.objs.setCurrentIndex(["auto", "all", "names", "collection"].index(kind))
        self.objs_txt = QLineEdit(ov.split(":", 1)[1] if ":" in ov else "")
        oh = QHBoxLayout(); oh.addWidget(self.objs); oh.addWidget(self.objs_txt)
        self.frames = QSpinBox(); self.frames.setRange(1, 24); self.frames.setValue(int(t.get("frames", 1)))
        self.uicrop = QCheckBox(tr("ui_crop", L)); self.uicrop.setChecked(t.get("ui_crop", "auto") == "auto")
        self.csize = QCheckBox(tr("check_size", L)); self.csize.setChecked(bool(t.get("check_size", False)))
        self.rhelp = QCheckBox(tr("render_helpers", L)); self.rhelp.setChecked(bool(t.get("render_helpers", False)))
        self.deadline = QLineEdit(t.get("deadline", ""))
        for lab, w in [(tr("task_id", L), self.id), (tr("title_uz", L), self.tuz), (tr("title_en", L), self.ten), (tr("desc", L) + " (UZ)", self.duz),
                       (tr("desc", L) + " (EN)", self.den), (tr("type", L), self.type)] + rows + [
                       (tr("align", L), self.align), (tr("render_mode", L), self.rmode), (tr("objects", L), oh), (tr("frames", L), self.frames),
                       ("", self.trim), ("", self.uicrop), ("", self.csize), ("", self.rhelp), (tr("max_attempts", L), self.maxa), (tr("deadline", L), self.deadline)]:
            f.addRow(lab, w)
        bb = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        f.addRow(bb)
        self.base = t

    def value(self):
        t = dict(self.base)
        t.update({"id": re.sub(r"[^A-Za-z0-9_-]", "_", self.id.text().strip()) or "T1", "title_uz": self.tuz.text(), "title_en": self.ten.text(),
                  "desc_uz": self.duz.toPlainText(), "desc_en": self.den.toPlainText(), "type": self.type.currentData(), "align": self.align.currentData(),
                  "trim": self.trim.isChecked(), "max_attempts": self.maxa.value(), "deadline": self.deadline.text().strip(),
                  "frames": self.frames.value(), "ui_crop": "auto" if self.uicrop.isChecked() else None, "check_size": self.csize.isChecked(),
                  "render_helpers": self.rhelp.isChecked()})
        k = self.objs.currentData()
        t["objects"] = k if k in ("auto", "all") else f"{k}:{self.objs_txt.text().strip()}"
        t.setdefault("render", {"res": 512, "engine": "CYCLES", "samples": 32})
        t["render"]["mode"] = self.rmode.currentData()
        files = {k: (le.text().strip() or None) for k, le in self.files.items()}
        return t, files


# ================================================================== manual crop
class CropLabel(QLabel):
    def __init__(self, pm):
        super().__init__()
        from PySide6.QtWidgets import QRubberBand
        self.setPixmap(pm)
        self.band = QRubberBand(QRubberBand.Rectangle, self)
        self.origin, self.rect_ = None, None

    def mousePressEvent(self, e):
        self.origin = e.position().toPoint()
        from PySide6.QtCore import QRect, QSize
        self.band.setGeometry(QRect(self.origin, QSize()))
        self.band.show()

    def mouseMoveEvent(self, e):
        if self.origin is not None:
            from PySide6.QtCore import QRect
            self.band.setGeometry(QRect(self.origin, e.position().toPoint()).normalized())

    def mouseReleaseEvent(self, e):
        self.rect_ = self.band.geometry()


class CropDialog(QDialog):
    def __init__(self, win, img_path):
        super().__init__(win)
        L = win.L
        self.setWindowTitle(tr("manual_crop", L).rstrip("…"))
        full = QPixmap(str(img_path))
        self.W, self.H = full.width(), full.height()
        pm = full.scaled(1000, 700, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.k = pm.width() / max(1, self.W)
        v = QVBoxLayout(self)
        v.addWidget(QLabel(tr("crop_hint", L)))
        self.lab = CropLabel(pm)
        v.addWidget(self.lab)
        bb = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        self.clear = bb.addButton(tr("crop_clear", L), QDialogButtonBox.ResetRole)
        self.cleared = False
        self.clear.clicked.connect(lambda: (setattr(self, "cleared", True), self.accept()))
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        v.addWidget(bb)

    def value(self):
        if self.cleared or self.lab.rect_ is None or self.lab.rect_.width() < 8:
            return None
        r = self.lab.rect_
        return {"x0": max(0, r.left() / self.k / self.W), "y0": max(0, r.top() / self.k / self.H),
                "x1": min(1, r.right() / self.k / self.W), "y1": min(1, r.bottom() / self.k / self.H)}


# ================================================================== main window
class Main(QMainWindow):
    def __init__(self, lang="uz"):
        super().__init__()
        self.st = QSettings("NDU", "CG-SDL-AAF")
        self.L = self.st.value("lang", lang)
        if self.st.value("blender", ""):
            os.environ["CGAAF_BLENDER"] = self.st.value("blender")
        self.course = None
        self.resize(1240, 820)
        tb = QToolBar(); self.addToolBar(tb)
        self.tb = tb
        last = self.st.value("course", "")
        if last and Path(last, "course.json").exists():
            self.load_course(last, build=False)
        self.build()

    def T(self, k, *a):
        return tr(k, self.L, *a)

    def title(self, t):
        return (t.get("title_uz") if self.L == "uz" else t.get("title_en")) or t.get("title_uz") or t["id"]

    # ------------------------------------------------------------ shell
    def build(self):
        self.setWindowTitle(f"{self.T('app_title')} · v{__version__}")
        self.tb.clear()
        for k, fn in (("open_course", self.open_course), ("new_course", self.new_course), ("demo", self.demo)):
            b = QPushButton(self.T(k)); b.clicked.connect(fn); self.tb.addWidget(b)
        self.tb.addSeparator()
        self.tb.addWidget(QLabel(" " + self.T("lang") + ": "))
        lc = QComboBox(); lc.addItems(["O'zbekcha", "English"]); lc.setCurrentIndex(0 if self.L == "uz" else 1)
        lc.currentIndexChanged.connect(lambda i: self.set_lang("uz" if i == 0 else "en"))
        self.tb.addWidget(lc)
        self.course_lbl = QLabel()
        self.tb.addSeparator(); self.tb.addWidget(self.course_lbl)
        tabs = QTabWidget()
        self.tabs = tabs
        if not self.course:
            w = QLabel(self.T("no_course")); w.setAlignment(Qt.AlignCenter)
            tabs.addTab(w, self.T("course"))
        else:
            self.course_lbl.setText(f"  📁 {self.course.meta.get('name')}  ({self.course.root})" + ("  ⚠ SYNTHETIC" if self.course.meta.get("synthetic") else ""))
            tabs.addTab(self.tasks_tab(), self.T("tasks"))
            tabs.addTab(self.student_tab(), self.T("student_tab"))
            tabs.addTab(self.batch_tab(), self.T("batch_tab"))
            tabs.addTab(self.expert_tab(), self.T("expert_tab"))
            tabs.addTab(self.valid_tab(), self.T("valid_tab"))
            tabs.addTab(self.paper_tab(), self.T("paper_tab"))
            tabs.addTab(self.export_tab(), self.T("export_tab"))
        self.setCentralWidget(tabs)
        self.statusBar().showMessage(f"{ALGO_VERSION} · Blender: {find_blender(self.st.value('blender', None)) or '—'}")

    def set_lang(self, L):
        self.L = L; self.st.setValue("lang", L); self.build()

    def err(self, msg):
        QMessageBox.critical(self, self.T("error"), str(msg)[:2500])

    def load_course(self, path, build=True):
        try:
            self.course = Course(path)
            bp = self.st.value("blender", "")
            if bp:
                self.course.blender = ("exe", bp)
            self.st.setValue("course", str(path))
        except Exception as e:
            self.err(e)
        if build:
            self.build()

    def open_course(self):
        d = QFileDialog.getExistingDirectory(self, self.T("open_course"))
        if d:
            self.load_course(d)

    def new_course(self):
        d = QFileDialog.getExistingDirectory(self, self.T("new_course"))
        if not d:
            return
        name, ok = QInputDialog.getText(self, self.T("new_course"), self.T("course_name"), text="Kompyuter grafikasi")
        if ok:
            Course.create(d, name, self.L)
            self.load_course(d)

    def demo(self):
        d = QFileDialog.getExistingDirectory(self, self.T("demo"))
        if not d:
            return
        d = str(Path(d) / "CG_demo_course")
        QMessageBox.information(self, self.T("demo"), self.T("synthetic_warn"))
        from .demo import make_demo
        self.statusBar().showMessage(self.T("processing"))
        run_bg(lambda: make_demo(d, 24, lang=self.L, log=lambda s: None), lambda _: self.load_course(d), self.err)

    # ------------------------------------------------------------ tasks
    def tasks_tab(self):
        w = QWidget(); v = QVBoxLayout(w)
        self.ttable = table(["ID", self.T("title_uz" if self.L == "uz" else "title_en"), self.T("type"), self.T("deadline"), self.T("max_attempts"), "Model"])
        v.addWidget(self.ttable)
        h = QHBoxLayout()
        for k, fn in (("add_task", lambda: self.edit_task(None)), ("edit_task", self.edit_selected), ("delete", self.delete_task)):
            b = QPushButton(self.T(k)); b.clicked.connect(fn); h.addWidget(b)
        h.addStretch(); v.addLayout(h)
        self.ttable.doubleClicked.connect(self.edit_selected)
        self.refresh_tasks()
        return w

    def refresh_tasks(self):
        self.task_list = self.course.tasks()
        fill(self.ttable, [[t["id"], self.title(t), self.T("type_" + t["type"]), t.get("deadline", ""), t.get("max_attempts", ""),
                            "calibrated (n=%d)" % t["calibration"]["n"] if t.get("calibration") else "default"] for t in self.task_list])

    def edit_selected(self):
        r = self.ttable.currentRow()
        if r >= 0:
            self.edit_task(self.task_list[r])

    def edit_task(self, task):
        dlg = TaskDialog(self, task)
        if dlg.exec() != QDialog.Accepted:
            return
        t, f = dlg.value()
        if not task and not any(f.values()):
            return self.err(self.T("need_files"))
        self.statusBar().showMessage(self.T("processing"))
        run_bg(self.course.save_task, lambda _: (self.refresh_tasks(), self.statusBar().showMessage(self.T("saved"))), self.err,
               t, f["ref_image"], f["ref_mesh"], f["ref_blend"])

    def delete_task(self):
        r = self.ttable.currentRow()
        if r >= 0 and QMessageBox.question(self, self.T("delete"), self.task_list[r]["id"] + " ?") == QMessageBox.Yes:
            self.course.delete_task(self.task_list[r]["id"]); self.refresh_tasks()

    # ------------------------------------------------------------ student (SRL)
    def student_tab(self):
        w = QWidget(); g = QGridLayout(w)
        left = QVBoxLayout()
        self.s_student, self.s_task = QComboBox(), QComboBox()
        bnew = QPushButton(self.T("new_student")); bnew.clicked.connect(self.new_student)
        hs = QHBoxLayout(); hs.addWidget(self.s_student, 1); hs.addWidget(bnew)
        left.addWidget(QLabel(self.T("student"))); left.addLayout(hs)
        left.addWidget(QLabel(self.T("tasks"))); left.addWidget(self.s_task)
        gb1 = QGroupBox(self.T("phase1")); f1 = QFormLayout(gb1)
        self.p_goal = QLineEdit(); self.p_min = QSpinBox(); self.p_min.setRange(0, 1000); self.p_pred = QSpinBox(); self.p_pred.setRange(-1, 100); self.p_pred.setValue(-1)
        self.p_pred.setSpecialValueText("—")
        f1.addRow(self.T("plan_goal"), self.p_goal); f1.addRow(self.T("plan_min"), self.p_min); f1.addRow(self.T("plan_pred"), self.p_pred)
        left.addWidget(gb1)
        gb2 = QGroupBox(self.T("phase2")); v2 = QVBoxLayout(gb2)
        self.s_files = QListWidget(); self.s_files.setMaximumHeight(70)
        bf = QPushButton(self.T("files")); bf.clicked.connect(self.pick_files)
        self.lp = QCheckBox(self.T("use_lpips")); self.lp.setChecked(True)
        self.s_go = QPushButton(self.T("evaluate")); self.s_go.setStyleSheet("font-weight:600;padding:6px")
        self.s_go.clicked.connect(self.do_submit)
        for x in (bf, self.s_files, self.lp, self.s_go):
            v2.addWidget(x)
        left.addWidget(gb2)
        self.hist = table([self.T("attempt"), self.T("score"), self.T("grade"), "source"])
        self.hist.itemSelectionChanged.connect(self.show_attempt)
        left.addWidget(self.hist)
        right = QVBoxLayout()
        self.r_score = QLabel("—"); self.r_score.setFont(QFont("", 30, QFont.Bold))
        imgs = QHBoxLayout()
        self.i_ref, self.i_stu, self.i_heat = QLabel(), QLabel(), QLabel()
        for lab, cap in ((self.i_ref, "reference"), (self.i_stu, "submission"), (self.i_heat, "heatmap")):
            bx = QVBoxLayout(); bx.addWidget(lab); c = QLabel(self.T(cap) if cap != "heatmap" else "SSIM map"); c.setAlignment(Qt.AlignCenter); bx.addWidget(c); imgs.addLayout(bx)
        self.r_metrics = table([self.T("metrics"), "m ∈ [0,1]", "raw"])
        self.r_metrics.setMaximumHeight(240)
        self.r_fb = QTextBrowser(); self.r_fb.setMaximumHeight(150)
        gb3 = QGroupBox(self.T("phase3")); v3 = QVBoxLayout(gb3)
        self.refl = QPlainTextEdit(); self.refl.setPlaceholderText(self.T("reflection")); self.refl.setMaximumHeight(90)
        br = QPushButton(self.T("save_reflection")); br.clicked.connect(self.save_refl)
        v3.addWidget(self.refl); v3.addWidget(br)
        right.addWidget(self.r_score); right.addLayout(imgs); right.addWidget(self.r_metrics); right.addWidget(QLabel(self.T("feedback"))); right.addWidget(self.r_fb); right.addWidget(gb3)
        g.addLayout(left, 0, 0); g.addLayout(right, 0, 1); g.setColumnStretch(1, 2)
        self.s_student.currentIndexChanged.connect(self.refresh_hist); self.s_task.currentIndexChanged.connect(self.refresh_hist)
        self.reload_student_lists()
        return w

    def reload_student_lists(self):
        self.s_student.blockSignals(True); self.s_task.blockSignals(True)
        self.s_student.clear(); self.s_task.clear()
        for sid, s in self.course.students().items():
            self.s_student.addItem(f"{sid} — {s.get('name', '')} ({s.get('group', '')})", sid)
        for t in self.course.tasks():
            self.s_task.addItem(f"{t['id']} — {self.title(t)}", t["id"])
        self.s_student.blockSignals(False); self.s_task.blockSignals(False)
        self.refresh_hist()

    def new_student(self):
        dlg = QDialog(self); f = QFormLayout(dlg)
        n, g, p = QLineEdit(), QLineEdit(), QLineEdit()
        f.addRow(self.T("name"), n); f.addRow(self.T("group"), g); f.addRow(self.T("program"), p)
        bb = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel); bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject); f.addRow(bb)
        if dlg.exec() == QDialog.Accepted and n.text().strip():
            sid = self.course.next_student_id()
            self.course.save_student(sid, n.text().strip(), g.text().strip(), p.text().strip())
            self.reload_student_lists()
            self.s_student.setCurrentIndex(self.s_student.findData(sid))

    def pick_files(self):
        fs, _ = QFileDialog.getOpenFileNames(self, self.T("files"), "", "Submissions (*.png *.jpg *.jpeg *.webp *.obj *.stl *.ply *.off *.glb *.gltf *.blend *.fbx)")
        if fs:
            self.s_files.clear(); self.s_files.addItems(fs)

    def do_submit(self):
        sid, tid = self.s_student.currentData(), self.s_task.currentData()
        files = [self.s_files.item(i).text() for i in range(self.s_files.count())]
        if not sid or not tid or not files:
            return self.err(self.T("need_files"))
        if not self.p_goal.text().strip() or self.p_pred.value() < 0:
            return self.err(self.T("need_plan"))
        plan = {"goal": self.p_goal.text().strip(), "minutes": self.p_min.value(), "pred": self.p_pred.value()}
        self.s_go.setEnabled(False); self.s_go.setText(self.T("processing"))

        def done(_):
            self.s_go.setEnabled(True); self.s_go.setText(self.T("evaluate")); self.p_pred.setValue(-1); self.s_files.clear()
            self.refresh_hist(select_last=True)

        def fail(m):
            self.s_go.setEnabled(True); self.s_go.setText(self.T("evaluate"))
            self.err(self.T("no_attempts") if "max_attempts" in m else m)
        run_bg(self.course.submit, done, fail, sid, tid, files, plan, self.lp.isChecked())

    def refresh_hist(self, *_, select_last=False):
        sid, tid = self.s_student.currentData(), self.s_task.currentData()
        self.cur_attempts = self.course.attempts(sid, tid) if sid and tid else []
        rows = []
        for d in self.cur_attempts:
            r = self.course.result(d) or {}
            rows.append([d.name, f"{r.get('score', float('nan')):.1f}" if r else "—", r.get("grade", "—"),
                         (r.get("image") or {}).get("source") or (r.get("mesh") or {}).get("source") or ""])
        fill(self.hist, rows)
        if self.cur_attempts:
            self.hist.selectRow(len(rows) - 1 if select_last or True else 0)
        else:
            self.clear_result()

    def clear_result(self):
        self.r_score.setText("—"); self.r_fb.clear(); self.r_metrics.setRowCount(0); self.refl.clear()
        for l in (self.i_ref, self.i_stu, self.i_heat):
            l.clear()

    def show_attempt(self):
        i = self.hist.currentRow()
        if i < 0 or i >= len(self.cur_attempts):
            return
        d = self.cur_attempts[i]
        r, s = self.course.result(d), self.course.submission(d)
        t = self.course.task(d.parent.name)
        if not r:
            return self.clear_result()
        col = {5: "#1d9a5b", 4: "#2b59d9", 3: "#c98400", 2: "#d2433f"}[r["grade"]]
        self.r_score.setText(f"<span style='color:{col}'>{r['score']:.0f}</span> <span style='font-size:14pt'>/ 100 · {self.T('grade')}: {r['grade']}</span>")
        tdir = self.course.task_dir(t["id"])
        if r.get("image"):
            self.i_ref.setPixmap(pix(tdir / t["reference"]["image"])); self.i_stu.setPixmap(pix(d / "aaf_norm.png")); self.i_heat.setPixmap(pix(d / "aaf_heatmap.png"))
        elif (d / "aaf_cloud.npy").exists():
            self.i_ref.setText("—"); self.i_heat.setText("—")
            self.i_stu.setPixmap(pix(cloud_png(d / "aaf_cloud.npy", self.course.ref_points(t))))
        rows = []
        raw_i, raw_m = (r.get("image") or {}).get("raw", {}), (r.get("mesh") or {}).get("raw", {})
        rawmap = {"ssim": raw_i.get("ssim"), "msssim": raw_i.get("msssim"), "lpips": raw_i.get("lpips"), "color": raw_i.get("deltaE00"), "edge": raw_i.get("edge_f1"),
                  "cd": raw_m.get("cd"), "h95": raw_m.get("h95"), "fscore": raw_m.get("fscore"), "topo": raw_m.get("boundary_edges")}
        for k, v in r["m"].items():
            rv = rawmap.get(k)
            rows.append([self.T("m_" + k), f"{v:.3f}", "" if rv is None else (f"{rv:.4f}" if isinstance(rv, float) else rv)])
        if raw_m:
            rows.append(["V / F / watertight", "", f"{raw_m['V']} / {raw_m['F']} / {raw_m['watertight']}"])
        fill(self.r_metrics, rows)
        self.r_fb.setHtml("<ul>" + "".join(f"<li>{x}</li>" for x in render_feedback(r["feedback"], self.L)) + "</ul>")
        self.refl.setPlainText(((s or {}).get("reflection") or {}).get("text", ""))

    def save_refl(self):
        i = self.hist.currentRow()
        if 0 <= i < len(self.cur_attempts):
            self.course.save_reflection(self.cur_attempts[i], self.refl.toPlainText().strip())
            self.statusBar().showMessage(self.T("saved"), 3000)

    # ------------------------------------------------------------ batch
    def batch_tab(self):
        w = QWidget(); v = QVBoxLayout(w)
        h = QHBoxLayout()
        self.b_force = QCheckBox(self.T("force")); self.b_lp = QCheckBox(self.T("use_lpips")); self.b_lp.setChecked(True)
        self.b_go = QPushButton(self.T("evaluate_all")); self.b_go.clicked.connect(self.do_batch)
        self.b_crop = QPushButton(self.T("manual_crop")); self.b_crop.clicked.connect(self.do_crop)
        for x in (self.b_force, self.b_lp, self.b_go, self.b_crop):
            h.addWidget(x)
        h.addStretch(); v.addLayout(h)
        self.b_prog = QProgressBar(); v.addWidget(self.b_prog)
        self.b_table = table([self.T("student"), self.T("tasks"), self.T("attempt"), self.T("score"), self.T("grade"), "source", "s"])
        v.addWidget(self.b_table)
        self.b_log = QPlainTextEdit(); self.b_log.setReadOnly(True); self.b_log.setMaximumHeight(120); v.addWidget(self.b_log)
        hint = QLabel("submissions/<student_id>/<task_id>/<attempt>/  ← " + self.T("files")); hint.setStyleSheet("color:#667085"); v.addWidget(hint)
        self.refresh_batch()
        return w

    def do_crop(self):
        r = self.b_table.currentRow()
        if r < 0 or r >= len(self.batch_dirs):
            return
        d = self.batch_dirs[r]
        from .course import IMG_EXT
        imgs = sorted(f for f in d.iterdir() if f.suffix.lower() in IMG_EXT and not f.name.startswith("aaf_"))
        if not imgs:
            return self.err(self.T("need_files"))
        f = imgs[0]
        if len(imgs) > 1:
            name, ok = QInputDialog.getItem(self, self.T("manual_crop"), self.T("files"), [x.name for x in imgs], 0, False)
            if not ok:
                return
            f = d / name
        dlg = CropDialog(self, f)
        if dlg.exec() != QDialog.Accepted:
            return
        self.course.set_crop(d, dlg.value(), file=f.name if dlg.value() else None)
        try:
            r_ = self.course.evaluate_attempt(d, use_lpips=self.b_lp.isChecked())
            self.b_log.appendPlainText(f"{d.relative_to(self.course.root)}  crop → {r_['score']:.1f}")
        except Exception as e:
            self.err(e)
        self.refresh_batch()

    def refresh_batch(self):
        rows = []
        self.batch_dirs = self.course.attempt_dirs()
        for d in self.batch_dirs:
            r = self.course.result(d)
            rows.append([d.parent.parent.name, d.parent.name, d.name, f"{r['score']:.1f}" if r else self.T("status_pending"), r["grade"] if r else "",
                         (((r or {}).get("image") or {}).get("source") or ((r or {}).get("mesh") or {}).get("source") or "")
                         + (" ✂" if ((r or {}).get("image") or {}).get("crop") else ""), (r or {}).get("seconds", "")])
        fill(self.b_table, rows)

    def do_batch(self):
        P = self.course.pending(self.b_force.isChecked())
        self.b_prog.setMaximum(max(1, len(P))); self.b_prog.setValue(0); self.b_go.setEnabled(False)
        self.b_log.appendPlainText(f"{len(P)} {self.T('pending')}")
        lp = self.b_lp.isChecked()

        def work(_progress):
            for i, d in enumerate(P, 1):
                try:
                    r = self.course.evaluate_attempt(d, use_lpips=lp)
                    _progress(i, f"{d.relative_to(self.course.root)}  {r['score']:.1f}")
                except Exception as e:
                    _progress(i, f"{d.relative_to(self.course.root)}  {self.T('error')}: {e}")
            return len(P)

        def prog(i, msg):
            self.b_prog.setValue(i); self.b_log.appendPlainText(msg)
        run_bg(work, lambda n: (self.b_go.setEnabled(True), self.refresh_batch(), self.b_log.appendPlainText(self.T("done"))),
               lambda m: (self.b_go.setEnabled(True), self.err(m)), progress=prog)

    # ------------------------------------------------------------ experts
    def expert_tab(self):
        w = QWidget(); v = QVBoxLayout(w)
        h = QHBoxLayout()
        b1 = QPushButton(self.T("expert_template")); b1.clicked.connect(self.make_template)
        b2 = QPushButton(self.T("expert_import")); b2.clicked.connect(self.import_experts)
        h.addWidget(b1); h.addWidget(b2); h.addStretch(); v.addLayout(h)
        gb = QGroupBox(self.T("blind_rating")); g = QGridLayout(gb)
        self.e_info = QLabel(); self.e_ref, self.e_stu = QLabel(), QLabel()
        self.e1, self.e2, self.en = QLineEdit(), QLineEdit(), QLineEdit()
        prev, nxt, sv = QPushButton(self.T("prev")), QPushButton(self.T("next")), QPushButton(self.T("save"))
        prev.clicked.connect(lambda: self.e_move(-1)); nxt.clicked.connect(lambda: self.e_move(1)); sv.clicked.connect(self.e_save)
        g.addWidget(self.e_info, 0, 0, 1, 4)
        g.addWidget(QLabel(self.T("reference")), 1, 0); g.addWidget(QLabel(self.T("submission")), 1, 1)
        g.addWidget(self.e_ref, 2, 0); g.addWidget(self.e_stu, 2, 1)
        f = QFormLayout(); f.addRow(self.T("expert1"), self.e1); f.addRow(self.T("expert2"), self.e2); f.addRow(self.T("note"), self.en)
        hb = QHBoxLayout(); hb.addWidget(prev); hb.addWidget(sv); hb.addWidget(nxt); f.addRow(hb)
        g.addLayout(f, 2, 2)
        v.addWidget(gb); v.addStretch()
        groups = {}
        for d in self.course.attempt_dirs():
            groups.setdefault((d.parent.parent.name, d.parent.name), []).append(d)
        self.e_items = [ds[-1] for _, ds in sorted(groups.items())]
        self.e_idx = 0
        self.e_show()
        return w

    def e_show(self):
        if not self.e_items:
            self.e_info.setText(self.T("status_pending")); return
        from .course import anon_code
        d = self.e_items[self.e_idx]
        sid, tid, att = d.parent.parent.name, d.parent.name, int(d.name)
        t = self.course.task(tid)
        ex = self.course.experts().get((sid, tid, att), {})
        self.e_info.setText(f"<b>{self.e_idx + 1} / {len(self.e_items)}</b> · {anon_code(sid)} · {tid} · {self.T('attempt')} {att}")
        tdir = self.course.task_dir(tid)
        if t["type"] != "mesh" and (d / "aaf_norm.png").exists():
            self.e_ref.setPixmap(pix(tdir / t["reference"]["image"], 320)); self.e_stu.setPixmap(pix(d / "aaf_norm.png", 320))
        elif (d / "aaf_cloud.npy").exists():
            self.e_ref.clear(); self.e_stu.setPixmap(pix(cloud_png(d / "aaf_cloud.npy", self.course.ref_points(t)), 320))
        self.e1.setText(str(ex.get("expert1", ""))); self.e2.setText(str(ex.get("expert2", ""))); self.en.setText(ex.get("note", ""))

    def e_save(self):
        if not self.e_items:
            return
        d = self.e_items[self.e_idx]
        num = lambda s: "" if not s.strip() else max(0.0, min(100.0, float(s.replace(",", "."))))
        try:
            self.course.save_expert(d.parent.parent.name, d.parent.name, int(d.name), num(self.e1.text()), num(self.e2.text()), self.en.text())
            self.statusBar().showMessage(self.T("saved"), 2000)
        except ValueError as e:
            self.err(e)

    def e_move(self, k):
        if self.e_items:
            self.e_save(); self.e_idx = (self.e_idx + k) % len(self.e_items); self.e_show()

    def make_template(self):
        p, _ = QFileDialog.getSaveFileName(self, "", str(self.course.root / "results" / "expert_rating_sheet.xlsx"), "Excel (*.xlsx)")
        if p:
            n = self.course.expert_template(p); QMessageBox.information(self, self.T("done"), f"{n} → {p}")

    def import_experts(self):
        p, _ = QFileDialog.getOpenFileName(self, "", "", "Excel/CSV (*.xlsx *.csv)")
        if p:
            try:
                QMessageBox.information(self, self.T("done"), f"{self.course.import_experts(p)} {self.T('saved')}")
            except Exception as e:
                self.err(e)

    # ------------------------------------------------------------ validation
    def valid_tab(self):
        w = QWidget(); v = QVBoxLayout(w)
        h = QHBoxLayout()
        self.v_task, self.v_model = QComboBox(), QComboBox()
        self.v_task.addItem(self.T("all_tasks"), None)
        for t in self.course.tasks():
            self.v_task.addItem(f"{t['id']} — {self.title(t)}", t["id"])
        self.v_model.addItem("stored", "stored"); self.v_model.addItem("current model", "current")
        bv, bc, ba, bo = QPushButton(self.T("run_validation")), QPushButton(self.T("calibrate")), QPushButton("Analytics"), QPushButton(self.T("open_folder"))
        bv.clicked.connect(self.do_validate); bc.clicked.connect(self.do_calibrate); ba.clicked.connect(self.do_analytics)
        bo.clicked.connect(lambda: open_path(getattr(self, "last_out", self.course.root / "results")))
        for x in (self.v_task, self.v_model, bv, bc, ba, bo):
            h.addWidget(x)
        v.addLayout(h)
        hv = QHBoxLayout()
        self.v_text = QTextBrowser(); self.v_fig = QLabel(); self.v_fig.setMinimumWidth(420); self.v_fig.setAlignment(Qt.AlignTop)
        hv.addWidget(self.v_text, 3); hv.addWidget(self.v_fig, 2)
        v.addLayout(hv)
        return w

    def do_validate(self):
        from .reports import validation_report
        self.statusBar().showMessage(self.T("processing"))

        def done(res):
            rep, out = res
            self.last_out = out
            md = out / f"report_{self.L}.md"
            if md.exists():
                self.v_text.setMarkdown(md.read_text(encoding="utf-8"))
            else:
                self.v_text.setPlainText(str(rep.get("error")))
            if rep.get("figures"):
                self.v_fig.setPixmap(pix(rep["figures"]["scatter"], 420))
            self.statusBar().showMessage(str(out), 8000)
        run_bg(validation_report, done, self.err, self.course, self.v_task.currentData(), self.v_model.currentData(), None, self.L)

    def do_calibrate(self):
        tid = self.v_task.currentData()
        if not tid:
            return self.err(self.T("all_tasks") + " → 1")
        from .scoring import calibrate, keys_for
        df = self.course.dataset()
        d = df[(df["task_id"] == tid) & df["expert_mean"].notna()]
        if len(d) < 8:
            return self.err("n < 8")
        t = self.course.task(tid)
        M = [{k[2:]: row[k] for k in d.columns if k.startswith("m_") and row[k] == row[k]} for _, row in d.iterrows()]
        res = calibrate(M, d["expert_mean"].values, keys_for(t["type"]))
        msg = f"b0 = {res['b0']:.2f}\n" + "\n".join(f"b[{k}] = {b:.2f}" for k, b in res["b"].items()) + \
              f"\n\nLOOCV r = {res['r_loo']:.3f}, MAE = {res['mae_loo']:.2f} (n = {res['n']})\n\n{self.T('save')}?"
        if QMessageBox.question(self, self.T("calibrate"), msg) == QMessageBox.Yes:
            t["calibration"] = {k: res[k] for k in ("b0", "b", "n", "r_loo", "mae_loo")}
            self.course.save_task(t); self.refresh_tasks()

    def do_analytics(self):
        from .reports import analytics_report

        def done(res):
            T, out = res
            self.last_out = out
            self.v_text.setMarkdown("## Learning analytics\n\n" + T.round(3).to_markdown(index=False) if hasattr(T, "to_markdown") else T.to_string())
            f = out / "fig_learning_gain.png"
            if f.exists():
                self.v_fig.setPixmap(pix(f, 420))
        run_bg(analytics_report, done, self.err, self.course)

    # ------------------------------------------------------------ paper (variant 2)
    def _dir_row(self, key):
        le, b = QLineEdit(), QPushButton(self.T("browse"))
        b.clicked.connect(lambda: le.setText(QFileDialog.getExistingDirectory(self, self.T(key)) or le.text()))
        h = QHBoxLayout(); h.addWidget(le); h.addWidget(b)
        return le, h

    def paper_tab(self):
        w = QWidget(); v = QVBoxLayout(w)
        g1 = QGroupBox(self.T("archive_box")); f1 = QFormLayout(g1)
        bt = QPushButton(self.T("archive_template")); bt.clicked.connect(self.do_archive_template)
        self.a_max = QSpinBox(); self.a_max.setRange(1, 1000); self.a_max.setValue(100)
        self.a_min = QSpinBox(); self.a_min.setRange(0, 999); self.a_min.setValue(0)
        self.a_scale = QComboBox(); self.a_scale.addItem(self.T("scale_linear"), "linear"); self.a_scale.addItem(self.T("scale_grade5"), "grade5")
        bi = QPushButton(self.T("archive_import")); bi.clicked.connect(self.do_archive_import)
        hint = QLabel(self.T("archive_hint")); hint.setWordWrap(True); hint.setStyleSheet("color:#667085")
        f1.addRow(bt); f1.addRow(hint); f1.addRow(self.T("scale"), self.a_scale); f1.addRow(self.T("expert_min"), self.a_min); f1.addRow(self.T("expert_max"), self.a_max); f1.addRow(bi)
        g2 = QGroupBox(self.T("bench_box")); f2 = QFormLayout(g2)
        self.bi_dir, h1 = self._dir_row("bench_imgs"); self.bm_dir, h2 = self._dir_row("bench_meshes")
        self.bb = QCheckBox(self.T("bench_blender")); self.bb.setChecked(find_blender(self.st.value("blender", None)) is not None)
        self.blp = QCheckBox(self.T("use_lpips")); self.blp.setChecked(True)
        self.b_run = QPushButton(self.T("bench_run")); self.b_run.clicked.connect(self.do_bench)
        f2.addRow(self.T("bench_imgs"), h1); f2.addRow(self.T("bench_meshes"), h2); f2.addRow(self.bb); f2.addRow(self.blp); f2.addRow(self.b_run)
        g3 = QGroupBox(self.T("paper_box")); f3 = QFormLayout(g3)
        self.pb_dir, h3 = self._dir_row("bench_dir")
        last = self.course.root / "results" / "benchmark"
        if last.exists():
            self.pb_dir.setText(str(last))
        bp = QPushButton(self.T("paper_make")); bp.clicked.connect(self.do_paper)
        f3.addRow(self.T("bench_dir"), h3); f3.addRow(bp)
        self.p_log = QPlainTextEdit(); self.p_log.setReadOnly(True)
        for x in (g1, g2, g3, self.p_log):
            v.addWidget(x)
        return w

    def do_archive_template(self):
        from .archive import make_mapping_template
        d = QFileDialog.getExistingDirectory(self, self.T("archive_template"))
        if not d:
            return
        out, _ = QFileDialog.getSaveFileName(self, "", str(self.course.root / "archive_mapping.xlsx"), "Excel (*.xlsx)")
        if out:
            df = make_mapping_template(d, out)
            self.p_log.appendPlainText(f"{len(df)} files → {out}  (CONVERT: {int(df['status'].str.startswith('CONVERT').sum())})")
            open_path(out)

    def do_archive_import(self):
        from .archive import import_mapping
        p, _ = QFileDialog.getOpenFileName(self, "", str(self.course.root), "Excel/CSV (*.xlsx *.csv)")
        if not p:
            return
        mx, mn, sc, lp = float(self.a_max.value()), float(self.a_min.value()), self.a_scale.currentData(), self.blp.isChecked()

        def work():
            s = import_mapping(self.course, p, mx, log=lambda m: None, expert_min=mn, scale=sc)
            n = 0
            for d in self.course.pending():
                try:
                    self.course.evaluate_attempt(d, use_lpips=lp); n += 1
                except Exception as e:
                    s["skipped"].append((str(d.relative_to(self.course.root)), str(e)[:200]))
            s["evaluated"] = n
            return s

        def done(s):
            self.p_log.appendPlainText(f"attempts {s['attempts']}, files {s['files']}, expert ratings {s['experts']}, evaluated {s['evaluated']}")
            for r, why in s["skipped"][:40]:
                self.p_log.appendPlainText(f"  ! {r}: {why}")
            self.build(); self.tabs.setCurrentIndex(6)
        self.p_log.appendPlainText(self.T("processing")); run_bg(work, done, self.err)

    def do_bench(self):
        from .benchmark import run_benchmark
        out = self.course.root / "results" / "benchmark"
        self.b_run.setEnabled(False); self.p_log.appendPlainText(self.T("processing") + " (1–5 min)")

        def done(r):
            self.b_run.setEnabled(True); self.pb_dir.setText(str(r["out"]))
            self.p_log.appendPlainText(json.loads((r["out"] / "benchmark_summary.json").read_text())["paragraph"]); self.last_out = r["out"]
        run_bg(run_benchmark, done, lambda m: (self.b_run.setEnabled(True), self.err(m)), out, self.bi_dir.text() or None, self.bm_dir.text() or None,
               self.bb.isChecked(), 3, self.blp.isChecked(), log=lambda s: None)

    def do_paper(self):
        from .paper import make_paper_bundle

        def done(out):
            self.p_log.appendPlainText(f"→ {out}"); self.last_out = out; open_path(out)
        self.p_log.appendPlainText(self.T("processing"))
        run_bg(make_paper_bundle, done, self.err, self.course, self.pb_dir.text() or None, None, log=lambda s: None)

    # ------------------------------------------------------------ export
    def export_tab(self):
        w = QWidget(); v = QVBoxLayout(w)
        self.x_anon = QCheckBox(self.T("anonymise")); self.x_anon.setChecked(True)
        bx = QPushButton(self.T("export_csv")); bx.clicked.connect(self.do_export)
        v.addWidget(self.x_anon); v.addWidget(bx)
        gb = QGroupBox(self.T("blender_path")); h = QHBoxLayout(gb)
        self.x_bl = QLineEdit(self.st.value("blender", "")); bb = QPushButton(self.T("browse"))
        bb.clicked.connect(lambda: self.x_bl.setText(QFileDialog.getOpenFileName(self, "Blender")[0] or self.x_bl.text()))
        bs = QPushButton(self.T("save")); bs.clicked.connect(self.save_blender)
        h.addWidget(self.x_bl); h.addWidget(bb); h.addWidget(bs)
        v.addWidget(gb)
        info = QLabel(f"CG-SDL-AAF {__version__} · {ALGO_VERSION}\nBlender: {find_blender(self.st.value('blender', None)) or self.T('blender_missing')}")
        info.setWordWrap(True); v.addWidget(info); v.addStretch()
        return w

    def save_blender(self):
        self.st.setValue("blender", self.x_bl.text().strip())
        if self.x_bl.text().strip():
            os.environ["CGAAF_BLENDER"] = self.x_bl.text().strip()
        if self.course and self.x_bl.text().strip():
            self.course.blender = ("exe", self.x_bl.text().strip())
        self.build()

    def do_export(self):
        df = self.course.dataset(anonymise=self.x_anon.isChecked())
        out = self.course.root / "results"
        out.mkdir(exist_ok=True)
        df.to_csv(out / "dataset.csv", index=False, encoding="utf-8-sig"); df.to_excel(out / "dataset.xlsx", index=False)
        self.last_out = out
        QMessageBox.information(self, self.T("done"), f"{len(df)} → {out / 'dataset.xlsx'}")
        open_path(out)


def run(lang="uz"):
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("CG-SDL-AAF")
    w = Main(lang)
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(run())
