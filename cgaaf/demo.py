"""Synthetic sample course for trying the software (NOT research data)."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .course import Course
from .mesh_metrics import torus

NAMES = ["Karimova Nodira", "Rashidov Sardor", "Toshpulatova Malika", "Ergashev Bekzod", "Yusupova Dilnoza", "Qodirov Jasur",
         "Nematova Zarina", "Hamidov Otabek", "Sobirova Madina", "Aliyev Sherzod", "Rahimova Shahnoza", "Murodov Akmal",
         "Saidova Kamola", "Turgunov Islom", "Xolmatova Gulnoza", "Ismoilov Diyor", "Jo'rayeva Sevara", "Bakirov Ulug'bek",
         "Normatova Feruza", "Abdurahmonov Eldor", "Usmonova Nilufar", "Karimov Javohir", "Haydarova Munisa", "Olimov Farrux"]
REFL = ["Yoritishni yaxshiladim, lekin kamera burchagi hali farq qiladi.", "Material rangini etalonga yaqinlashtirdim.",
        "Mesh dagi teshiklarni yopdim, keyingi safar modifikatorlarni tekshiraman.", "Proporsiyalarni o'lchab chiqdim.",
        "Soyalar juda qattiq chiqdi, yorug'lik manbaini kattalashtiraman.", ""]


def scene(dx=0, dy=0, hue=0.0, cube=True, blur=0.0, size=(640, 480), noise=0.0, seed=0):
    W, H = size
    y = np.linspace(0, 1, H)[:, None, None]
    bg = (np.array([223, 230, 238]) * (1 - y) + np.array([154, 167, 180]) * y) * np.ones((H, W, 3))
    img = Image.fromarray(bg.astype(np.uint8))
    d = ImageDraw.Draw(img)
    d.polygon([(0, 330), (W, 300), (W, H), (0, H)], fill=(123, 135, 148))
    sx, sy, r = 230 + dx, 250 + dy, 85
    d.ellipse([sx - 65, 322, sx + 115, 358], fill=(95, 104, 114))
    yy, xx = np.mgrid[0:H, 0:W]
    m = (xx - sx) ** 2 + (yy - sy) ** 2 < r * r
    nz = np.sqrt(np.clip(1 - ((xx - sx) ** 2 + (yy - sy) ** 2) / r ** 2, 0, 1))
    lam = np.clip((-(xx - sx) * 0.45 - (yy - sy) * 0.5) / r + nz * 0.75, 0.08, 1)
    base = np.array([216, 69, 47], float)
    if hue:
        c, s = np.cos(hue), np.sin(hue)
        Rm = np.array([[.299 + .701 * c + .168 * s, .587 - .587 * c + .330 * s, .114 - .114 * c - .497 * s],
                       [.299 - .299 * c - .328 * s, .587 + .413 * c + .035 * s, .114 - .114 * c + .292 * s],
                       [.299 - .3 * c + 1.25 * s, .587 - .588 * c - 1.05 * s, .114 + .886 * c - .203 * s]])
        base = np.clip(Rm @ base, 0, 255)
    a = np.asarray(img, float)
    shade = base[None, None] * lam[..., None] + 60 * nz[..., None] ** 30
    a[m] = np.clip(shade[m], 0, 255)
    img = Image.fromarray(a.astype(np.uint8))
    if cube:
        d = ImageDraw.Draw(img)
        bx, by, s = 420, 240, 100
        d.ellipse([bx, 322, bx + 140, 350], fill=(98, 107, 117))
        d.polygon([(bx, by), (bx + .6 * s, by + .2 * s), (bx + .6 * s, by + 1.2 * s), (bx, by + s)], fill=(59, 111, 216))
        d.polygon([(bx + .6 * s, by + .2 * s), (bx + 1.1 * s, by - .05 * s), (bx + 1.1 * s, by + .95 * s), (bx + .6 * s, by + 1.2 * s)], fill=(36, 74, 156))
        d.polygon([(bx, by), (bx + .5 * s, by - .25 * s), (bx + 1.1 * s, by - .05 * s), (bx + .6 * s, by + .2 * s)], fill=(126, 162, 239))
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    if noise:
        rng = np.random.default_rng(seed)
        a = np.asarray(img, float) + rng.normal(0, noise, (H, W, 3))
        img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    return img


def make_demo(root, n_students=24, seed=42, lang="uz", log=print):
    rng = np.random.default_rng(seed)
    root = Path(root)
    c = Course.create(root, "Kompyuter grafikasi — NAMUNA (sintetik)" if lang == "uz" else "Computer Graphics — SAMPLE (synthetic)", lang, synthetic=True)
    tmp = root / "_ref"
    tmp.mkdir(exist_ok=True)
    scene().save(tmp / "scene.png")
    torus().export(tmp / "torus.obj")
    today = dt.date.today()
    c.save_task({"id": "T1_render", "title_uz": "Render: yoritilgan sfera va kub sahnasi", "title_en": "Render: lit sphere and cube scene",
                 "desc_uz": "Etalondagi sahnani Blender'da takrorlang va PNG render yuboring.", "desc_en": "Recreate the reference scene in Blender and submit a PNG render.",
                 "type": "image", "trim": False, "max_attempts": 3, "deadline": str(today + dt.timedelta(days=14))}, ref_image=tmp / "scene.png")
    c.save_task({"id": "T2_torus", "title_uz": "Modellashtirish: tor (R=1, r=0.35), yopiq mesh", "title_en": "Modelling: torus (R=1, r=0.35), watertight mesh",
                 "desc_uz": "Torni modellashtiring va OBJ/STL eksport qiling.", "desc_en": "Model the torus and export OBJ/STL.",
                 "type": "mesh", "align": "pca_icp", "max_attempts": 3, "deadline": str(today + dt.timedelta(days=21))}, ref_mesh=tmp / "torus.obj")
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()
    up = root / "_upload"
    up.mkdir(exist_ok=True)
    for i in range(n_students):
        sid = "S%03d" % (i + 1)
        c.save_student(sid, NAMES[i % len(NAMES)], "AT-2%d" % (1 + i % 2), "60610200")
        q0 = float(rng.beta(2.5, 2))
        for tid in ["T1_render", "T2_torus"]:
            q = q0 + rng.normal(0, 0.08)
            bias = rng.normal(12, 10)
            for att in range(1, int(rng.integers(1, 4)) + 1):
                q = float(np.clip(q + (0.12 * rng.random() if att > 1 else 0), 0.02, 0.99))
                e = 1 - q
                if tid == "T1_render":
                    f = up / f"{sid}_{tid}_{att}.png"
                    scene(dx=int(rng.normal(0, 60 * e)), dy=int(rng.normal(0, 30 * e)), hue=float(rng.normal(0, 1.2 * e)),
                          cube=rng.random() > 0.6 * e ** 1.5, blur=float(abs(rng.normal(0, 3 * e))), noise=12 * e, seed=i).save(f)
                else:
                    f = up / f"{sid}_{tid}_{att}.obj"
                    torus(R=1 + rng.normal(0, 0.25 * e), r=float(np.clip(0.35 + rng.normal(0, 0.12 * e), 0.08, 0.6)), nu=int(rng.integers(24, 64)),
                          nv=int(rng.integers(12, 32)), rot=float(rng.random() * 3), scale=float(rng.uniform(0.5, 4)), noise=0.5 * e,
                          hole=bool(rng.random() < 0.7 * e), seed=i * 10 + att).export(f)
                pred = int(np.clip(100 * q + bias / att, 0, 100))
                plan = {"goal": "Etalonga yaqin natija olish" if att == 1 else "Oldingi xatolarni tuzatish", "minutes": int(rng.integers(20, 120)), "pred": pred}
                d = c.root / "submissions" / sid / tid / str(att)
                res = c.submit(sid, tid, [f], plan=plan, use_lpips=False)
                if rng.random() < 0.7:
                    c.save_reflection(d, REFL[int(rng.integers(0, len(REFL)))])
                true = 100 * q
                e1 = float(np.clip(true + rng.normal(0, 6), 0, 100))
                e2 = float(np.clip(true + rng.normal(2, 6), 0, 100)) if rng.random() < 0.6 else ""
                c.save_expert(sid, tid, att, round(e1), round(e2) if e2 != "" else "", "")
                f.unlink()
        log(f"{sid} ✓")
    up.rmdir()
    return c
