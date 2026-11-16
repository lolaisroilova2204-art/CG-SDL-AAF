"""Executed INSIDE Blender (or a Python with the `bpy` module).

usage: blender -b --factory-startup --python blender_script.py -- \
        INPUT OUT_PNG OUT_OBJ MODE RES ENGINE SAMPLES [AZ EL DIST] [FILTER] [FRAMES]
INPUT  : .blend (opened) or .fbx/.obj/.stl/.ply/.glb/.gltf (imported into an empty scene)
OUT_PNG: render path or "-"   OUT_OBJ: mesh export path or "-"
MODE   : canonical | scene    (canonical = standard camera/colour management, same for every file)
FILTER : auto | all | names:Torus*,Cube* | collection:Model   objects that are assessed (export + camera framing);
         auto drops plane-like helper objects (floors, backdrops). The render itself shows the whole scene.
HELPERS: hide | show  — in canonical mode helper objects are hidden from the render (default) so that
         only the assessed objects are compared; 'show' keeps the full scene composition.
FRAMES : 1 = current frame; K>1 = K frames evenly spaced over [frame_start, frame_end] -> OUT_PNG stem + _fNN.png
"""
import fnmatch
import math
import os
import sys

import bpy
import mathutils

argv = sys.argv[sys.argv.index("--") + 1:]
src, out_png, out_obj, mode, res, engine, samples = argv[:7]
res, samples = int(res), int(samples)
az_off, el_off, dist_fac = [float(v) for v in (argv[7:10] if len(argv) >= 10 else ["0", "0", "1"])]  # viewpoint jitter (benchmark)
obj_filter = argv[10] if len(argv) > 10 else "auto"
n_frames = int(argv[11]) if len(argv) > 11 else 1
helpers = argv[12] if len(argv) > 12 else "hide"  # hide | show : render helper objects (floors) or not
ext = os.path.splitext(src)[1].lower()

if ext == ".blend":
    bpy.ops.wm.open_mainfile(filepath=src)
else:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    imp = {".fbx": lambda p: bpy.ops.import_scene.fbx(filepath=p),
           ".obj": lambda p: bpy.ops.wm.obj_import(filepath=p),
           ".stl": lambda p: bpy.ops.wm.stl_import(filepath=p),
           ".ply": lambda p: bpy.ops.wm.ply_import(filepath=p),
           ".glb": lambda p: bpy.ops.import_scene.gltf(filepath=p),
           ".gltf": lambda p: bpy.ops.import_scene.gltf(filepath=p)}[ext]
    imp(src)

scene = bpy.context.scene
print("AAF_BLENDER_VERSION", bpy.app.version_string, flush=True)
if n_frames > 1 or out_obj != "-":
    scene.frame_set(scene.frame_start)
renderable = [o for o in scene.objects if o.type in {"MESH", "CURVE", "SURFACE", "META", "FONT"} and not o.hide_render]
HELPER_NAMES = ["plane*", "floor*", "ground*", "background*", "backdrop*", "pol*", "tekislik*", "zamin*", "фон*", "пол*"]


def plane_like(o):
    d = sorted(o.dimensions)
    if d[2] <= 0:
        return True
    n_poly = len(o.data.polygons) if o.type == "MESH" else 999
    flat = d[0] < 0.01 * d[2]
    named = any(fnmatch.fnmatch(o.name.lower(), p) for p in HELPER_NAMES)
    return (flat and n_poly <= 64) or (named and d[0] < 0.05 * d[2])


if obj_filter == "all":
    meshes = renderable
elif obj_filter.startswith("names:"):
    pats = [p.strip().lower() for p in obj_filter[6:].split(",") if p.strip()]
    meshes = [o for o in renderable if any(fnmatch.fnmatch(o.name.lower(), p) for p in pats)]
elif obj_filter.startswith("collection:"):
    col = bpy.data.collections.get(obj_filter[11:])
    meshes = [o for o in renderable if col and o.name in col.all_objects]
else:
    meshes = [o for o in renderable if not plane_like(o)] or renderable
print("AAF_OBJECTS", ",".join(o.name for o in meshes), flush=True)
if helpers == "hide" and mode == "canonical":
    for o in renderable:
        if o not in meshes:
            o.hide_render = True
if not meshes:
    print("AAF_ERROR: no renderable geometry", flush=True)
    sys.exit(3)

# ------------------------------------------------------------ mesh export
if out_obj != "-":
    for o in scene.objects:
        o.select_set(o in meshes)
    bpy.context.view_layer.objects.active = meshes[0]
    try:
        bpy.ops.wm.obj_export(filepath=out_obj, export_selected_objects=True, apply_modifiers=True,
                              export_materials=False, export_uv=False, export_normals=False,
                              export_triangulated_mesh=True)
    except AttributeError:  # Blender < 3.2
        bpy.ops.export_scene.obj(filepath=out_obj, use_selection=True, use_materials=False, use_triangles=True)

# ------------------------------------------------------------ render
if out_png != "-":
    dg = bpy.context.evaluated_depsgraph_get()
    pts = []
    for o in meshes:
        pts += [o.matrix_world @ mathutils.Vector(c) for c in o.bound_box]
    lo = mathutils.Vector([min(p[i] for p in pts) for i in range(3)])
    hi = mathutils.Vector([max(p[i] for p in pts) for i in range(3)])
    centre, radius = (lo + hi) / 2, max((hi - lo).length / 2, 1e-3)

    if mode == "canonical" or scene.camera is None:
        cam_data = bpy.data.cameras.new("AAF_Camera")
        cam_data.lens, cam_data.sensor_width = 50, 36
        cam = bpy.data.objects.new("AAF_Camera", cam_data)
        scene.collection.objects.link(cam)
        az, el = math.radians(45 + az_off), math.radians(25 + el_off)
        fov = 2 * math.atan(cam_data.sensor_width / (2 * cam_data.lens))
        dist = radius / math.sin(fov / 2) * 1.05 * dist_fac
        d = mathutils.Vector((math.cos(el) * math.cos(az), -math.cos(el) * math.sin(az), math.sin(el)))
        cam.location = centre + d * dist
        cam.rotation_euler = (centre - cam.location).to_track_quat("-Z", "Y").to_euler()
        cam_data.clip_end = dist * 10
        scene.camera = cam
    if not any(o.type == "LIGHT" and not o.hide_render for o in scene.objects):
        ld = bpy.data.lights.new("AAF_Sun", "SUN")
        ld.energy = 3.0
        lo_ = bpy.data.objects.new("AAF_Sun", ld)
        lo_.rotation_euler = (math.radians(50), 0, math.radians(30))
        scene.collection.objects.link(lo_)
    if scene.world is None:
        scene.world = bpy.data.worlds.new("AAF_World")

    for name in ([engine] + (["BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"] if engine.startswith("BLENDER_EEVEE") else [])):
        try:
            scene.render.engine = name
            break
        except TypeError:
            continue
    if scene.render.engine == "CYCLES":
        scene.cycles.device = "CPU"
        scene.cycles.samples = samples
        scene.cycles.use_denoising = False
        scene.cycles.seed = 0
    if mode == "canonical":
        try:
            scene.view_settings.view_transform = "Standard"
            scene.view_settings.look = "None"
        except TypeError:
            pass
    r = scene.render
    r.resolution_x = r.resolution_y = res
    r.resolution_percentage = 100
    r.film_transparent = False
    r.image_settings.file_format = "PNG"
    if n_frames <= 1:
        r.filepath = out_png
        bpy.ops.render.render(write_still=True)
    else:
        f0, f1 = scene.frame_start, scene.frame_end
        stem = os.path.splitext(out_png)[0]
        for i in range(n_frames):
            scene.frame_set(int(round(f0 + (f1 - f0) * i / (n_frames - 1))))
            r.filepath = f"{stem}_f{i:02d}.png"
            bpy.ops.render.render(write_still=True)
print("AAF_OK", flush=True)
