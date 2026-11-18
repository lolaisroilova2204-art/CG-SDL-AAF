"""Executed inside Blender: parametric benchmark scenes (12 compositions) with controlled errors.

args: OUT.blend COMPOSITION HUE_SHIFT MAIN_Z_SCALE DROP_SECONDARY(0|1) LIGHT_FACTOR EXTRA_OBJECT(0|1)
      [MESH_OUT.obj|-] [TESSELLATION 0|1|2] [SETTINGS 0|1|2|3]
COMPOSITION 0 torus+cube · 1 Suzanne on a plinth · 2 still life (sphere, cone, textured box) ·
            3 chair · 4 mug · 5 desk lamp      (development set, used while the defaults were fixed)
            6 table + vase · 7 bottle + glass · 8 snowman · 9 rocket · 10 robot · 11 kettle
            (held-out set, created after the defaults of v1.3 had been fixed)
The *main* object receives the hue / proportion errors; the *secondary* object can be removed;
LIGHT_FACTOR scales all light energy; EXTRA_OBJECT adds an unexpected sphere.
MESH_OUT (optional) exports the main object as OBJ (for the mesh benchmark).
TESSELLATION builds the same objects with other segment counts / subdivision levels
(1 = half, 2 = double): an independently tessellated but correct model.
SETTINGS stores student-like render settings that canonical re-rendering must neutralise:
1 = EEVEE + AgX + 1920x1080, 2 = own camera + Filmic + 128 samples, 3 = transparent film + 50 % + denoising.
"""
import colorsys
import math
import sys

import bpy

a = sys.argv[sys.argv.index("--") + 1:]
out, comp, hue, zs, drop, lf, extra = a[0], int(a[1]), float(a[2]), float(a[3]), int(a[4]), float(a[5]), int(a[6])
mesh_out = a[7] if len(a) > 7 and a[7] != "-" else None
tess = int(a[8]) if len(a) > 8 else 0
settings = int(a[9]) if len(a) > 9 else 0
TF = {0: 1.0, 1: 0.5, 2: 2.0}[tess]


def seg(n):
    return max(6, int(round(n * TF)))


SUB = {0: 2, 1: 1, 2: 3}[tess]
bpy.ops.wm.read_factory_settings(use_empty=True)
S = bpy.context.scene


def mat(name, h, s=0.75, v=0.85, metal=0.0, rough=0.45, noise=False):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, s, v)
    bsdf.inputs["Base Color"].default_value = (r, g, b, 1)
    bsdf.inputs["Metallic"].default_value = metal
    bsdf.inputs["Roughness"].default_value = rough
    if noise:  # procedural texture
        tex = nt.nodes.new("ShaderNodeTexNoise")
        tex.inputs["Scale"].default_value = 6.0
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].color = (r * 0.5, g * 0.5, b * 0.5, 1)
        ramp.color_ramp.elements[1].color = (r, g, b, 1)
        nt.links.new(tex.outputs["Fac"], ramp.inputs["Fac"])
        nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    return m


def put(obj, m, name):
    obj.name = name
    if obj.type == "MESH":
        obj.data.materials.append(m)
    return obj


def active():
    return bpy.context.object


def applied(objs):
    """Bake rotation and scale into the vertices (location kept) before joining."""
    for o in objs:
        bpy.ops.object.select_all(action="DESELECT")
        o.select_set(True)
        bpy.context.view_layer.objects.active = o
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    return objs


def join(objs, name):
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.join()
    o = bpy.context.object
    o.name = name
    return o


main = sec = None
if comp == 0:
    bpy.ops.mesh.primitive_torus_add(major_radius=1.0, minor_radius=0.35, major_segments=seg(64), minor_segments=seg(24))
    main = put(active(), mat("m_main", 0.02 + hue), "Torus")
    bpy.ops.mesh.primitive_cube_add(size=0.7, location=(1.9, 0.2, 0.0), rotation=(0, 0, math.radians(25)))
    sec = put(active(), mat("m_sec", 0.62, 0.8, 0.85), "Cube")
elif comp == 1:
    bpy.ops.mesh.primitive_monkey_add(location=(0, 0, 0.55))
    mo = active()
    sub = mo.modifiers.new("sub", "SUBSURF"); sub.levels = SUB; sub.render_levels = SUB
    bpy.ops.object.shade_smooth()
    main = put(mo, mat("m_main", 0.08 + hue, 0.7, 0.9, metal=0.6, rough=0.3), "Suzanne")
    bpy.ops.mesh.primitive_cylinder_add(radius=0.7, depth=0.4, location=(0, 0, -0.45), vertices=seg(48))
    sec = put(active(), mat("m_sec", 0.55, 0.15, 0.7), "Plinth")
elif comp == 2:
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.55, location=(-0.6, 0, 0.0), segments=seg(48), ring_count=seg(24))
    bpy.ops.object.shade_smooth()
    main = put(active(), mat("m_main", 0.33 + hue, 0.7, 0.75), "Sphere")
    bpy.ops.mesh.primitive_cone_add(radius1=0.45, depth=1.1, location=(0.6, 0.4, 0.0), vertices=seg(48))
    sec = put(active(), mat("m_sec", 0.12, 0.8, 0.9), "Cone")
    bpy.ops.mesh.primitive_cube_add(size=0.6, location=(0.5, -0.7, -0.25))
    put(active(), mat("m_box", 0.07, 0.6, 0.7, noise=True), "Box")
elif comp == 3:  # chair
    parts = []
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0.45)); s = active(); s.scale = (0.9, 0.9, 0.08); parts.append(s)
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0.41, 0.95)); b = active(); b.scale = (0.9, 0.08, 0.9); parts.append(b)
    for x, y in [(0.38, 0.38), (-0.38, 0.38), (0.38, -0.38), (-0.38, -0.38)]:
        bpy.ops.mesh.primitive_cube_add(size=1, location=(x, y, 0.0)); l = active(); l.scale = (0.08, 0.08, 0.85); parts.append(l)
    for p in parts:
        bpy.context.view_layer.objects.active = p
        bpy.ops.object.transform_apply(scale=True)
    main = put(join(parts, "Chair"), mat("m_main", 0.07 + hue, 0.55, 0.6, rough=0.6), "Chair")
    bpy.ops.mesh.primitive_cylinder_add(radius=0.35, depth=0.5, location=(1.2, -0.4, -0.15), vertices=seg(32))
    sec = put(active(), mat("m_sec", 0.58, 0.6, 0.7), "Stool")
elif comp == 4:  # mug
    bpy.ops.mesh.primitive_cylinder_add(radius=0.5, depth=1.1, location=(0, 0, 0), vertices=seg(64))
    body = active()
    bpy.ops.mesh.primitive_torus_add(major_radius=0.3, minor_radius=0.07, location=(0.55, 0, 0.05), rotation=(math.radians(90), 0, 0), major_segments=seg(48), minor_segments=seg(12))
    handle = active()
    main = put(join([body, handle], "Mug"), mat("m_main", 0.95 + hue, 0.6, 0.85, rough=0.25), "Mug")
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.25, location=(-0.9, 0.3, -0.3), segments=seg(32), ring_count=seg(16)); bpy.ops.object.shade_smooth()
    sec = put(active(), mat("m_sec", 0.15, 0.8, 0.9), "Ball")
elif comp == 5:  # desk lamp
    bpy.ops.mesh.primitive_cylinder_add(radius=0.45, depth=0.1, location=(0, 0, -0.6), vertices=seg(48)); base = active()
    bpy.ops.mesh.primitive_cylinder_add(radius=0.04, depth=1.2, location=(0, 0, 0.0), vertices=seg(16)); rod = active()
    bpy.ops.mesh.primitive_cone_add(radius1=0.45, radius2=0.12, depth=0.45, location=(0, 0, 0.75), vertices=seg(48)); shade = active()
    main = put(join([base, rod, shade], "Lamp"), mat("m_main", 0.6 + hue, 0.5, 0.8, metal=0.8, rough=0.2), "Lamp")
    bpy.ops.mesh.primitive_cube_add(size=0.5, location=(1.0, 0.3, -0.4))
    sec = put(active(), mat("m_sec", 0.1, 0.7, 0.85), "Book")

elif comp == 6:  # table with a vase (held-out)
    parts = []
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0.35)); t = active(); t.scale = (1.4, 0.8, 0.06); parts.append(t)
    for x, y in [(0.62, 0.32), (-0.62, 0.32), (0.62, -0.32), (-0.62, -0.32)]:
        bpy.ops.mesh.primitive_cylinder_add(radius=0.045, depth=0.95, location=(x, y, -0.13), vertices=seg(24)); parts.append(active())
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0.35, -0.3, 0.22)); dr = active(); dr.scale = (0.5, 0.06, 0.16); parts.append(dr)
    main = put(join(applied(parts), "Table"), mat("m_main", 0.09 + hue, 0.5, 0.55, rough=0.55), "Table")
    bpy.ops.mesh.primitive_cone_add(radius1=0.16, radius2=0.07, depth=0.45, location=(-0.3, 0.05, 0.6), vertices=seg(32))
    sec = put(active(), mat("m_sec", 0.55, 0.6, 0.8, rough=0.2), "Vase")
elif comp == 7:  # bottle and glass (held-out)
    bpy.ops.mesh.primitive_cylinder_add(radius=0.32, depth=0.9, location=(0, 0, -0.15), vertices=seg(48)); body = active()
    bpy.ops.mesh.primitive_cone_add(radius1=0.32, radius2=0.12, depth=0.3, location=(0, 0, 0.45), vertices=seg(48)); sh = active()
    bpy.ops.mesh.primitive_cylinder_add(radius=0.11, depth=0.35, location=(0, 0, 0.77), vertices=seg(32)); neck = active()
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, -0.3, -0.1)); lab = active(); lab.scale = (0.3, 0.05, 0.3)
    main = put(join(applied([body, sh, neck, lab]), "Bottle"), mat("m_main", 0.36 + hue, 0.7, 0.55, rough=0.15), "Bottle")
    bpy.ops.mesh.primitive_cylinder_add(radius=0.18, depth=0.4, location=(0.75, -0.25, -0.4), vertices=seg(32))
    sec = put(active(), mat("m_sec", 0.6, 0.2, 0.9, rough=0.1), "Glass")
elif comp == 8:  # snowman (held-out, asymmetric: nose and one arm)
    parts = []
    for r_, z_ in [(0.45, -0.2), (0.32, 0.45), (0.22, 0.9)]:
        bpy.ops.mesh.primitive_uv_sphere_add(radius=r_, location=(0, 0, z_), segments=seg(32), ring_count=seg(16)); parts.append(active())
    bpy.ops.mesh.primitive_cone_add(radius1=0.06, depth=0.3, location=(0, -0.32, 0.9), rotation=(math.radians(90), 0, 0), vertices=seg(16)); parts.append(active())
    bpy.ops.mesh.primitive_cylinder_add(radius=0.025, depth=0.6, location=(0.45, 0, 0.5), rotation=(0, math.radians(60), 0), vertices=seg(12)); parts.append(active())
    main = put(join(applied(parts), "Snowman"), mat("m_main", 0.58 + hue, 0.12, 0.95, rough=0.7), "Snowman")
    bpy.ops.mesh.primitive_cylinder_add(radius=0.17, depth=0.2, location=(0, 0, 1.18), vertices=seg(32))
    sec = put(active(), mat("m_sec", 0.0, 0.8, 0.5), "Hat")
elif comp == 9:  # rocket with three fins and a window (held-out)
    parts = []
    bpy.ops.mesh.primitive_cylinder_add(radius=0.22, depth=1.2, location=(0, 0, 0), vertices=seg(32)); parts.append(active())
    bpy.ops.mesh.primitive_cone_add(radius1=0.22, depth=0.45, location=(0, 0, 0.825), vertices=seg(32)); parts.append(active())
    for k in range(3):
        ang = 2 * math.pi * k / 3
        bpy.ops.mesh.primitive_cube_add(size=1, location=(0.3 * math.cos(ang), 0.3 * math.sin(ang), -0.42), rotation=(0, 0, ang))
        f = active(); f.scale = (0.22, 0.03, 0.32); parts.append(f)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.07, location=(0, -0.2, 0.25), segments=seg(16), ring_count=seg(8)); parts.append(active())
    main = put(join(applied(parts), "Rocket"), mat("m_main", 0.0 + hue, 0.75, 0.85, metal=0.3, rough=0.35), "Rocket")
    bpy.ops.mesh.primitive_cube_add(size=0.4, location=(0.9, 0.4, -0.45))
    sec = put(active(), mat("m_sec", 0.12, 0.6, 0.6, noise=True), "Crate")
elif comp == 10:  # robot figure (held-out, asymmetric arms and antenna)
    parts = []
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0.1)); t = active(); t.scale = (0.5, 0.3, 0.6); parts.append(t)
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0.6)); h = active(); h.scale = (0.3, 0.25, 0.25); parts.append(h)
    for x in (-0.13, 0.13):
        bpy.ops.mesh.primitive_cylinder_add(radius=0.08, depth=0.5, location=(x, 0, -0.45), vertices=seg(24)); parts.append(active())
    bpy.ops.mesh.primitive_cylinder_add(radius=0.06, depth=0.5, location=(-0.33, 0, 0.1), vertices=seg(24)); parts.append(active())
    bpy.ops.mesh.primitive_cylinder_add(radius=0.06, depth=0.5, location=(0.42, 0, 0.32), rotation=(0, math.radians(-40), 0), vertices=seg(24)); parts.append(active())
    bpy.ops.mesh.primitive_cylinder_add(radius=0.015, depth=0.2, location=(0.08, 0, 0.82), vertices=seg(12)); parts.append(active())
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.04, location=(0.08, 0, 0.94), segments=seg(16), ring_count=seg(8)); parts.append(active())
    main = put(join(applied(parts), "Robot"), mat("m_main", 0.55 + hue, 0.25, 0.75, metal=0.7, rough=0.3), "Robot")
    bpy.ops.mesh.primitive_cube_add(size=0.35, location=(-0.85, 0.3, -0.5))
    sec = put(active(), mat("m_sec", 0.08, 0.7, 0.75), "Box")
elif comp == 11:  # kettle (held-out, asymmetric spout and handle)
    parts = []
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.5, location=(0, 0, -0.1), segments=seg(48), ring_count=seg(24)); b_ = active(); b_.scale = (1, 1, 0.75); parts.append(b_)
    bpy.ops.mesh.primitive_cylinder_add(radius=0.06, depth=0.5, location=(0.55, 0, 0.05), rotation=(0, math.radians(55), 0), vertices=seg(24)); parts.append(active())
    bpy.ops.mesh.primitive_torus_add(major_radius=0.25, minor_radius=0.05, location=(-0.5, 0, 0.05), rotation=(math.radians(90), 0, 0), major_segments=seg(32), minor_segments=seg(12)); parts.append(active())
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.08, location=(0, 0, 0.32), segments=seg(16), ring_count=seg(8)); parts.append(active())
    main = put(join(applied(parts), "Kettle"), mat("m_main", 0.02 + hue, 0.65, 0.7, metal=0.5, rough=0.3), "Kettle")
    bpy.ops.mesh.primitive_cylinder_add(radius=0.17, depth=0.3, location=(0.9, -0.5, -0.45), vertices=seg(32))
    sec = put(active(), mat("m_sec", 0.6, 0.5, 0.85), "Cup")

if zs != 1.0:
    main.scale.z *= zs
    bpy.context.view_layer.objects.active = main
    bpy.ops.object.select_all(action="DESELECT"); main.select_set(True)
    bpy.ops.object.transform_apply(scale=True)
if drop and sec is not None:
    bpy.data.objects.remove(sec, do_unlink=True)
if extra:
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.35, location=(-1.5, -0.6, 0.0)); bpy.ops.object.shade_smooth()
    put(active(), mat("m_extra", 0.8, 0.7, 0.9), "Extra")

bpy.ops.mesh.primitive_plane_add(size=6, location=(0, 0, -0.66))
put(active(), mat("m_floor", 0.6, 0.05, 0.55), "Floor")
for name, en, rot in [("key", 3.5, (45, 0, 35)), ("fill", 1.0, (60, 0, -120))]:
    ld = bpy.data.lights.new(name, "SUN"); ld.energy = en * lf
    lo = bpy.data.objects.new(name, ld); lo.rotation_euler = tuple(math.radians(v) for v in rot)
    S.collection.objects.link(lo)
w = bpy.data.worlds.new("w"); S.world = w; w.use_nodes = True
bg = next(n for n in w.node_tree.nodes if n.type == "BACKGROUND")
bg.inputs["Color"].default_value = (0.78, 0.8, 0.84, 1); bg.inputs["Strength"].default_value = 0.6 * max(lf, 0.2) ** 0.5
# ---- student-like render settings (canonical re-rendering must neutralise them)
R = S.render
if settings == 1:
    for eng in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
        try:
            R.engine = eng
            break
        except TypeError:
            continue
    try:
        S.view_settings.view_transform = "AgX"
    except TypeError:
        pass
    R.resolution_x, R.resolution_y = 1920, 1080
elif settings == 2:
    cd = bpy.data.cameras.new("StudentCam"); cd.lens = 35
    cam = bpy.data.objects.new("StudentCam", cd); S.collection.objects.link(cam)
    cam.location = (3.2, -2.4, 1.1); cam.rotation_euler = (math.radians(80), 0, math.radians(52))
    S.camera = cam
    for vt in ("Filmic", "Filmic Base Contrast", "AgX"):
        try:
            S.view_settings.view_transform = vt
            break
        except TypeError:
            continue
    R.engine = "CYCLES"; S.cycles.samples = 128
elif settings == 3:
    R.engine = "CYCLES"; R.film_transparent = True; R.resolution_percentage = 50
    S.cycles.use_denoising = True; S.cycles.samples = 512
bpy.ops.wm.save_as_mainfile(filepath=out)
if mesh_out:
    bpy.ops.object.select_all(action="DESELECT"); main.select_set(True); bpy.context.view_layer.objects.active = main
    bpy.ops.wm.obj_export(filepath=mesh_out, export_selected_objects=True, apply_modifiers=True, export_materials=False,
                          export_uv=False, export_normals=False, export_triangulated_mesh=True)
print("AAF_OK", flush=True)
