"""Executed inside Blender: parametric benchmark scenes (6 compositions) with controlled errors.

args: OUT.blend COMPOSITION HUE_SHIFT MAIN_Z_SCALE DROP_SECONDARY(0|1) LIGHT_FACTOR EXTRA_OBJECT(0|1) [MESH_OUT.obj]
COMPOSITION 0 torus+cube · 1 Suzanne on a plinth · 2 still life (sphere, cone, textured box) ·
            3 chair · 4 mug · 5 desk lamp
The *main* object receives the hue / proportion errors; the *secondary* object can be removed;
LIGHT_FACTOR scales all light energy; EXTRA_OBJECT adds an unexpected sphere.
MESH_OUT (optional) exports the main object as OBJ (for the mesh benchmark).
"""
import colorsys
import math
import sys

import bpy

a = sys.argv[sys.argv.index("--") + 1:]
out, comp, hue, zs, drop, lf, extra = a[0], int(a[1]), float(a[2]), float(a[3]), int(a[4]), float(a[5]), int(a[6])
mesh_out = a[7] if len(a) > 7 else None
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
    bpy.ops.mesh.primitive_torus_add(major_radius=1.0, minor_radius=0.35, major_segments=64, minor_segments=24)
    main = put(active(), mat("m_main", 0.02 + hue), "Torus")
    bpy.ops.mesh.primitive_cube_add(size=0.7, location=(1.9, 0.2, 0.0), rotation=(0, 0, math.radians(25)))
    sec = put(active(), mat("m_sec", 0.62, 0.8, 0.85), "Cube")
elif comp == 1:
    bpy.ops.mesh.primitive_monkey_add(location=(0, 0, 0.55))
    mo = active()
    sub = mo.modifiers.new("sub", "SUBSURF"); sub.levels = 2; sub.render_levels = 2
    bpy.ops.object.shade_smooth()
    main = put(mo, mat("m_main", 0.08 + hue, 0.7, 0.9, metal=0.6, rough=0.3), "Suzanne")
    bpy.ops.mesh.primitive_cylinder_add(radius=0.7, depth=0.4, location=(0, 0, -0.45), vertices=48)
    sec = put(active(), mat("m_sec", 0.55, 0.15, 0.7), "Plinth")
elif comp == 2:
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.55, location=(-0.6, 0, 0.0), segments=48, ring_count=24)
    bpy.ops.object.shade_smooth()
    main = put(active(), mat("m_main", 0.33 + hue, 0.7, 0.75), "Sphere")
    bpy.ops.mesh.primitive_cone_add(radius1=0.45, depth=1.1, location=(0.6, 0.4, 0.0), vertices=48)
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
    bpy.ops.mesh.primitive_cylinder_add(radius=0.35, depth=0.5, location=(1.2, -0.4, -0.15), vertices=32)
    sec = put(active(), mat("m_sec", 0.58, 0.6, 0.7), "Stool")
elif comp == 4:  # mug
    bpy.ops.mesh.primitive_cylinder_add(radius=0.5, depth=1.1, location=(0, 0, 0), vertices=64)
    body = active()
    bpy.ops.mesh.primitive_torus_add(major_radius=0.3, minor_radius=0.07, location=(0.55, 0, 0.05), rotation=(math.radians(90), 0, 0))
    handle = active()
    main = put(join([body, handle], "Mug"), mat("m_main", 0.95 + hue, 0.6, 0.85, rough=0.25), "Mug")
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.25, location=(-0.9, 0.3, -0.3)); bpy.ops.object.shade_smooth()
    sec = put(active(), mat("m_sec", 0.15, 0.8, 0.9), "Ball")
elif comp == 5:  # desk lamp
    bpy.ops.mesh.primitive_cylinder_add(radius=0.45, depth=0.1, location=(0, 0, -0.6), vertices=48); base = active()
    bpy.ops.mesh.primitive_cylinder_add(radius=0.04, depth=1.2, location=(0, 0, 0.0), vertices=16); rod = active()
    bpy.ops.mesh.primitive_cone_add(radius1=0.45, radius2=0.12, depth=0.45, location=(0, 0, 0.75), vertices=48); shade = active()
    main = put(join([base, rod, shade], "Lamp"), mat("m_main", 0.6 + hue, 0.5, 0.8, metal=0.8, rough=0.2), "Lamp")
    bpy.ops.mesh.primitive_cube_add(size=0.5, location=(1.0, 0.3, -0.4))
    sec = put(active(), mat("m_sec", 0.1, 0.7, 0.85), "Book")

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
bpy.ops.wm.save_as_mainfile(filepath=out)
if mesh_out:
    bpy.ops.object.select_all(action="DESELECT"); main.select_set(True); bpy.context.view_layer.objects.active = main
    bpy.ops.wm.obj_export(filepath=mesh_out, export_selected_objects=True, apply_modifiers=True, export_materials=False,
                          export_uv=False, export_normals=False, export_triangulated_mesh=True)
print("AAF_OK", flush=True)
