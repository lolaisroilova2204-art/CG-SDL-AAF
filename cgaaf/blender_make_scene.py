"""Executed inside Blender: build a parametric benchmark scene and save it as .blend.
args: OUT.blend HUE_SHIFT(0..1) MINOR_RADIUS CUBE(0|1)"""
import colorsys
import math
import sys

import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
out, hue, minor, cube = argv[0], float(argv[1]), float(argv[2]), int(argv[3])
bpy.ops.wm.read_factory_settings(use_empty=True)


def mat(name, rgb):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (*rgb, 1)
    bsdf.inputs["Roughness"].default_value = 0.45
    return m


bpy.ops.mesh.primitive_plane_add(size=4.2, location=(0.5, 0, -0.36))
bpy.context.object.data.materials.append(mat("floor", (0.45, 0.48, 0.52)))
bpy.ops.mesh.primitive_torus_add(major_radius=1.0, minor_radius=minor, major_segments=64, minor_segments=24)
r, g, b = colorsys.hsv_to_rgb((0.02 + hue) % 1.0, 0.8, 0.85)
bpy.context.object.data.materials.append(mat("torus", (r, g, b)))
if cube:
    bpy.ops.mesh.primitive_cube_add(size=0.7, location=(1.9, 0.2, 0.0), rotation=(0, 0, math.radians(25)))
    bpy.context.object.data.materials.append(mat("cube", (0.15, 0.3, 0.85)))
ld = bpy.data.lights.new("sun", "SUN"); ld.energy = 3.5
lo = bpy.data.objects.new("sun", ld); lo.rotation_euler = (math.radians(45), 0, math.radians(35))
bpy.context.scene.collection.objects.link(lo)
w = bpy.data.worlds.new("w"); bpy.context.scene.world = w
w.use_nodes = True
bg = next(n for n in w.node_tree.nodes if n.type == "BACKGROUND")
bg.inputs["Color"].default_value = (0.75, 0.78, 0.82, 1); bg.inputs["Strength"].default_value = 0.8
bpy.ops.wm.save_as_mainfile(filepath=out)
print("AAF_OK", flush=True)
