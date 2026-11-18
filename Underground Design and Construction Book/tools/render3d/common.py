"""Shared helpers for the book's procedural Blender scenes (tools/render3d/scene_*.py).

House palette: malachite for linings, support and acceptable elements; carmine
for plastic or damaged ground, failing blocks and elements needing assessment;
greys for ground and context. No text is baked into a render.
"""

import math

import bmesh
import bpy
from mathutils import Vector

MAL, CAR = "#2F6B57", "#8C2B3E"
MAL_DARK, CAR_DARK = "#1F4A3C", "#6E1F30"
CAR_TINT, MAL_TINT = "#C48C99", "#A9C9BD"
MAL_PALE, CAR_PALE = "#E6EFEC", "#F3E6E9"
GROUND, GROUND_DARK, GROUND_LIGHT = "#A8A8A8", "#8A8A8A", "#C8C8C8"
CONCRETE, STEEL, WATER = "#E4E4E0", "#5A5A5A", "#9DB8C4"


def rgba(hexstr, a=1.0):
    h = hexstr.lstrip("#")
    srgb = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in srgb]
    return (*lin, a)


_MATS = {}


def material(name, colour, rough=0.75, alpha=1.0, metallic=0.0):
    """A Principled material; alpha < 1 gives a see-through solid (ground cut-aways, water)."""
    key = (name, colour, rough, alpha, metallic)
    if key in _MATS and _MATS[key].name in bpy.data.materials:
        return _MATS[key]
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    p = m.node_tree.nodes.get("Principled BSDF")
    p.inputs["Base Color"].default_value = rgba(colour)
    p.inputs["Roughness"].default_value = rough
    p.inputs["Metallic"].default_value = metallic
    if alpha < 1.0:
        p.inputs["Alpha"].default_value = alpha
        try:
            m.surface_render_method = "BLENDED"
        except AttributeError:
            pass
    _MATS[key] = m
    return m


def reset_materials():
    _MATS.clear()


def assign(obj, mat):
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    return obj


def link(obj):
    bpy.context.collection.objects.link(obj)
    return obj


def mesh(name, verts, faces):
    """An object from vertex and face lists (faces as index tuples)."""
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
    me.validate()
    me.update()
    return link(bpy.data.objects.new(name, me))


def box(name, x0, x1, y0, y1, z0, z1):
    bpy.ops.mesh.primitive_cube_add(size=1)
    o = bpy.context.active_object
    o.name = name
    o.scale = (x1 - x0, y1 - y0, z1 - z0)
    o.location = ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)
    bpy.ops.object.transform_apply(scale=True, location=False)
    return o


def cylinder_y(name, r, y0, y1, x=0.0, z=0.0, n=128):
    """A cylinder whose axis is parallel to Y, from y0 to y1."""
    bpy.ops.mesh.primitive_cylinder_add(vertices=n, radius=r, depth=y1 - y0)
    o = bpy.context.active_object
    o.name = name
    o.rotation_euler = (math.radians(90), 0, 0)
    o.location = (x, (y0 + y1) / 2, z)
    bpy.ops.object.transform_apply(rotation=True, location=False)
    return o


def cylinder_between(name, p0, p1, r, n=24):
    """A cylinder (bolt, pipe, cable, borehole) from point p0 to point p1."""
    p0, p1 = Vector(p0), Vector(p1)
    d = p1 - p0
    bpy.ops.mesh.primitive_cylinder_add(vertices=n, radius=r, depth=d.length)
    o = bpy.context.active_object
    o.name = name
    o.location = (p0 + p1) / 2
    o.rotation_euler = d.to_track_quat("Z", "Y").to_euler()
    return o


def sphere(name, r, loc, n=48):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=n, ring_count=n // 2, radius=r, location=loc)
    o = bpy.context.active_object
    o.name = name
    return o


def revolve_y(name, ys, rs, n=128):
    """A closed solid of revolution about the Y axis with radius rs[i] at ys[i]."""
    bm = bmesh.new()
    rings = []
    for y, r in zip(ys, rs):
        rings.append([bm.verts.new((r * math.cos(2 * math.pi * k / n), y, r * math.sin(2 * math.pi * k / n)))
                      for k in range(n)])
    for a, b in zip(rings[:-1], rings[1:]):
        for k in range(n):
            bm.faces.new((a[k], a[(k + 1) % n], b[(k + 1) % n], b[k]))
    bm.faces.new(list(reversed(rings[0])))
    bm.faces.new(rings[-1])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    return link(bpy.data.objects.new(name, me))


def extrude_y(name, outline_xz, y0, y1):
    """A prism: a closed polygon in the XZ plane (list of (x, z)) extruded from y0 to y1."""
    n = len(outline_xz)
    verts = [(x, y0, z) for x, z in outline_xz] + [(x, y1, z) for x, z in outline_xz]
    faces = [tuple(range(n))[::-1], tuple(range(n, 2 * n))]
    faces += [(k, (k + 1) % n, n + (k + 1) % n, n + k) for k in range(n)]
    o = mesh(name, verts, faces)
    recalc_normals(o)
    return o


def heightfield(name, xs, ys, zfun, base=None):
    """A surface z = zfun(x, y) on the grid xs by ys; with base, a closed solid down to z = base."""
    nx, ny = len(xs), len(ys)
    verts = [(x, y, zfun(x, y)) for y in ys for x in xs]
    faces = [(j * nx + i, j * nx + i + 1, (j + 1) * nx + i + 1, (j + 1) * nx + i)
             for j in range(ny - 1) for i in range(nx - 1)]
    o = mesh(name, verts, faces)
    if base is not None:
        top = max(v[2] for v in verts)
        sol = o.modifiers.new("solid", "SOLIDIFY")
        sol.thickness = top - base + 6.0
        sol.offset = -1.0
        apply_modifiers(o)
        flat_base(o, base)
    return o


def flat_base(obj, base):
    """Cut a solidified surface flat at z = base, so the block has a level bottom."""
    xs = [v.co.x for v in obj.data.vertices]
    ys = [v.co.y for v in obj.data.vertices]
    zs = [v.co.z for v in obj.data.vertices]
    cutter = box("base_cut", min(xs) - 5, max(xs) + 5, min(ys) - 5, max(ys) + 5, min(zs) - 5, base)
    boolean(obj, cutter)
    remove(cutter)
    return obj


def recalc_normals(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(obj.data)
    bm.free()


def apply_modifiers(obj):
    bpy.context.view_layer.objects.active = obj
    for m in list(obj.modifiers):
        bpy.ops.object.modifier_apply(modifier=m.name)


def boolean(target, cutter, op="DIFFERENCE"):
    mod = target.modifiers.new(name=f"bool_{cutter.name}", type="BOOLEAN")
    mod.operation = op
    mod.solver = "EXACT"
    mod.object = cutter
    bpy.context.view_layer.objects.active = target
    bpy.ops.object.modifier_apply(modifier=mod.name)
    return target


def remove(*objs):
    for o in objs:
        bpy.data.objects.remove(o, do_unlink=True)


def duplicate(obj, name):
    o = obj.copy()
    o.data = obj.data.copy()
    o.name = name
    bpy.context.collection.objects.link(o)
    return o


def smooth(obj, angle=35):
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    try:
        bpy.ops.object.shade_auto_smooth(angle=math.radians(angle))
    except Exception:
        bpy.ops.object.shade_smooth()
    obj.select_set(False)
    return obj


def camera(loc, target, lens=40, ortho=None):
    cam = bpy.data.cameras.new("cam")
    if ortho:
        cam.type = "ORTHO"
        cam.ortho_scale = ortho
    cam.lens = lens
    cam.clip_end = 5000
    o = bpy.data.objects.new("cam", cam)
    bpy.context.collection.objects.link(o)
    o.location = Vector(loc)
    o.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = o
    return o


def lights(sun_dir=(-0.5, 0.35, -0.8), strength=3.5, world=0.45):
    sun = bpy.data.lights.new("sun", "SUN")
    sun.energy = strength
    sun.angle = math.radians(12)
    o = bpy.data.objects.new("sun", sun)
    bpy.context.collection.objects.link(o)
    o.rotation_euler = Vector(sun_dir).to_track_quat("-Z", "Y").to_euler()
    w = bpy.data.worlds.new("world")
    bpy.context.scene.world = w
    w.use_nodes = True
    bg = w.node_tree.nodes.get("Background")
    bg.inputs["Color"].default_value = (1, 1, 1, 1)
    bg.inputs["Strength"].default_value = world
    return o
