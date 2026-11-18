"""Ch 4: the weathering profile as a block, with a fault-controlled trough, corestones and a mixed face."""
import csv
import math
import os
import random

from common import *  # noqa: F401,F403

NAME = "weathering_block"
CHAPTER = "c04-engineering-geology"
DATA_KEY = None
DEFAULTS = {
    # figures/ch04/geo-weathering-profile.csv, built by tools/figs/f1.py geo_weathering()
    "csv": "figures/ch04/geo-weathering-profile.csv",
    "every": 4,               # use every 4th of the 961 CSV points (0.5 m spacing)
    "bottom": 42.0,           # block base depth (m); grade II reaches 38.8 m in the trough
    "strike": 30.0,           # block extent along strike behind the section (m), illustrative
    "xf": 82.0,               # fault trace at tunnel level (f1.py)
    "fault": ((94.0, 0.0), (73.0, 36.0)),   # (x, depth) ends of the dashed fault in f1.py
    "boreholes": (20.0, 60.0, 100.0),       # CSV boreholes_x_m, drilled to 30 m (f1.py)
    "bh_depth": 30.0,
    "zt": 21.0, "r": 3.0,     # CSV tunnel_axis_m, tunnel_radius_m
    "lining_t": 0.35,         # illustrative
    "n_core": 60, "seed": 11,  # corestone draw as in f1.py
}
TARGET = (8.0, 58.0, -17.0)

# grade (VI residual soil ... I fresh rock): light to dark grey
GRADE_GREY = {6: "#E8E8E6", 5: "#D2D2D0", 4: "#BBBBB9", 3: "#A2A2A0", 2: "#8A8A88", 1: "#6E6E6C"}


def _book_path(rel):
    """Data file relative to the book root (tools/render3d/../..), or, on M3, to the
    folder that holds the driver and render3d/ (render3d/..)."""
    here = os.path.dirname(os.path.abspath(__file__))
    for root in (os.path.join(here, "..", ".."), os.path.join(here, "..")):
        p = os.path.normpath(os.path.join(root, rel))
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{rel} not found under {here}/../.. or {here}/..")


def _prism_x(name, outline_yz, x0, x1):
    """A closed polygon in the YZ plane (list of (y, z)) extruded along X from x0 to x1."""
    n = len(outline_yz)
    verts = [(x0, y, z) for y, z in outline_yz] + [(x1, y, z) for y, z in outline_yz]
    faces = [tuple(range(n)), tuple(range(2 * n - 1, n - 1, -1))]
    faces += [(k, n + k, n + (k + 1) % n, (k + 1) % n) for k in range(n)]
    o = mesh(name, verts, faces)
    recalc_normals(o)
    return o


def _interp(x, xs, vs):
    lo, hi = 0, len(xs) - 1
    if x <= xs[0]:
        return vs[0]
    if x >= xs[-1]:
        return vs[-1]
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    t = (x - xs[lo]) / (xs[hi] - xs[lo])
    return vs[lo] + t * (vs[hi] - vs[lo])


def build(d):
    rows = list(csv.DictReader(open(_book_path(d["csv"]), encoding="utf-8")))[::d["every"]]
    ys = [float(r["x_m"]) for r in rows]
    base = {g: [float(r[f"base_grade_{g}_m"]) for r in rows] for g in (6, 5, 4, 3, 2)}
    B = d["bottom"]
    base[1] = [B] * len(ys)
    X1 = d["strike"]
    zt, r = d["zt"], d["r"]
    L = ys[-1]

    # grade solids between successive base surfaces (scene z = -depth), section face at x = 0
    upper = [0.0] * len(ys)
    grades, spans = {}, {}
    for g in (6, 5, 4, 3, 2, 1):
        lower = [min(v, B) for v in base[g]]
        top = [min(u, B) for u in upper]
        outline = [(y, -t) for y, t in zip(ys, top)] + [(y, -b) for y, b in zip(ys[::-1], lower[::-1])]
        grades[g] = _prism_x(f"grade{g}", outline, 0.0, X1)
        spans[g] = any(t < zt + r and b > zt - r for t, b in zip(top, lower))
        upper = base[g]

    # fault: a thin damaged slab on the dashed trace of f1.py, striking along x
    (fy0, fd0), (fy1, fd1) = d["fault"]
    dy, dz = fy1 - fy0, -(fd1 - fd0)
    ln = math.hypot(dy, dz)
    ny, nz = -dz / ln * 0.35, dy / ln * 0.35
    fz0, fz1 = 1.0, -(B + 1.0)
    fyA = fy0 + (fz0 - 0.0) / dz * dy
    fyB = fy0 + (fz1 - 0.0) / dz * dy
    fault = _prism_x("fault", [(fyA - ny, fz0 - nz), (fyB - ny, fz1 - nz), (fyB + ny, fz1 + nz),
                               (fyA + ny, fz0 + nz)], -0.05, X1 + 0.05)

    # corestones of fresh rock in grades III to V, away from the fault, centred on the section
    rng = random.Random(d["seed"])
    stones = []
    for k in range(d["n_core"]):
        yc = rng.uniform(3, L - 3)
        lo, hi = _interp(yc, ys, base[5]), _interp(yc, ys, base[3])
        if hi - lo < 3.0 or abs(yc - d["xf"]) < 9:
            continue
        zc = rng.uniform(lo + 1.0, hi - 0.8)
        a, c = rng.uniform(1.4, 2.8) / 2, rng.uniform(0.8, 1.4) / 2
        s = sphere(f"core{k}", 1.0, (rng.uniform(-0.2, 0.6), yc, -zc), n=24)
        s.scale = (rng.uniform(0.8, 1.2) * a, a, c)
        s.rotation_euler = (math.radians(rng.uniform(-20, 20)), 0, 0)
        bpy.ops.object.select_all(action="DESELECT")
        s.select_set(True)
        bpy.context.view_layer.objects.active = s
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        stones.append(s)

    tunnel = cylinder_y("tunnel", r, -2, L + 2, z=-zt)
    halfcut = box("halfcut", -50, 0.0, -10, L + 10, -B - 10, 20)
    for g, o in grades.items():
        boolean(o, fault)
        if g in (5, 4, 3):
            for s in stones:
                boolean(o, s)
        if spans[g]:
            boolean(o, tunnel)
        assign(o, material(f"grade{g}", GRADE_GREY[g], 0.9))
    for o in [fault] + stones:
        boolean(o, tunnel)
        boolean(o, halfcut)
    assign(fault, material("fault", CAR_TINT, 0.85))
    smat = material("corestone", GRADE_GREY[2], 0.85)
    for s in stones:
        assign(s, smat)
        smooth(s)

    # lining as a half tube in the section; carmine where rockhead (base of IV) is above the crown
    mixed = [y for y, b in zip(ys, base[4]) if b > zt - r]
    ym0, ym1 = min(mixed), max(mixed)
    for nm, a0, a1, col in (("lining_a", 0.0, ym0, MAL), ("lining_mixed", ym0, ym1, CAR),
                            ("lining_b", ym1, L, MAL)):
        o = cylinder_y(nm, r, a0, a1, z=-zt)
        inner = cylinder_y(nm + "_in", r - d["lining_t"], a0 - 1, a1 + 1, z=-zt)
        boolean(o, inner)
        boolean(o, halfcut)
        remove(inner)
        assign(o, material(nm, col, 0.6))
        smooth(o)

    # boreholes on the section face
    bmat = material("borehole", "#3C3C3C", 0.5)
    for yb in d["boreholes"]:
        o = cylinder_between("bh", (0.0, yb, 1.2), (0.0, yb, -d["bh_depth"]), 0.45)
        assign(o, bmat)                             # half stands proud of the section face

    remove(tunnel, halfcut)
    print(f"  mixed-face reach {ym0:.1f}-{ym1:.1f} m ({ym1 - ym0:.1f} m)")
    camera((-118, -12, 42), TARGET, lens=40)
    lights((0.55, 0.25, -0.7), strength=2.4, world=0.36)
