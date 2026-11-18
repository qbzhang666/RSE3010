"""Ch 18: a conventional heading with sprayed concrete, steel sets and a staggered radial bolt pattern."""
import math

from common import *  # noqa: F401,F403
from mathutils import Vector

NAME = "supported_heading"
CHAPTER = "c18-support"
DATA_KEY = None
DEFAULTS = {
    "a": 4.75,             # tunnel radius, Chapter 16 / Table tab:support-element-scc
    "t_sc": 0.20,          # sprayed concrete t_c = 0.2 m, Table tab:support-element-scc
    "set_s": 1.0,          # steel sets at s = 1.0 m, Table tab:support-element-scc
    "bolt_L": 4.0,         # bolts L_b = 4 m, Table tab:support-element-scc
    "bolt_s": 1.5,         # bolts s_c = s_l = 1.5 m, Table tab:support-element-scc and Section 18 pattern example
    "span": 1.0,           # unsupported span at the face, one round (illustrative)
    "spray_lag": 0.6,      # sprayed edge behind the last set (illustrative)
    "set_depth": 0.22,     # set section depth, standing 20 mm proud of the shotcrete (illustrative, A_s = 50 cm2)
    "set_width": 0.16,     # set flange width (illustrative)
    "z_floor": -3.0,       # flat working floor below the springline (illustrative)
    "length": 10.5,        # length of heading shown behind the face
}
TARGET = (0.5, -4.5, 0.2)


def _annulus(name, r0, r1, y0, y1):
    outer = cylinder_y(name, r1, y0, y1, n=160)
    inner = cylinder_y(name + "_in", r0, y0 - 1, y1 + 1, n=160)
    boolean(outer, inner)
    remove(inner)
    return outer


def _rim_edges(obj, base, rim):
    """Faces whose normal is not radial (ends and cut faces) take the rim material."""
    obj.data.materials.clear()
    obj.data.materials.append(base)
    obj.data.materials.append(rim)
    for p in obj.data.polygons:
        c = p.center
        r = Vector((c.x, 0.0, c.z))
        if r.length < 1e-6:
            continue
        r.normalize()
        p.material_index = 1 if abs(p.normal.dot(r)) < 0.5 else 0


def build(d):
    a, t = d["a"], d["t_sc"]
    zf, Lt = d["z_floor"], d["length"]
    y_back, y_face, y_ahead = -Lt, 0.0, 3.0
    y_set0 = -d["span"]                               # first set, one round behind the face
    y_spray = y_set0 - d["spray_lag"]                 # sprayed edge
    B = a + d["bolt_L"] + 0.8

    below = box("below_floor", -B - 5, B + 5, y_back - 5, y_ahead + 5, -B - 5, zf)
    # cut-away: the upper right quarter, over the heading and the ground ahead of the face
    cut = box("cut", 0.0, B + 5, y_back - 5, y_ahead + 5, 0.0, B + 5)

    # ground and the excavated heading (flat floor)
    rock = box("rock", -B, B, y_back, y_ahead, zf - 2.5, B)
    hole = cylinder_y("hole", a, y_back - 1, y_face, n=160)
    boolean(hole, below)
    boolean(rock, hole)
    boolean(rock, cut)
    assign(rock, material("rock", GROUND, 0.9))

    # sprayed concrete shell with malachite rim edges
    shell = _annulus("shotcrete", a - t, a, y_back, y_spray)
    boolean(shell, below)
    boolean(shell, cut)
    _rim_edges(shell, material("shotcrete", CONCRETE, 0.85), material("rim", MAL, 0.6))

    # steel sets, standing just proud of the shotcrete
    steel = material("steel", STEEL, 0.45, metallic=0.3)
    ys = []
    y = y_set0
    while y > y_back + 0.3:
        ys.append(y)
        y -= d["set_s"]
    for k, y in enumerate(ys):
        s = _annulus(f"set_{k}", a - d["set_depth"], a, y - d["set_width"] / 2, y + d["set_width"] / 2)
        boolean(s, below)
        boolean(s, cut)
        assign(s, steel)

    # radial bolts, staggered rings at 1.5 m, 1.5 m around the arch, behind the sprayed edge
    bolt = material("bolt", MAL, 0.4)
    plate = material("plate", MAL_DARK, 0.5)
    nq = round(0.5 * math.pi * a / d["bolt_s"])       # whole spaces per quarter, so bolts lie on both cut planes
    dphi = 0.5 * math.pi / nq                         # 18 deg, 1.49 m of arc
    yb = y_spray - 0.6
    ring = 0
    while yb > y_back + 0.4:
        off = 0.5 * dphi if ring % 2 else 0.0
        k = -10
        while k <= 10:
            phi = math.pi / 2 + off + k * dphi
            k += 1
            x, z = math.cos(phi), math.sin(phi)
            if z * a < zf + 1.0:
                continue
            if x > 1e-3 and z > 1e-3:                 # in the removed quarter
                continue
            n = Vector((x, 0.0, z))
            p0 = Vector((0.0, yb, 0.0)) + n * (a - t - 0.04)
            p1 = p0 + n * d["bolt_L"]
            assign(cylinder_between("bolt", p0, p1, 0.06), bolt)
            pl = box("plate", -0.13, 0.13, -0.13, 0.13, -0.015, 0.015)
            pl.rotation_euler = n.to_track_quat("Z", "Y").to_euler()
            pl.location = p0 + n * 0.02
            assign(pl, plate)
        ring += 1
        yb -= d["bolt_s"]
    remove(below, cut, hole)
    camera((31.0, -22.5, 16.5), TARGET, lens=36)
    lights((-0.5, 0.45, -0.75))
