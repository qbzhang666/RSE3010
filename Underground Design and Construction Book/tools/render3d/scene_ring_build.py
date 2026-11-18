"""Ch 19: an exploded segmental ring (five segments and a key) in front of two built rings with staggered joints."""
import math

from common import *  # noqa: F401,F403

NAME = "ring_build"
CHAPTER = "c19-linings"
DATA_KEY = None
DEFAULTS = {
    "R": 3.4,              # segment centroid radius, tools/figs/f7.py SEG["radius"]
    "t": 0.3,              # segment thickness, SEG["thickness"]
    "b": 1.5,              # ring width, SEG["width"]
    "seg_deg": 67.5,       # five segments of 67.5 deg, SEG["angle"]
    "key_deg": 22.5,       # key 360 - 5 x 67.5
    "rots": [22.5, 337.5, 292.5],   # key centre from the crown, rings 1-3 of figures/ch19/ring-build.csv (f7.py rots)
    "joint": 0.04,         # open width of the joints, drawn so they read in print (schematic)
    "pull": 0.85,          # radial pull of the exploded segments (schematic)
    "pull_key": 1.35,      # the key pulled further: it goes in last
    "lift": 1.9,           # gap between the built rings and the exploded ring (schematic)
}
TARGET = (0.0, 3.4, -0.3)


def _sector(name, ri, ro, th0, th1, y0, y1, n=40):
    """An annular sector between angles th0 < th1 from the crown, clockwise looking back along -Y (x = -r sin)."""
    ths = [math.radians(th0 + (th1 - th0) * k / n) for k in range(n + 1)]
    outer = [(-ro * math.sin(t), ro * math.cos(t)) for t in ths]
    inner = [(-ri * math.sin(t), ri * math.cos(t)) for t in reversed(ths)]
    o = extrude_y(name, outer + inner, y0, y1)
    return o


def _ring(tag, d, rot, y0, mats, key_mat, pull=0.0, pull_key=0.0, joint_mat=None, next_ring=False):
    R, t, b = d["R"], d["t"], d["b"]
    ri, ro = R - t / 2, R + t / 2
    gap = math.degrees(d["joint"] / R) / 2
    gy = d["joint"] / 2
    kd, sd = d["key_deg"], d["seg_deg"]
    pieces = [(rot - kd / 2, rot + kd / 2, key_mat, pull_key)]
    for k in range(5):
        a0 = rot + kd / 2 + k * sd
        pieces.append((a0, a0 + sd, mats[k % len(mats)], pull))
    for i, (a0, a1, m, p) in enumerate(pieces):
        o = _sector(f"{tag}_{i}", ri, ro, a0 + gap, a1 - gap, y0 + gy, y0 + b - gy)
        mid = math.radians((a0 + a1) / 2)
        o.location = (-p * math.sin(mid), 0.0, p * math.cos(mid))
        assign(o, m)
        smooth(o, 30)
        if joint_mat is not None:                     # dark strip in each radial joint of a built ring
            j = _sector(f"{tag}_j{i}", ri + 0.01, ro - 0.01, a0 - gap, a0 + gap, y0, y0 + b)
            assign(j, joint_mat)
    if joint_mat is not None and next_ring:           # and in the circumferential joint ahead of it
        j = _sector(f"{tag}_cj", ri + 0.01, ro - 0.01, 0.0, 360.0, y0 + b - gy, y0 + b + gy, n=180)
        assign(j, joint_mat)


def build(d):
    b = d["b"]
    built = [material("seg_built", MAL_TINT, 0.7)]
    key_built = material("key_built", CAR_TINT, 0.7)
    mid = "#5E9583"                                   # between MAL and MAL_TINT
    shades = [material("seg_a", MAL, 0.6), material("seg_b", mid, 0.6), material("seg_c", MAL_DARK, 0.6),
              material("seg_d", mid, 0.6), material("seg_e", MAL, 0.6)]
    key = material("key", CAR, 0.6)
    jm = material("joint", STEEL, 0.8)
    rots = d["rots"]
    for i, rot in enumerate(rots[:-1]):
        _ring(f"ring{i + 1}", d, rot, i * b, built, key_built, joint_mat=jm, next_ring=i < len(rots) - 2)
    y_ex = (len(rots) - 1) * b + d["lift"]
    _ring("exploded", d, rots[-1], y_ex, shades, key, d["pull"], d["pull_key"])
    camera((17.0, 10.0, 10.0), TARGET, lens=42)
    lights((-0.35, -0.55, -0.75))
