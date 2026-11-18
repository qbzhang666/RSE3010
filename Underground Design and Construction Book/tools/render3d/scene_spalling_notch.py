"""Ch 14: V-shaped spalling notches along a circular tunnel in brittle rock (quarter cut-away)."""
import math
import random

from common import *  # noqa: F401,F403

NAME = "spalling_notch"
CHAPTER = "c14-high-stress-ground"
DATA_KEY = None
DEFAULTS = {
    "a": 2.5,              # illustrative radius; the notch is scaled on a (r/a is dimensionless)
    "r_over_a": 1.443,     # figures/ch14/spalling-notch.csv, r_over_a (tools/figs/f5.py: 0.5 + 1.25*169/224)
    "half_deg": 39.5,      # figures/ch14/spalling-notch.csv, notch_half_angle_deg (sigma_theta > 0.4 UCS)
    "length_D": 3.0,       # tunnel shown over three diameters
    "damage": 0.12,        # thickness of the carmine damaged skin on the notch flanks (schematic)
    "flakes": 9,           # loose slabs lying in the invert notch (schematic)
    "cut_deg": 42.0,       # half-angle of the cut-away wedge about the right springline
}
TARGET = (0.0, 6.0, -0.3)


def _kite(name, a, apex, half_deg, sgn, y0, y1, extra=0.0):
    """Prism of the notch: chord points on the boundary at 90 +- half, apex at r = apex, closed via the centre."""
    h = math.radians(half_deg + extra)
    p0 = (a * math.sin(h), sgn * a * math.cos(h))
    p1 = (-a * math.sin(h), sgn * a * math.cos(h))
    pts = [p0, (0.0, sgn * apex), p1, (0.0, 0.0)]
    if sgn < 0:
        pts = pts[::-1]
    return extrude_y(name, pts, y0, y1)


def build(d):
    a = d["a"]
    ra = d["r_over_a"] * a
    half = d["half_deg"]
    L = d["length_D"] * 2 * a
    y0, y1 = 0.0, L
    B = 2.0 * a
    rock = box("rock", -B, B, y0, y1, -B, B)
    hole = cylinder_y("hole", a, y0 - 1, y1 + 1, n=160)
    # cut-away: a wedge about the right springline, clear of both notches
    R = 3 * B
    t = math.radians(d["cut_deg"])
    cut = extrude_y("cut", [(0.0, 0.0), (R * math.cos(t), -R * math.sin(t)), (R * math.cos(t), R * math.sin(t))],
                    y0 - 5, y1 + 5)

    damaged = []
    for sgn in (1, -1):
        outer = _kite(f"notch_outer_{sgn}", a, ra + d["damage"], half, sgn, y0, y1, extra=2.0)
        inner = _kite(f"notch_inner_{sgn}", a, ra, half, sgn, y0 - 1, y1 + 1)
        boolean(rock, outer)
        skin = duplicate(outer, f"damage_{sgn}")
        boolean(skin, inner)
        boolean(skin, hole)
        boolean(skin, cut)
        damaged.append(skin)
        remove(outer, inner)
    boolean(rock, hole)
    boolean(rock, cut)

    assign(rock, material("rock", GROUND, 0.9))
    car = material("spalled", CAR, 0.75)
    for s in damaged:
        assign(s, car)

    # loose slabs on the flanks of the invert notch
    rnd = random.Random(14)
    flake = material("flake", CAR_TINT, 0.8)
    h = math.radians(half)
    for k in range(d["flakes"]):
        side = 1 if k % 2 else -1
        f = rnd.uniform(0.35, 0.75)                     # position up the flank from the apex
        cx = side * f * a * math.sin(h)
        cz = -ra + f * (ra - a * math.cos(h))
        yy = y0 + 1.2 + (L - 2.4) * (k + rnd.uniform(0.1, 0.9)) / d["flakes"]
        w, l, t = rnd.uniform(0.5, 0.8), rnd.uniform(0.8, 1.3), 0.07
        o = box(f"flake_{k}", -w / 2, w / 2, -l / 2, l / 2, 0, t)
        slope = math.atan2(ra - a * math.cos(h), a * math.sin(h))
        o.rotation_euler = (rnd.uniform(-0.15, 0.15), -side * slope, rnd.uniform(-0.12, 0.12))
        o.location = (cx, yy, cz + 0.02)
        assign(o, flake)

    remove(hole, cut)
    camera((19.0, -34.0, 6.0), TARGET, lens=60)
    lights((-0.35, 0.7, -0.6))
