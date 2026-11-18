"""Ch 5: orientation bias in RQD, a vertical and an inclined core through one jointed block."""
import math
import random

from common import *  # noqa: F401,F403
from mathutils import Vector

NAME = "rqd_orientation"
CHAPTER = "c05-site-investigation"
DATA_KEY = None
DEFAULTS = {
    # figures/ch05/rqd-orientation-bias.csv and tools/figs/f2.py rqd_orientation()
    "lam_b": 6.8,             # bedding frequency (1/m), spacing 0.147 m
    "lam_v": 12.0,            # sub-vertical set frequency (1/m), spacing 0.083 m
    "scale": 2.5,             # spacings drawn 2.5 times true, as in the 2D section of f2.py
    "jitter": 0.35,           # +- fraction of spacing, so the joints are not a regular grid
    "tilt": 2.0,              # sub-vertical set, degrees from vertical (illustrative)
    "theta": 60.0,            # inclined hole, degrees from vertical (f2.py)
    "B": 4.0, "W": 1.4, "H": 2.2,   # block (m, drawn scale)
    "gap": 0.022,             # drawn joint aperture
    "r_core": 0.055, "r_hole": 0.068,
    "x_vert": None,           # chosen clear of the sub-vertical set
    "x_incl": 0.95,           # collar of the inclined hole
    "seed": 5,
}
TARGET = (2.0, 0.6, 1.0)


def _hexa(name, c):
    """A hexahedron from 8 corners: bottom face c[0..3], top face c[4..7], same order."""
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    o = mesh(name, c, faces)
    recalc_normals(o)
    return o


def _planes(mean, lo, hi, rng, jit):
    out, x = [], lo + rng.uniform(0.3, 1.0) * mean
    while x < hi - 0.25 * mean:
        out.append(x)
        x += mean * rng.uniform(1 - jit, 1 + jit)
    return out


def build(d):
    rng = random.Random(d["seed"])
    B, W, H, g = d["B"], d["W"], d["H"], d["gap"]
    sb, sv = d["scale"] / d["lam_b"], d["scale"] / d["lam_v"]
    t = math.tan(math.radians(d["tilt"]))
    zb = _planes(sb, 0.0, H, rng, d["jitter"])                 # bedding planes z = const
    xv = _planes(sv, 0.04, B - H * t - 0.12, rng, d["jitter"])  # sub-vertical: x = x0 + z tan(tilt)

    # vertical hole: in the widest column, clear of the sub-vertical joints over the full height
    cols = [(xv[k], xv[k + 1]) for k in range(len(xv) - 1) if xv[k] > 0.15 and xv[k + 1] + H * t < 1.4]
    a, b = max(cols, key=lambda c: c[1] - c[0])
    x_vert = d["x_vert"] or 0.5 * (a + b) + 0.5 * H * t
    th = math.radians(d["theta"])
    p_in = Vector((d["x_incl"], 0.0, H))
    dir_in = Vector((math.sin(th), 0.0, -math.cos(th)))
    L_in = min((B - p_in.x) / dir_in.x, H / -dir_in.z)
    holes = [(Vector((x_vert, 0.0, H)), Vector((0, 0, -1)), H, MAL),
             (p_in, dir_in, L_in, CAR)]

    cutters = []
    for p, u, L, _ in holes:
        cutters.append(cylinder_between("hole", p - u * 0.3, p + u * (L + 0.3), d["r_hole"], n=48))

    # rock pieces between the joints, with a small drawn aperture
    zs = [0.0] + zb + [H]
    xs = [None] + xv + [None]
    rock = material("rock", "#B4B4B2", 0.85)
    for i in range(len(zs) - 1):
        z0 = zs[i] + (g / 2 if i > 0 else 0.0)
        z1 = zs[i + 1] - (g / 2 if i + 1 < len(zs) - 1 else 0.0)
        for k in range(len(xs) - 1):
            def xl(z):
                return 0.0 if xs[k] is None else max(0.0, xs[k] + z * t + g / 2)

            def xr(z):
                return B if xs[k + 1] is None else min(B, xs[k + 1] + z * t - g / 2)
            if xr(z0) - xl(z0) < 0.02 or xr(z1) - xl(z1) < 0.02:
                continue
            c = [(xl(z0), 0, z0), (xr(z0), 0, z0), (xr(z0), W, z0), (xl(z0), W, z0),
                 (xl(z1), 0, z1), (xr(z1), 0, z1), (xr(z1), W, z1), (xl(z1), W, z1)]
            o = _hexa(f"piece_{i}_{k}", c)
            for cu in cutters:
                boolean(o, cu)
            assign(o, rock)
    remove(*cutters)

    # core pieces: the core breaks wherever a joint crosses the hole axis
    for p, u, L, col in holes:
        cuts = []
        for z in zb:
            if abs(u.z) > 1e-9:
                s = (z - p.z) / u.z
                if 0 < s < L:
                    cuts.append(s)
        for x0 in xv:                              # plane x - z tan = x0
            den = u.x - u.z * t
            if abs(den) > 1e-9:
                s = (x0 + p.z * t - p.x) / den
                if 0 < s < L:
                    cuts.append(s)
        cuts = [0.0] + sorted(cuts) + [L]
        cmat = material(f"core_{col}", col, 0.55)
        for s0, s1 in zip(cuts[:-1], cuts[1:]):
            a0 = s0 + (g / 2 if s0 > 0 else 0.0)
            a1 = s1 - (g / 2 if s1 < L else 0.0)
            if a1 - a0 < 0.01:
                continue
            o = cylinder_between("core", p + u * a0, p + u * a1, d["r_core"], n=40)
            assign(o, cmat)
            smooth(o)
        n_long = sum(1 for s0, s1 in zip(cuts[:-1], cuts[1:]) if (s1 - s0) / d["scale"] >= 0.1)
        print(f"  hole {col}: length {L:.2f} m drawn, {len(cuts) - 1} pieces, {n_long} longer than 0.1 m true")

    camera((1.1, -6.4, 5.0), TARGET, lens=46)
    lights((-0.35, 0.6, -0.75), strength=2.6, world=0.38)
