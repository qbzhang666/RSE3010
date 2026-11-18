"""Ch 8: borehole breakouts and drilling-induced tensile fractures in a rock cube (quarter cut-away)."""
import math

from common import *  # noqa: F401,F403

NAME = "breakout_block"
CHAPTER = "c08-in-situ-stress"
DATA_KEY = None
DEFAULTS = {
    # figures/ch08/borehole-breakout.csv and tools/figs/f3.py breakout()
    "sH": 45.0, "sh": 20.0, "C0": 80.0,   # MPa: the sigma_H = 45 MPa case of the worked figure
    "depth": 0.45,            # breakout depth as a fraction of the radius (f3.py section)
    "tif_depth": 0.35,        # tensile fracture length as a fraction of the radius (f3.py section)
    "a": 0.19,                # borehole radius (m), schematic
    "half": 0.45,             # half side of the cube (m), schematic
    "tif_t": 0.008,           # drawn fracture thickness (m)
    "step": -0.12,            # floor of the cut-away quadrant, a horizontal slice (m)
    "toe": -0.33,             # bottom of the drilled hole (m)
}
TARGET = (0.0, 0.0, -0.05)


def half_width_deg(sH, sh, C0):
    """Breakout half-width about the sigma_h azimuth from the Kirsch wall stress (f3.py)."""
    return 90.0 - 0.5 * math.degrees(math.acos((sH + sh - C0) / (2 * (sH - sh))))


def _boolean_transfer(target, cutter):
    """Difference that carries the cutter's material onto the new faces."""
    mod = target.modifiers.new(name=f"bool_{cutter.name}", type="BOOLEAN")
    mod.operation = "DIFFERENCE"
    mod.solver = "EXACT"
    mod.object = cutter
    mod.material_mode = "TRANSFER"
    bpy.context.view_layer.objects.active = target
    bpy.ops.object.modifier_apply(modifier=mod.name)
    return target


def _loft(name, rings):
    """A closed solid through rings of equal length (each a closed loop of 3D points)."""
    n = len(rings[0])
    verts = [p for r in rings for p in r]
    faces = [tuple(range(n))[::-1], tuple(range((len(rings) - 1) * n, len(rings) * n))]
    for k in range(len(rings) - 1):
        a, b = k * n, (k + 1) * n
        faces += [(a + i, a + (i + 1) % n, b + (i + 1) % n, b + i) for i in range(n)]
    o = mesh(name, verts, faces)
    recalc_normals(o)
    return o


def build(d):
    a, h = d["a"], d["half"]
    hw = half_width_deg(d["sH"], d["sh"], d["C0"])
    print(f"  breakout width {2 * hw:.1f} deg")
    # sigma_H along x (azimuth 0), sigma_h along y: breakouts centred on +-y (90 and 270 deg)
    rock = box("rock", -h, h, -h, h, -h, h)
    assign(rock, material("rock", "#ADADAB", 0.85))
    hole = cylinder_between("hole", (0, 0, d["toe"]), (0, 0, h + 0.1), a, n=192)
    assign(hole, material("wall", "#ADADAB", 0.85))
    _boolean_transfer(rock, hole)

    nz, m = 28, 60
    for c0 in (90.0, 270.0):
        rings = []
        for j in range(nz + 1):
            z = d["toe"] + 0.04 + (h + 0.05 - d["toe"] - 0.04) * j / nz
            f = d["depth"] * (0.86 + 0.08 * math.sin(7.0 * z / h + c0) + 0.06 * math.sin(17.0 * z / h))
            inner, outer = [], []
            for i in range(m + 1):
                ph = c0 - hw + 2 * hw * i / m
                u = (ph - c0) / hw
                rr = 1.0 + f * max(0.0, 1.0 - u * u) ** 0.3
                t = math.radians(ph)
                inner.append((0.9 * a * math.cos(t), 0.9 * a * math.sin(t), z))
                outer.append((rr * a * math.cos(t), rr * a * math.sin(t), z))
            rings.append(inner + outer[::-1])
        notch = _loft(f"notch_{int(c0)}", rings)
        assign(notch, material("breakout", CAR, 0.8))
        _boolean_transfer(rock, notch)
        remove(notch)
    remove(hole)

    # quarter cut-away: the quadrant x > 0, y < 0 is removed
    cut = box("cut", 0.0, h + 1, -h - 1, 0.0, d["step"], h + 1)
    assign(cut, material("rock_cut", "#C2C2C0", 0.85))
    _boolean_transfer(rock, cut)
    remove(cut)

    # drilling-induced tensile fractures: thin plates at the sigma_H azimuth, standing just proud
    tmat = material("tif", MAL, 0.5)
    L = d["tif_depth"] * a
    for sgn in (1, -1):
        x0 = sgn * (a - 0.003)
        x1 = sgn * (a + L)
        o = box("tif", min(x0, x1), max(x0, x1), -d["tif_t"] / 2, d["tif_t"] / 2, d["toe"] + 0.06, h + 0.002)
        assign(o, tmat)

    camera((1.45, -1.85, 2.55), TARGET, lens=50)
    lights((-0.45, 0.25, -0.85), strength=2.8, world=0.38)
