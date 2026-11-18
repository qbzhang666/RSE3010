"""Ch 26: the transverse settlement trough over twin bores, with buildings coloured by assessment."""
from common import *  # noqa: F401,F403

NAME = "settlement_trough"
CHAPTER = "c26-movement-monitoring"
DATA_KEY = "settlement_trough"
DEFAULTS = {}
TARGET = (0.0, 2.0, -8.0)


def build(d):
    exag = 300.0                                    # vertical exaggeration of the settlement
    xs, S = d["y"], d["S_mm"]
    ny, y0, y1 = 61, -45.0, 45.0
    verts, faces = [], []
    for j in range(ny):
        y = y0 + (y1 - y0) * j / (ny - 1)
        verts += [(x, y, -s / 1000 * exag) for x, s in zip(xs, S)]
    nx = len(xs)
    faces = [(j * nx + i, j * nx + i + 1, (j + 1) * nx + i + 1, (j + 1) * nx + i)
             for j in range(ny - 1) for i in range(nx - 1)]
    surf = mesh("surface", verts, faces)
    sol = surf.modifiers.new("solid", "SOLIDIFY")
    sol.thickness = 40.0
    sol.offset = -1.0
    apply_modifiers(surf)
    flat_base(surf, -32.0)
    for b in d["bores"]:
        t = cylinder_y("bore", d["D"] / 2, y0 - 2, y1 + 2, x=b["y"], z=-d["z0"])
        boolean(surf, t)
        remove(t)
        ring = cylinder_y("ring", d["D"] / 2, y0, y1, x=b["y"], z=-d["z0"])
        inner = cylinder_y("ring_in", d["D"] / 2 - 0.35, y0 - 1, y1 + 1, x=b["y"], z=-d["z0"])
        boolean(ring, inner)
        remove(inner)
        assign(ring, material("segments", MAL, 0.6))
        smooth(ring)
    assign(surf, material("ground", GROUND, 0.9))

    def surface_z(x):
        k = min(range(len(xs)), key=lambda i: abs(xs[i] - x))
        return -S[k] / 1000 * exag

    for b in d["buildings"]:
        lo = min(surface_z(x) for x in (b["y0"], (b["y0"] + b["y1"]) / 2, b["y1"]))
        yc = 12.0                                   # one row behind the section, none hidden
        o = box(f"bldg_{b['name']}", b["y0"], b["y1"], yc - b["depth"] / 2, yc + b["depth"] / 2, lo, lo + b["height"])
        assign(o, material(f"b_{b['name']}", CAR if b["assess"] else MAL_TINT, 0.7))
    camera((60, -185, 100), (0.0, 2.0, -12.0), lens=42)
    lights((-0.3, 0.6, -0.75))
