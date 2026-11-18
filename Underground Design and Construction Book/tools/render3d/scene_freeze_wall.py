"""Ch 25: a ring of freeze pipes around a shaft; the closed frozen wall and a window left open by seepage."""
import math

from common import *  # noqa: F401,F403

NAME = "freeze_wall"
CHAPTER = "c25-ground-improvement"
DATA_KEY = None
DEFAULTS = {
    "n": 16, "R": 3.0,                  # 16 freeze pipes on a 3 m ring: f9.freeze_wall
    "r_closed": 1.05,                   # column radius once the wall has closed: f9.freeze_wall
    "r_min": 0.32, "drop": 0.75,        # seepage law for the upstream columns: f9.freeze_wall
    "shift": 0.25,                      # downstream displacement of the columns under seepage: f9.freeze_wall
    "depth": 12.0,                      # pipe depth (illustrative)
    "band": (-7.5, -4.0),              # permeable layer with seepage (illustrative)
    "pipe_r": 0.07, "pipe_up": 0.8,     # freeze-pipe radius (thickened) and stick-up
}
TARGET = (0.0, 0.5, -5.5)


def _wall(name, centres, r, z0, z1, m=720):
    """The union of overlapping vertical circular columns as one closed solid (a scalloped ring).

    Along each ray from the ring centre the columns cover one interval, from the nearest entry
    to the farthest exit, so the wall is an annulus with radii rho_in(psi), rho_out(psi)."""
    rin, rout = [], []
    for k in range(m):
        psi = 2 * math.pi * k / m
        ux, uy = math.cos(psi), math.sin(psi)
        near, far = 1e9, 0.0
        for cx, cy in centres:
            b = cx * ux + cy * uy
            c = cx * cx + cy * cy - r * r
            disc = b * b - c
            if disc > 0 and b > 0:                          # only columns ahead along the ray
                near, far = min(near, b - math.sqrt(disc)), max(far, b + math.sqrt(disc))
        rin.append((near * ux, near * uy))
        rout.append((far * ux, far * uy))
    verts = [(x, y, z) for z in (z0, z1) for ring in (rout, rin) for x, y in ring]
    o0, i0, o1, i1 = 0, m, 2 * m, 3 * m
    faces = []
    for k in range(m):
        kn = (k + 1) % m
        faces += [(o0 + k, o0 + kn, o1 + kn, o1 + k),        # outer face
                  (i0 + kn, i0 + k, i1 + k, i1 + kn),        # inner face
                  (o1 + k, o1 + kn, i1 + kn, i1 + k),        # top
                  (o0 + kn, o0 + k, i0 + k, i0 + kn)]        # bottom
    return mesh(name, verts, faces)                    # wound outward by construction


def build(d):
    n, R, rc = d["n"], d["R"], d["r_closed"]
    depth, (zb0, zb1) = d["depth"], d["band"]
    base = -depth - 3.0
    # ground in three layers, the middle one permeable; the front half is cut away
    layers = [("upper", zb1, 0.0, GROUND), ("aquifer", zb0, zb1, GROUND_LIGHT), ("lower", base, zb0, GROUND)]
    for name, z0, z1, col in layers:
        g = box(name, -9.0, 9.0, -9.0, 9.0, z0, z1)
        c = box("cut_" + name, -10.0, 10.0, -10.0, 0.0, z0 - 1, z1 + 1)
        boolean(g, c)
        remove(c)
        assign(g, material(name, col, 0.9))

    mal = material("frozen", MAL, 0.55)
    car = material("frozen_open", CAR, 0.55)
    steel = material("pipe", STEEL, 0.35, metallic=0.6)
    phis = [2 * math.pi * k / n for k in range(n)]
    gap = 2 * R * math.sin(math.pi / n)                 # pipe spacing on the ring
    # seepage towards +y: the upstream columns (facing the viewer) stay small
    rad = [max(d["r_min"], rc - d["drop"] * max(0.0, -math.sin(p)) ** 0.8) for p in phis]
    centres = [(R * math.cos(p), R * math.sin(p)) for p in phis]
    for name, z0, z1 in (("wall_lower", -depth, zb0), ("wall_upper", zb1, 0.05)):
        assign(smooth(_wall(name, centres, rc, z0, z1)), mal)
    for k, p in enumerate(phis):
        cx, cy = centres[k]
        # in the permeable layer: smaller and displaced downstream; carmine where a gap stays open
        r = rad[k]
        closes = r + rad[k - 1] >= gap and r + rad[(k + 1) % n] >= gap
        e = 0.004 * (k % 3)                              # no two caps coplanar
        o = cylinder_between("col_band", (cx, cy + d["shift"], zb0 + 0.01 + e), (cx, cy + d["shift"], zb1 - 0.01 - e),
                             r, n=64)
        assign(o, mal if closes else car)
        smooth(o)
        assign(cylinder_between("pipe", (cx, cy, -depth - 0.3), (cx, cy, d["pipe_up"]), d["pipe_r"], n=16), steel)
    camera((17.0, -24.0, 12.0), TARGET, lens=38)
    lights((-0.35, 0.6, -0.7), strength=3.2)
