"""Ch 26: a building on the greenfield trough, its sagging and hogging lengths split at the inflection point."""
import math

from common import *  # noqa: F401,F403

NAME = "building_trough"
CHAPTER = "c26-movement-monitoring"
DATA_KEY = None
DEFAULTS = {
    "z0": 20.0, "D": 6.0, "VL": 1.0, "K": 0.5,      # f9.deflection_ratio, Fig. deflection-ratio, i = K z0 = 10 m
    "y0": -6.0, "y1": 24.0,                         # building ends across the trough: f9.deflection_ratio
    "depth": 12.0, "height": 10.0,                  # building plan depth and height (illustrative)
    "exag": 300.0,                                  # vertical exaggeration of the settlement
    "half_len": 18.0,                               # ground block half-length along the tunnel
}
TARGET = (5.0, 0.0, -5.0)


def build(d):
    z0, D, ex = d["z0"], d["D"], d["exag"]
    i = d["K"] * z0
    vs = d["VL"] / 100 * math.pi * D ** 2 / 4
    smax = vs / (i * math.sqrt(2 * math.pi))         # 11.3 mm, the peak of figures/ch26/deflection-ratio.csv

    def w(x):                                         # exaggerated settlement (m, downward)
        return smax * math.exp(-x ** 2 / (2 * i ** 2)) * ex

    # transverse offset y of the book runs along scene X; the tunnel axis along scene Y
    xs = [-40.0 + 0.5 * k for k in range(161)]
    hl = d["half_len"]
    ys = [-hl + 2.0 * k for k in range(int(hl) + 1)]
    ground = heightfield("ground", xs, ys, lambda x, y: -w(x), base=-29.0)
    bore = cylinder_y("bore", D / 2, -hl - 2, hl + 2, x=0.0, z=-z0)
    boolean(ground, bore)
    ring = cylinder_y("ring", D / 2, -hl, hl, x=0.0, z=-z0)
    ring_in = cylinder_y("ring_in", D / 2 - 0.35, -hl - 1, hl + 1, x=0.0, z=-z0)
    boolean(ring, ring_in)
    remove(bore, ring_in)
    assign(ground, material("ground", GROUND, 0.9))
    assign(ring, material("segments", CONCRETE, 0.7))
    smooth(ring)

    # the building follows the greenfield trough (no stiffness): each part a deformed block
    dep, hgt = d["depth"], d["height"]

    def block(name, xa, xb, colour):
        nseg = max(2, int((xb - xa) / 0.25))
        xx = [xa + (xb - xa) * k / nseg for k in range(nseg + 1)]
        verts = []
        for x in xx:
            base = -w(x)
            verts += [(x, -dep / 2, base), (x, dep / 2, base), (x, dep / 2, base + hgt), (x, -dep / 2, base + hgt)]
        faces = []
        for k in range(nseg):
            a0, b0 = 4 * k, 4 * (k + 1)
            for q in range(4):
                faces.append((a0 + q, b0 + q, b0 + (q + 1) % 4, a0 + (q + 1) % 4))
        faces.append((3, 2, 1, 0))
        e = 4 * nseg
        faces.append((e, e + 1, e + 2, e + 3))
        o = mesh(name, verts, faces)
        recalc_normals(o)
        assign(o, material(name, colour, 0.65))
        return o

    block("sagging", d["y0"], i, MAL)
    block("hogging", i, d["y1"], CAR)
    camera((30.0, -118.0, 34.0), (3.0, 0.0, -9.0), lens=38)
    lights((-0.35, 0.55, -0.75), strength=3.2)
