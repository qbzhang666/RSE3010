"""Ch 9: a tetrahedral wedge loosening from a tunnel crown, with pattern bolts.

The rock block is drawn by its edges only, so the wedge, solid carmine, is seen whole above
the crown, and the bolts, malachite, are seen crossing its faces into the
stable rock beyond.
"""
import math

from common import *  # noqa: F401,F403
from mathutils import Vector

NAME = "crown_wedge"
CHAPTER = "c09-structural-analysis"
DATA_KEY = "crown_wedge"
DEFAULTS = {}
TARGET = (0.0, 5.0, -0.2)


def build(d):
    a, L = d["a"], d["length"]
    h = d["apex_height"]
    y0, y1 = -1.0, L - 1.0
    rock = box("rock", -7, 7, y0, y1, -a - 0.8, a + 5.5)
    tunnel = cylinder_y("tunnel", a, y0 - 1, y1 + 1)
    boolean(rock, tunnel)
    # the excavated surface, opaque, so the opening reads as a tunnel
    shell = cylinder_y("shell", a + 0.04, y0, y1)
    boolean(shell, tunnel)
    # wedge: a pyramid on the crown, its base an arc patch, apex above
    th = math.radians(30)
    yb0, yb1 = 1.5, 7.5
    base = [(-a * math.sin(th), yb0, a * math.cos(th)), (a * math.sin(th), yb0, a * math.cos(th)),
            (a * math.sin(th), yb1, a * math.cos(th)), (-a * math.sin(th), yb1, a * math.cos(th))]
    apex = (0.0, (yb0 + yb1) / 2, a + h)
    wedge = mesh("wedge", base + [apex], [(3, 2, 1, 0), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)])
    recalc_normals(wedge)
    boolean(wedge, tunnel)
    boolean(shell, wedge)                           # the wedge's trace opens in the crown
    boolean(rock, wedge)
    wedge.location.z -= 0.3                         # shown as loosening from the crown
    remove(tunnel)
    remove(rock)                                    # the rock is shown by its block edges only
    frame = box("frame", -7, 7, y0, y1, -a - 0.8, a + 5.5)  # block edges only
    wf = frame.modifiers.new("wire", "WIREFRAME")
    wf.thickness = 0.06
    apply_modifiers(frame)
    assign(frame, material("edges", GROUND_DARK, 0.8))
    assign(shell, material("shell", GROUND, 0.9))
    assign(wedge, material("wedge", CAR, 0.6))
    smooth(shell)
    bolt_mat = material("bolt", MAL, 0.35, metallic=0.3)
    plate_mat = material("plate", MAL_DARK, 0.5)
    for x, y in d["bolts"]:
        ang = math.asin(max(-1, min(1, x / a)))
        p = Vector((a * math.sin(ang), y, a * math.cos(ang)))
        n = p.copy()
        n.y = 0
        n.normalize()
        assign(cylinder_between("bolt", p - n * 0.05, p + n * 4.0, 0.05), bolt_mat)
        assign(cylinder_between("plate", p - n * 0.06, p, 0.2, n=32), plate_mat)
    camera((18.0, -14.0, 9.5), TARGET, lens=31)
    lights((-0.5, 0.5, -0.7), strength=3.0, world=0.6)
