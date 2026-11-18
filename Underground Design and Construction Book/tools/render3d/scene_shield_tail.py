"""Ch 23: longitudinal half cut-away of a shield tail, with annulus grout and a grout shortfall at the crown."""
import math

from common import *  # noqa: F401,F403

NAME = "shield_tail"
CHAPTER = "c23-mechanised"
DATA_KEY = None
DEFAULTS = {
    "Db": 6.2,             # bore diameter, c23 annulus example and tools/figs/f8.py annulus_grout
    "De": 6.0,             # segment extrados diameter, same source
    "Lr": 1.5,             # ring length, same source
    "t_seg": 0.3,          # segment thickness, f8.py schematic (intrados radius 2.7 m)
    "V_ann": 2.87,         # theoretical annulus per ring, c23 text: (pi/4)(6.2^2 - 6.0^2)(1.5)
    "short": 0.5,          # persistent grout shortfall per ring, c23 text (V_L 1.1 per cent)
    "exag": 3.0,           # radial exaggeration of the annulus, as in fig:annulus-grout
    "skin": 0.05,          # tail skin thickness (illustrative), the rest of the annulus is tail clearance
    "seals": [0.25, 0.6, 0.95],    # wire-brush tail seal rows, distance ahead of the tail end (illustrative)
    "seal_w": 0.12,        # seal row width (illustrative)
    "rings_behind": 4,     # rings shown behind the tail
    "rings_in": 2,         # rings still inside the tail
    "shield_L": 3.6,       # length of the shield shown ahead of the tail end
}
TARGET = (-1.0, -0.9, 0.0)


def _annulus(name, r0, r1, y0, y1, n=192):
    outer = cylinder_y(name, r1, y0, y1, n=n)
    inner = cylinder_y(name + "_in", r0, y0 - 1, y1 + 1, n=n)
    boolean(outer, inner)
    remove(inner)
    return outer


def _sector_y(name, r0, r1, th0, th1, y0, y1, n=48):
    """Annular sector about the crown: angles from the crown, x = r sin, z = r cos."""
    ths = [math.radians(th0 + (th1 - th0) * k / n) for k in range(n + 1)]
    pts = [(r1 * math.sin(t), r1 * math.cos(t)) for t in ths]
    pts += [(r0 * math.sin(t), r0 * math.cos(t)) for t in reversed(ths)]
    return extrude_y(name, pts, y0, y1)


def build(d):
    re = d["De"] / 2
    ri = re - d["t_seg"]
    gap = (d["Db"] - d["De"]) / 2 * d["exag"]         # 0.1 m drawn as 0.3 m
    rb = re + gap
    skin_t = d["skin"] * d["exag"]
    Lr = d["Lr"]
    y_back = -d["rings_behind"] * Lr
    y_ring_end = d["rings_in"] * Lr
    y_front = d["shield_L"]
    # shortfall at the crown: the fraction of the annulus left empty, 0.5 / 2.87
    frac = d["short"] / d["V_ann"]
    half = 180.0 * frac                               # +- 31.4 deg about the crown

    # cut-away: the near half, a longitudinal section through crown and invert
    cut = box("cut", 0.0, 20, y_back - 5, y_front + 5, -20, 20)
    B = rb + 0.9

    ground = box("ground", -B, B, y_back, y_front, -B, B)
    bore = cylinder_y("bore", rb, y_back - 1, y_front + 1, n=192)
    boolean(ground, bore)
    boolean(ground, cut)
    assign(ground, material("ground", GROUND, 0.9))

    # segment rings, with dark circumferential joints
    conc = material("segments", CONCRETE, 0.8)
    joint = material("joint", STEEL, 0.8)
    for k in range(d["rings_behind"] + d["rings_in"]):
        y0 = y_back + k * Lr
        r = _annulus(f"ring_{k}", ri, re, y0 + 0.015, y0 + Lr - 0.015)
        boolean(r, cut)
        assign(r, conc)
        if k:
            j = _annulus(f"joint_{k}", ri + 0.01, re - 0.01, y0 - 0.015, y0 + 0.015)
            boolean(j, cut)
            assign(j, joint)

    # shield tail skin and the wire-brush seals in the tail clearance
    skin = _annulus("skin", rb - skin_t, rb, 0.0, y_front)
    boolean(skin, cut)
    assign(skin, material("skin", STEEL, 0.4, metallic=0.4))
    seal_m = material("seal", "#2E2E2E", 0.9)
    for s in d["seals"]:
        o = _annulus("seal", re, rb - skin_t, s - d["seal_w"] / 2, s + d["seal_w"] / 2)
        boolean(o, cut)
        assign(o, seal_m)

    # annulus behind the tail: grout, short at the crown
    grout = _annulus("grout", re, rb, y_back, 0.0)
    crown = _sector_y("short", re - 0.05, rb + 0.05, -half, half, y_back - 1, 1.0)
    boolean(grout, crown)
    boolean(grout, cut)
    assign(grout, material("grout", MAL, 0.7))
    void = _sector_y("void", re, rb, -half, half, y_back, 0.0)
    boolean(void, cut)
    assign(void, material("shortfall", CAR, 0.7))
    remove(crown, cut, bore)

    camera((17.0, 6.5, 6.5), TARGET, lens=42)
    lights((-0.45, 0.5, -0.75), strength=3.0)
