"""Ch 24: a large cavern part-way through benching, with its bolts and cables (cut-away)."""
import math

from common import *  # noqa: F401,F403
from mathutils import Vector

NAME = "cavern_sequence"
CHAPTER = "c24-openings"
DATA_KEY = None
DEFAULTS = {
    "B": 24.0, "Hw": 36.0, "rise": 6.0,             # span, wall height, arch rise: f9.cavern_sequence, Fig. cavern-sequence
    "levels": [28.0, 21.0, 14.0, 7.0],              # floors of the top heading and benches: f9.cavern_sequence
    "excavated_to": 14.0,                           # snapshot: heading, benches 2 and 3 out; bench 4 next (illustrative stage)
    "L_cable": 14.0, "L_bolt": 5.0,                 # cable and bolt lengths: f9.cavern_sequence
    "wall_rows": [4, 8, 11.5, 15, 18.5, 22, 25.5, 30, 33.5],   # wall rows, cable and bolt alternating: f9
    "n_crown": 9,                                   # crown positions, cable and bolt alternating: f9
    "ring": 3.0,                                    # ring spacing along the cavern (illustrative)
    "length": 45.0,                                 # cavern length shown (illustrative)
    "cut_y": 16.5,                                  # extent of the cut-away along the cavern
    "skin": 0.3,                                    # shotcrete skin, thickened for visibility
}
TARGET = (2.0, 14.0, 22.0)


def _outline(a, b, h, z_lo, n=96):
    """Cavern profile in XZ: vertical walls of half-width a to height h, elliptical arch of rise b."""
    pts = [(a, z_lo)]
    pts += [(a * math.cos(math.pi * k / n), h + b * math.sin(math.pi * k / n)) for k in range(n + 1)]
    pts += [(-a, z_lo)]
    return pts


def build(d):
    B, H, rise = d["B"], d["Hw"], d["rise"]
    a, L, t = B / 2, d["length"], d["skin"]
    zx = d["excavated_to"]
    y_end = L - 3.0
    void = extrude_y("void", _outline(a, rise, H, 0.0), -2.0, y_end)
    rock = box("rock", -29, 29, 0.0, L, -5.0, 58.0)
    boolean(rock, void)
    cut = box("cut", 0.0, 40.0, -2.0, d["cut_y"], 0.0, 70.0)
    boolean(rock, cut)

    # shotcrete skin over the excavated surface only (down to the current floor)
    skin = extrude_y("skin", _outline(a + t, rise + t, H, zx), 0.0, y_end)
    void2 = extrude_y("void2", _outline(a - 0.05, rise - 0.05, H, zx - 1.0), -1.0, y_end + 1.0)
    boolean(skin, void2)
    remove(void, void2, cut)

    # benches still in place: the next one translucent, the last one solid
    lv = d["levels"]
    nxt = box("bench_next", -a, a, 0.0, y_end, lv[3], lv[2])
    last = box("bench_last", -a, a, 0.0, y_end, 0.0, lv[3])

    assign(rock, material("rock", GROUND, 0.9))
    assign(last, material("rock", GROUND, 0.9))
    assign(nxt, material("bench_next", GROUND_DARK, 0.6, alpha=0.4))
    assign(skin, material("skin", CONCRETE, 0.8))

    cable_mat = material("cable", CAR, 0.45)
    bolt_mat = material("bolt", MAL, 0.45)
    Lc, Lb = d["L_cable"], d["L_bolt"]
    ys = [1.5 + d["ring"] * k for k in range(int((y_end - 1.5) / d["ring"]) + 1)]
    for y in ys:
        # crown: normal to the elliptical arch, starting on the skin
        for i in range(d["n_crown"]):
            tt = math.pi * (0.12 + 0.76 * i / (d["n_crown"] - 1))
            px, pz = (a + t) * math.cos(tt), H + (rise + t) * math.sin(tt)
            nx, nz = rise * math.cos(tt), a * math.sin(tt)
            nn = math.hypot(nx, nz)
            nx, nz = nx / nn, nz / nn
            ln, mat, r = (Lc, cable_mat, 0.2) if i % 2 == 0 else (Lb, bolt_mat, 0.16)
            assign(cylinder_between("crown", (px, y, pz), (px + ln * nx, y, pz + ln * nz), r, n=12), mat)
        # walls: horizontal rows, only on the lifts already exposed
        for j, z in enumerate(d["wall_rows"]):
            if z < zx:
                continue
            ln, mat, r = (Lc, cable_mat, 0.2) if j % 2 == 0 else (Lb, bolt_mat, 0.16)
            for s in (1, -1):
                x0 = s * (a + t)
                assign(cylinder_between("wall", (x0, y, z), (x0 + s * ln, y, z), r, n=12), mat)
    camera((95, -95, 62), (2.0, 14.0, 23.0), lens=40)
    lights((-0.35, 0.85, -0.45), strength=3.2, world=0.6)
    # soft fill from the open end, so the inside of the cavern is not lost in shadow
    fill = bpy.data.lights.new("fill", "AREA")
    fill.size, fill.energy = 30.0, 3000.0
    fo = bpy.data.objects.new("fill", fill)
    bpy.context.collection.objects.link(fo)
    fo.location = (10.0, -40.0, 30.0)
    fo.rotation_euler = (Vector(TARGET) - fo.location).to_track_quat("-Z", "Y").to_euler()
