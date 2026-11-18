"""Ch 25: two overlapping stages of umbrella arch over the face, and fibreglass dowels in the core (cut-away)."""
import math

from common import *  # noqa: F401,F403

NAME = "umbrella_arch"
CHAPTER = "c25-ground-improvement"
DATA_KEY = None
DEFAULTS = {
    "H": 9.0,                            # tunnel height: f9.presupport
    "a": 5.0, "z_spring": 4.0,           # horseshoe of 10 m span, springline at 4 m (illustrative, gives H = 9 m)
    "L_pipe": 15.0,                      # umbrella pipe length: f9.presupport
    "stages": [-14.0, -3.0],             # starting chainage of each umbrella, face at y = 0: f9.presupport
    "slope": 0.07,                       # outward inclination of the pipes (illustrative, 7 per cent)
    "arc_deg": 65.0, "pipe_spacing": 0.5,   # umbrella over +-65 deg of the crown at about 0.5 m (illustrative)
    "pipe_r": 0.09,                      # pipe radius, thickened for visibility
    "L_dowel": 15.0, "dowel_grid": 1.5,  # face dowels: length from f9.presupport, grid illustrative
    "lining_t": 0.3,
    "y_back": -19.0, "y_front": 17.0,
}
TARGET = (0.0, -1.0, 5.0)


def _profile(a, zs, z_lo=0.0, n=64):
    """Horseshoe in XZ: walls of half-width a to the springline zs, semicircular crown of radius a."""
    pts = [(a, z_lo), (a, zs)]
    pts += [(a * math.cos(math.pi * k / n), zs + a * math.sin(math.pi * k / n)) for k in range(1, n)]
    pts += [(-a, zs), (-a, z_lo)]
    return pts


def build(d):
    a, zs, t = d["a"], d["z_spring"], d["lining_t"]
    yb, yf = d["y_back"], d["y_front"]
    void = extrude_y("void", _profile(a + 0.03, zs, -0.01), yb - 2, 0.0)   # clear of the lining extrados
    ground = box("ground", -14, 14, yb, yf, -3.0, 14.0)
    core_cut = extrude_y("core_cut", _profile(a, zs, -0.01), -0.5, yf + 1)
    boolean(ground, void)
    boolean(ground, core_cut)
    core = extrude_y("core", _profile(a, zs, 0.0), 0.0, yf)
    # the left half of the lining, inside the excavated profile, built directly (no boolean)
    n, ai = 32, a - t
    outer = [(-a * math.sin(math.pi / 2 * k / n), zs + a * math.cos(math.pi / 2 * k / n)) for k in range(n + 1)]
    inner = [(-ai * math.sin(math.pi / 2 * k / n), zs + ai * math.cos(math.pi / 2 * k / n)) for k in range(n + 1)]
    shell = outer + [(-a, 0.0), (-ai, 0.0)] + inner[::-1]
    lining = extrude_y("lining", shell, yb, -1.0)
    half = box("half", 0.0, 30.0, yb - 5, yf + 5, -10.0, 30.0)
    for o in (ground, core):
        boolean(o, half)
    remove(void, core_cut, half)
    # opaque below the springline, lightly translucent above it so the umbrellas show
    upper = duplicate(ground, "ground_upper")
    lo_cut = box("lo_cut", -20, 20, yb - 5, yf + 5, -10.0, zs)
    hi_cut = box("hi_cut", -20, 20, yb - 5, yf + 5, zs, 30.0)
    boolean(upper, lo_cut)
    boolean(ground, hi_cut)
    remove(lo_cut, hi_cut)
    assign(upper, material("ground_upper", GROUND, 0.8, alpha=0.14))
    assign(ground, material("ground", GROUND, 0.9))
    assign(core, material("core", GROUND_DARK, 0.9))
    assign(lining, material("lining", CONCRETE, 0.8))

    # umbrella pipes: start just outside the lining, diverging outward at the given slope
    pipe = material("pipe", MAL, 0.4)
    r0 = a + 0.2
    arc = math.radians(d["arc_deg"])
    m = int(2 * arc * r0 / d["pipe_spacing"]) + 1
    run = d["L_pipe"] / math.sqrt(1 + d["slope"] ** 2)
    for y0 in d["stages"]:
        for k in range(m):
            th = -arc + 2 * arc * k / (m - 1)
            ux, uz = math.sin(th), math.cos(th)
            p0 = (r0 * ux, y0, zs + r0 * uz)
            r1 = r0 + d["slope"] * run
            p1 = (r1 * ux, y0 + run, zs + r1 * uz)
            assign(cylinder_between("pipe", p0, p1, d["pipe_r"], n=12), pipe)

    # fibreglass dowels in the advance core, on a grid inside the face
    dowel = material("dowel", MAL_TINT, 0.5)
    g, edge = d["dowel_grid"], 0.8
    for i in range(-3, 3):
        x = (i + 0.5) * g
        if abs(x) > a - edge:
            continue
        for j in range(8):
            z = 0.9 + j * g
            inside = (z <= zs and abs(x) <= a - edge) or (z > zs and math.hypot(x, z - zs) <= a - edge)
            if inside:
                assign(cylinder_between("dowel", (x, -0.02, z), (x, d["L_dowel"], z), 0.1, n=10), dowel)
    camera((42.0, -14.0, 25.0), TARGET, lens=38)
    lights((-0.75, 0.35, -0.55), strength=3.2, world=0.6)   # low from the open side, into the tunnel
