"""Ch 16: the plastic zone developing behind the face of a deep tunnel (cut-away)."""
from common import *  # noqa: F401,F403

NAME = "plastic_zone"
CHAPTER = "c16-analytical"
DATA_KEY = "plastic_zone"            # geometry from figures/render/scenes.json
DEFAULTS = {}
TARGET = (0.0, -6.0, -6.0)


def build(d):
    a, rp = d["a"], d["rp_supported"]
    y_back, y_front = -30.0, 18.0
    # plastic zone envelope: grows with the displacement released (an illustration
    # of the supported radius developing behind the face, not a 3D solution)
    X, u = d["ldp_X"], d["ldp_u"]
    pairs = sorted(zip([-x for x in X], u))                    # scene y = -X
    ys = [y for y, _ in pairs if y_back < y <= 0.0]
    us = [w for y, w in pairs if y_back < y <= 0.0]
    umax = max(us)
    rs = [a + (rp - a) * min(1.0, w / umax) ** 0.8 for w in us]
    ys, rs = [y_back] + ys, [rs[0]] + rs                      # flush with the back of the block
    # ahead of the face the yielded core closes as a rounded nose over about one radius
    r_face, nose = rs[-1], 1.0 * a
    for k in range(1, 13):
        t = k / 12
        ys.append(nose * t)
        rs.append(max(0.05, r_face * math.sqrt(max(0.0, 1.0 - t * t))))
    zone = revolve_y("plastic_envelope", ys, rs)
    hole = cylinder_y("excavation", a, y_back - 2, 0.0)
    cut = box("cut", 0, 40, y_back - 5, y_front + 5, 0, 40)

    ground = box("ground", -15, 15, y_back, y_front, -15, 15)
    boolean(ground, zone)
    boolean(ground, cut)
    plastic = duplicate(zone, "plastic")
    boolean(plastic, hole)
    boolean(plastic, cut)
    lining = cylinder_y("lining", a, y_back, -d["install_X"])
    inner = cylinder_y("lining_inner", a - d["lining_t"], y_back - 1, 0.0)
    boolean(lining, inner)
    boolean(lining, cut)
    remove(zone, hole, cut, inner)
    assign(ground, material("ground", GROUND, 0.9))
    assign(plastic, material("plastic", CAR, 0.8))
    assign(lining, material("lining", MAL, 0.6))
    for o in (plastic, lining):
        smooth(o)
    camera((52, 42, 34), TARGET, lens=32)
    lights((-0.4, -0.6, -0.8), world=0.6)
