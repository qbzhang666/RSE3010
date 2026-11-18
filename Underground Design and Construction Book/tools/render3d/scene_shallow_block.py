"""Ch 1: a street block over a shallow tunnel, with its settlement trough and drawdown (cut-away)."""
import csv
import math
import os

from common import *  # noqa: F401,F403

NAME = "shallow_block"
CHAPTER = "c01-challenges"
DATA_KEY = None
DEFAULTS = {
    # figures/ch01/chal-shallow-setting.csv and tools/figs/f1.py shallow_setting()
    "D": 6.3,                 # tunnel diameter (m), c01 Sec. shallow-setting
    "z0": 20.0,               # axis depth (m)
    "i": 10.0,                # trough width parameter, K z0 with K = 0.5 (illustrative)
    "smax_mm": 12.436,        # Smax for VL = 1 % (illustrative), CSV column Smax_mm
    "exag": 300.0,            # vertical exaggeration of the settlement only
    "lining_t": 0.30,         # lining thickness (m), illustrative
    # Table tab:shallow-metro-units: unit, top, base (m)
    "units": [(4, 0.0, 5.0), (3, 5.0, 15.0), (2, 15.0, 30.0), (1, 30.0, 42.0)],
    # water table: -2.5 m far field, drawn down 9 m with sigma 30 m (CSV water_table_m)
    "csv": "figures/ch01/chal-shallow-setting.csv",
    "half_x": 80.0, "half_y": 45.0,
    # buildings from the 2D section (x0, width, height), mirrored into two rows
    "buildings": [(-70, 14, 12), (-40, 16, 18), (28, 18, 22), (60, 12, 9)],
}
TARGET = (0.0, 0.0, -14.0)

UNIT_GREY = {4: "#C9C9C7", 3: "#B0B0AE", 2: "#959593", 1: "#7A7A78"}


def _book_path(rel):
    """Data file relative to the book root (tools/render3d/../..), or, on M3, to the
    folder that holds the driver and render3d/ (render3d/..)."""
    here = os.path.dirname(os.path.abspath(__file__))
    for root in (os.path.join(here, "..", ".."), os.path.join(here, "..")):
        p = os.path.normpath(os.path.join(root, rel))
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{rel} not found under {here}/../.. or {here}/..")


def _water_table(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    ys = [float(r["y_m"]) for r in rows]
    wt = [float(r["water_table_m"]) for r in rows]
    return ys, wt


def _interp(x, xs, vs):
    if x <= xs[0]:
        return vs[0]
    if x >= xs[-1]:
        return vs[-1]
    lo, hi = 0, len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    t = (x - xs[lo]) / (xs[hi] - xs[lo])
    return vs[lo] + t * (vs[hi] - vs[lo])


def _warped_slab(name, xs, ys, ztop, zbase):
    """A closed solid with top z = ztop(x) on the grid xs by ys and a flat base at zbase."""
    nx, ny = len(xs), len(ys)
    verts = [(x, y, ztop(x)) for y in ys for x in xs] + [(x, y, zbase) for y in ys for x in xs]
    n = nx * ny
    idx = lambda i, j: j * nx + i              # noqa: E731
    faces = [(idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1))
             for j in range(ny - 1) for i in range(nx - 1)]
    faces += [(n + idx(i, j + 1), n + idx(i + 1, j + 1), n + idx(i + 1, j), n + idx(i, j))
              for j in range(ny - 1) for i in range(nx - 1)]
    for i in range(nx - 1):                    # sides at y0 and y1
        faces.append((idx(i, 0), n + idx(i, 0), n + idx(i + 1, 0), idx(i + 1, 0)))
        faces.append((idx(i + 1, ny - 1), n + idx(i + 1, ny - 1), n + idx(i, ny - 1), idx(i, ny - 1)))
    for j in range(ny - 1):                    # sides at x0 and x1
        faces.append((idx(0, j + 1), n + idx(0, j + 1), n + idx(0, j), idx(0, j)))
        faces.append((idx(nx - 1, j), n + idx(nx - 1, j), n + idx(nx - 1, j + 1), idx(nx - 1, j + 1)))
    o = mesh(name, verts, faces)
    recalc_normals(o)
    return o


def _corner_colour_material(name, attr):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    p = nt.nodes.get("Principled BSDF")
    a = nt.nodes.new("ShaderNodeAttribute")
    a.attribute_name = attr
    nt.links.new(a.outputs["Color"], p.inputs["Base Color"])
    p.inputs["Roughness"].default_value = 0.9
    return m


def build(d):
    hx, hy = d["half_x"], d["half_y"]
    r, z0, i = d["D"] / 2, d["z0"], d["i"]
    ex = d["exag"]
    smax = d["smax_mm"] / 1000.0

    def settle(x):                                  # metres, true scale
        return smax * math.exp(-x * x / (2 * i * i))

    xs = [-hx + 2 * hx * k / 160 for k in range(161)]
    xs = sorted(set(xs + [k * 0.5 for k in range(-40, 41)]))
    ys = [-hy + 2 * hy * k / 18 for k in range(19)]

    cut = box("cut", 0.0, hx + 5, -hy - 5, 0.0, -60, 30)   # front-right quarter removed
    tunnel = cylinder_y("tunnel", r, -hy - 2, hy + 2, z=-z0)

    solids = []
    for unit, top, base in d["units"]:
        if unit == 4:
            o = _warped_slab("unit4", xs, ys, lambda x: -settle(x) * ex, -base)
        else:
            o = box(f"unit{unit}", -hx, hx, -hy, hy, -base, -top)
        if top < z0 + r and base > z0 - r:
            boolean(o, tunnel)
        boolean(o, cut)
        solids.append((unit, o))

    # colour the top of unit 4 by settlement: grey far field to carmine at the axis
    u4 = dict(solids)[4]
    me = u4.data
    attr = me.color_attributes.new("settle", "FLOAT_COLOR", "CORNER")
    g0, c1 = rgba(UNIT_GREY[4]), rgba(CAR)
    for poly in me.polygons:
        up = poly.normal.z > 0.6
        for li in poly.loop_indices:
            v = me.vertices[me.loops[li].vertex_index].co
            t = settle(v.x) / smax if (up and v.z > -4.5) else 0.0
            t = min(1.0, t) ** 0.8
            attr.data[li].color = tuple(g0[k] * (1 - t) + c1[k] * t for k in range(3)) + (1.0,)
    assign(u4, _corner_colour_material("unit4_settle", "settle"))
    for unit, o in solids:
        if unit != 4:
            assign(o, material(f"unit{unit}", UNIT_GREY[unit], 0.9))

    # lining: full tube, half exposed in the cut-away
    lining = cylinder_y("lining", r, -hy, hy, z=-z0)
    inner = cylinder_y("lining_in", r - d["lining_t"], -hy - 1, hy + 1, z=-z0)
    boolean(lining, inner)
    assign(lining, material("lining", MAL, 0.6))
    smooth(lining)

    # water table drawn down towards the drained tunnel, a translucent sheet in the cut-away
    wy, wt = _water_table(_book_path(d["csv"]))
    sheet = heightfield("water_table", [x for x in xs if x >= 0.0], [y for y in ys if y <= 0.0],
                        lambda x, y: _interp(x, wy, wt))
    sol = sheet.modifiers.new("solid", "SOLIDIFY")
    sol.thickness = 0.12
    apply_modifiers(sheet)
    assign(sheet, material("water_table", MAL, 0.35, alpha=0.32))
    smooth(sheet)

    # light-grey buildings on the settled surface, none in the cut-away quarter
    bmat = material("building", "#DCDCD8", 0.8)
    rows = [(8.0, 30.0, 1.0), (-36.0, -14.0, 0.8)]
    for y0, y1, hscale in rows:
        for x0, w, h in d["buildings"]:
            if y1 <= 0.0 and x0 + w > 0.0:
                continue
            lo = min(-settle(x) * ex for x in (x0, x0 + w / 2, x0 + w, max(x0, min(0.0, x0 + w))))
            assign(box("bldg", x0, x0 + w, y0, y1, lo - 0.5, lo + h * hscale), bmat)

    remove(cut, tunnel, inner)
    camera((150, -185, 95), TARGET, lens=40)
    lights((-0.55, 0.35, -0.7), strength=2.4, world=0.32)
