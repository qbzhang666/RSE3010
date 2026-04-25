
import math
import numpy as np
import pandas as pd

def acute_angle(a):
    """Return acute orientation difference in degrees, 0–180."""
    a = abs(a) % 360
    if a > 180:
        a = 360 - a
    return a

def circular_diff(a, b):
    return acute_angle(a - b)

def plane_normal(dip_dir_deg, dip_deg):
    """Plane normal in ENU coordinates from dip direction/dip."""
    a = math.radians(dip_dir_deg)
    d = math.radians(dip_deg)
    dip_vec = np.array([math.sin(a)*math.cos(d), math.cos(a)*math.cos(d), -math.sin(d)])
    strike_az = math.radians(dip_dir_deg - 90)
    strike_vec = np.array([math.sin(strike_az), math.cos(strike_az), 0.0])
    n = np.cross(strike_vec, dip_vec)
    return n / np.linalg.norm(n)

def trend_plunge(v):
    """Return trend/plunge of a line vector in ENU coordinates; lower hemisphere direction."""
    v = np.asarray(v, dtype=float)
    v = v / np.linalg.norm(v)
    if v[2] > 0:
        v = -v
    h = math.hypot(v[0], v[1])
    trend = (math.degrees(math.atan2(v[0], v[1])) + 360) % 360
    plunge = math.degrees(math.atan2(-v[2], h))
    return trend, plunge

def intersection_line(dd1, dip1, dd2, dip2):
    n1 = plane_normal(dd1, dip1)
    n2 = plane_normal(dd2, dip2)
    line = np.cross(n1, n2)
    return trend_plunge(line)

def smr_f1(A):
    A = abs(A)
    if A > 30: return 0.15
    if A > 20: return 0.40
    if A > 10: return 0.70
    if A > 5: return 0.85
    return 1.00

def smr_f2(B, mode):
    if mode.lower() == "toppling":
        return 1.00
    B = abs(B)
    if B < 20: return 0.15
    if B < 30: return 0.40
    if B < 35: return 0.70
    if B <= 45: return 0.85
    return 1.00

def smr_f3(C, mode):
    m = mode.lower()
    if m in ["planar", "wedge"]:
        if C > 10: return 0
        if C > 0: return -6
        if abs(C) < 1e-9: return -25
        if C >= -10: return -50
        return -60
    if m == "toppling":
        if C < 110: return 0
        if C <= 120: return -6
        return -25
    raise ValueError("mode must be planar, wedge, or toppling")

def smr_class(smr):
    if smr >= 81:
        return "I", "Very good", "Completely stable", "None", "None"
    if smr >= 61:
        return "II", "Good", "Stable", "Some blocks", "Spot bolting"
    if smr >= 41:
        return "III", "Fair", "Partially stable", "Joints / wedges", "Systematic bolting"
    if smr >= 21:
        return "IV", "Poor", "Unstable", "Planar / large wedges", "Corrective measures"
    return "V", "Very poor", "Completely unstable", "Large wedges / circular", "Re-excavation"

def equal_angle_project(trend_deg, plunge_deg):
    """Wulff net: equal-angle lower-hemisphere stereographic projection."""
    tr = math.radians(trend_deg)
    r = math.tan(math.radians(90 - plunge_deg) / 2)
    return r * math.sin(tr), r * math.cos(tr)

def great_circle_points(dip_dir_deg, dip_deg, npts=721):
    n = plane_normal(dip_dir_deg, dip_deg)
    ref = np.array([0, 0, 1.0])
    if abs(np.dot(ref, n)) > 0.95:
        ref = np.array([1.0, 0, 0])
    u = np.cross(n, ref); u /= np.linalg.norm(u)
    w = np.cross(n, u); w /= np.linalg.norm(w)
    xs, ys = [], []
    for t in np.linspace(0, 2*np.pi, npts):
        v = math.cos(t)*u + math.sin(t)*w
        if v[2] <= 1e-9:
            tr, pl = trend_plunge(v)
            x, y = equal_angle_project(tr, pl)
            xs.append(x); ys.append(y)
    return np.array(xs), np.array(ys)

def default_joint_dataframe():
    return pd.DataFrame({
        "Joint": ["J1", "J2", "J3"],
        "Dip β (deg)": [40.0, 55.0, 80.0],
        "Dip direction α (deg)": [160.0, 110.0, 340.0],
    })


import streamlit as st
import matplotlib.pyplot as plt
from itertools import combinations

st.set_page_config(page_title="Week 11 Kinematic Stereonet", layout="wide")

st.title("RSE3010 Week 11 — Kinematic Analysis with Wulff Net")
st.caption("Standalone teaching app: equal-angle stereonet for kinematic analysis — What can fail?")

with st.sidebar:
    st.header("Slope and friction inputs")
    slope_dip = st.number_input("Slope angle ψf / βs (deg)", min_value=0.0, max_value=90.0, value=60.0, step=1.0)
    slope_dd = st.number_input("Slope dip direction αs (deg)", min_value=0.0, max_value=360.0, value=170.0, step=1.0)
    dry_phi = st.number_input("Dry friction angle φdry (deg)", min_value=0.0, max_value=90.0, value=45.0, step=1.0)
    wet_phi = st.number_input("Wet friction angle φwet (deg)", min_value=0.0, max_value=90.0, value=28.0, step=1.0)
    tolerance = st.number_input("Directional tolerance (deg)", min_value=0.0, max_value=90.0, value=20.0, step=1.0)

st.subheader("Joint-set input")
jdf = st.data_editor(default_joint_dataframe(), num_rows="dynamic", use_container_width=True)
jdf = jdf.dropna()
jdf["Dip β (deg)"] = pd.to_numeric(jdf["Dip β (deg)"], errors="coerce")
jdf["Dip direction α (deg)"] = pd.to_numeric(jdf["Dip direction α (deg)"], errors="coerce")
jdf = jdf.dropna()

def planar_possible(alpha, beta, phi):
    return (circular_diff(alpha, slope_dd) <= tolerance) and (phi < beta < slope_dip)

def wedge_possible(trend, plunge, phi):
    return (circular_diff(trend, slope_dd) <= tolerance) and (phi < plunge < slope_dip)

def toppling_possible(alpha, beta, phi):
    return (circular_diff(alpha - slope_dd - 180, 0) <= tolerance) and (beta >= (90 - slope_dip + phi))

results = []
for _, row in jdf.iterrows():
    name, beta, alpha = row["Joint"], float(row["Dip β (deg)"]), float(row["Dip direction α (deg)"])
    results.append(["Planar", name, beta, alpha, "Dry", planar_possible(alpha, beta, dry_phi)])
    results.append(["Planar", name, beta, alpha, "Wet", planar_possible(alpha, beta, wet_phi)])
    results.append(["Toppling", name, beta, alpha, "Dry", toppling_possible(alpha, beta, dry_phi)])
    results.append(["Toppling", name, beta, alpha, "Wet", toppling_possible(alpha, beta, wet_phi)])

for (_, r1), (_, r2) in combinations(jdf.iterrows(), 2):
    tr, pl = intersection_line(float(r1["Dip direction α (deg)"]), float(r1["Dip β (deg)"]),
                               float(r2["Dip direction α (deg)"]), float(r2["Dip β (deg)"]))
    label = f"{r1['Joint']}–{r2['Joint']}"
    results.append(["Wedge", label, pl, tr, "Dry", wedge_possible(tr, pl, dry_phi)])
    results.append(["Wedge", label, pl, tr, "Wet", wedge_possible(tr, pl, wet_phi)])

res_df = pd.DataFrame(results, columns=["Mode", "Joint / pair", "Dip or plunge β (deg)", "Direction/trend α (deg)", "Condition", "Kinematically possible"])

col1, col2 = st.columns([1.15, 1])

with col1:
    st.subheader("Model stereonet — Wulff net (equal-angle)")
    fig, ax = plt.subplots(figsize=(7,7))
    theta = np.linspace(0, 2*np.pi, 720)
    ax.plot(np.sin(theta), np.cos(theta), color="black", linewidth=1.3)
    for az in range(0, 360, 30):
        ax.plot([0, np.sin(np.radians(az))], [0, np.cos(np.radians(az))], color="0.88", linewidth=0.5)
    for pl in range(10, 90, 10):
        r = np.tan(np.radians(90-pl)/2)
        ax.plot(r*np.sin(theta), r*np.cos(theta), color="0.92", linewidth=0.5)

    items = [("Slope", slope_dd, slope_dip)] + [(r["Joint"], float(r["Dip direction α (deg)"]), float(r["Dip β (deg)"])) for _, r in jdf.iterrows()]
    for label, dd, dip in items:
        x, y = great_circle_points(dd, dip)
        lw = 3.0 if label == "Slope" else 2.0
        ax.plot(x, y, linewidth=lw, label=f"{label} {dip:.0f}°/{dd:.0f}°")

    for phi, ls, label in [(dry_phi, "--", f"Dry φ={dry_phi:.0f}°"), (wet_phi, ":", f"Wet φ={wet_phi:.0f}°")]:
        r = np.tan(np.radians(90-phi)/2)
        ax.plot(r*np.sin(theta), r*np.cos(theta), linestyle=ls, color="black", linewidth=1.2, label=label)

    for (_, r1), (_, r2) in combinations(jdf.iterrows(), 2):
        tr, pl = intersection_line(float(r1["Dip direction α (deg)"]), float(r1["Dip β (deg)"]),
                                   float(r2["Dip direction α (deg)"]), float(r2["Dip β (deg)"]))
        x, y = equal_angle_project(tr, pl)
        ax.scatter([x], [y], s=55, color="yellow", edgecolor="black", zorder=5)
        ax.text(x+0.03, y+0.03, f"{r1['Joint']}-{r2['Joint']}\n{pl:.1f}/{tr:.1f}", fontsize=8)

    for txt, az in [("N",0),("E",90),("S",180),("W",270)]:
        ax.text(1.11*np.sin(np.radians(az)), 1.11*np.cos(np.radians(az)), txt,
                ha="center", va="center", fontsize=13, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.axis("off")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8)
    st.pyplot(fig)
    st.caption("Projection used: Wulff net / equal-angle lower-hemisphere stereographic projection. This is appropriate for kinematic analysis because angular relationships are preserved.")

with col2:
    st.subheader("Kinematic checks")
    st.dataframe(res_df, use_container_width=True)

    st.markdown("""
### Teaching interpretation
- This app uses a **Wulff net (equal-angle projection)**, not a Schmidt equal-area net.
- **Planar:** \( \\phi < \\beta_j < \\psi_f \) and joint direction close to slope direction.
- **Wedge:** \( \\phi < \\beta_i < \\psi_f \) and intersection trend exits through the slope.
- **Toppling:** joint dips steeply opposite the slope face and satisfies the slip-limit condition.
""")

csv = res_df.to_csv(index=False).encode("utf-8")
st.download_button("Download kinematic check table", csv, "week11_kinematic_checks.csv", "text/csv")
