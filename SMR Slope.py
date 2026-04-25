
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
    """Equal-angle lower-hemisphere stereographic projection."""
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
from itertools import combinations

st.set_page_config(page_title="Week 10 SMR Calculator", layout="wide")

st.title("RSE3010 Week 10 — Slope Mass Rating (SMR) Calculator")
st.caption("Standalone teaching app: empirical screening — What might fail?")

with st.sidebar:
    st.header("Slope and rock mass inputs")
    rmr = st.number_input("Basic RMR", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
    slope_dip = st.number_input("Slope angle ψf / βs (deg)", min_value=0.0, max_value=90.0, value=60.0, step=1.0)
    slope_dd = st.number_input("Slope dip direction αs (deg)", min_value=0.0, max_value=360.0, value=170.0, step=1.0)
    f4_choice = st.selectbox(
        "Excavation method F4",
        ["Natural slope (+15)", "Pre-splitting (+10)", "Smooth blasting (+8)", "Blasting / mechanical (0)", "Deficient blasting (-8)"],
        index=4
    )
    f4_map = {
        "Natural slope (+15)": 15,
        "Pre-splitting (+10)": 10,
        "Smooth blasting (+8)": 8,
        "Blasting / mechanical (0)": 0,
        "Deficient blasting (-8)": -8,
    }
    f4 = f4_map[f4_choice]

st.subheader("Joint-set input")
st.write("Use **dip / dip direction** format. Week 10 notation: α = dip direction, β = dip or plunge.")
jdf = st.data_editor(default_joint_dataframe(), num_rows="dynamic", use_container_width=True)

jdf = jdf.dropna()
jdf["Dip β (deg)"] = pd.to_numeric(jdf["Dip β (deg)"], errors="coerce")
jdf["Dip direction α (deg)"] = pd.to_numeric(jdf["Dip direction α (deg)"], errors="coerce")
jdf = jdf.dropna()

rows = []

for _, row in jdf.iterrows():
    name = row["Joint"]
    beta = float(row["Dip β (deg)"])
    alpha = float(row["Dip direction α (deg)"])

    # Planar
    A = circular_diff(alpha, slope_dd)
    B = abs(beta)
    C = beta - slope_dip
    F1, F2, F3 = smr_f1(A), smr_f2(B, "planar"), smr_f3(C, "planar")
    prod = F1 * F2 * F3
    smr = rmr + prod + f4
    cls, quality, stability, mode_text, support = smr_class(smr)
    rows.append(["Planar", name, A, B, C, F1, F2, F3, prod, f4, rmr, smr, cls, quality, stability, mode_text, support])

    # Toppling
    A = circular_diff(alpha - slope_dd - 180, 0)
    B = None
    C = beta + slope_dip
    F1, F2, F3 = smr_f1(A), 1.0, smr_f3(C, "toppling")
    prod = F1 * F2 * F3
    smr = rmr + prod + f4
    cls, quality, stability, mode_text, support = smr_class(smr)
    rows.append(["Toppling", name, A, B, C, F1, F2, F3, prod, f4, rmr, smr, cls, quality, stability, mode_text, support])

for (_, r1), (_, r2) in combinations(jdf.iterrows(), 2):
    n1, n2 = r1["Joint"], r2["Joint"]
    trend, plunge = intersection_line(float(r1["Dip direction α (deg)"]), float(r1["Dip β (deg)"]),
                                      float(r2["Dip direction α (deg)"]), float(r2["Dip β (deg)"]))
    label = f"{n1}–{n2} ({plunge:.1f}°/{trend:.1f}°)"
    A = circular_diff(trend, slope_dd)
    B = abs(plunge)
    C = plunge - slope_dip
    F1, F2, F3 = smr_f1(A), smr_f2(B, "wedge"), smr_f3(C, "wedge")
    prod = F1 * F2 * F3
    smr = rmr + prod + f4
    cls, quality, stability, mode_text, support = smr_class(smr)
    rows.append(["Wedge", label, A, B, C, F1, F2, F3, prod, f4, rmr, smr, cls, quality, stability, mode_text, support])

out = pd.DataFrame(rows, columns=[
    "Failure type", "Joint / intersection", "A (deg)", "B (deg)", "C (deg)",
    "F1", "F2", "F3", "F1×F2×F3", "F4", "RMR", "SMR",
    "Class", "Quality", "Stability", "Failure mode", "Support"
])

st.subheader("SMR results")
st.dataframe(out.style.format({
    "A (deg)": "{:.1f}", "B (deg)": "{:.1f}", "C (deg)": "{:.1f}",
    "F1": "{:.2f}", "F2": "{:.2f}", "F3": "{:.0f}",
    "F1×F2×F3": "{:.2f}", "SMR": "{:.1f}"
}), use_container_width=True)

st.subheader("Teaching interpretation")
critical = out.sort_values("SMR").iloc[0]
st.info(f"Lowest SMR = **{critical['SMR']:.1f}**, for **{critical['Failure type']} — {critical['Joint / intersection']}**. "
        f"Class {critical['Class']} ({critical['Quality']}), stability: {critical['Stability']}.")

st.markdown("""
### Common checks for students
- Do not use α as dip angle. In Week 10, **α = dip direction**, **β = dip or plunge**.
- For toppling, use **A = |αj − αs − 180°|**, **C = βj + βs**, and **F2 = 1.0**.
- For wedge failure, use the **intersection line trend/plunge** as **αi / βi**.
""")

csv = out.to_csv(index=False).encode("utf-8")
st.download_button("Download SMR table as CSV", csv, "week10_smr_results.csv", "text/csv")
