import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Tunnel Settlement Calculator", layout="centered")

# Title
st.title("Transverse Settlement Trough Calculator")

# Reference tables
with st.expander("🔍 Reference Tables: Volume Loss and k-values"):
    st.markdown(r"""
### Trough Width Parameter \(k\)

| Ground Type             | Typical \(k\) Range |
|------------------------|---------------------|
| Stiff Clay             | 0.3 – 0.5           |
| Soft Clay              | 0.5 – 0.7           |
| Loose/Medium Sand      | 0.4 – 0.6           |
| Dense Sand/Granular    | 0.25 – 0.4          |
| Rock                   | < 0.25              |

---

### Volume Loss \(V_L\) (% of Excavated Volume)

| Tunnel Method       | Ground Type         | \(V_L\) [%] Typical |
|---------------------|---------------------|---------------------|
| TBM (closed face)   | Clay/Sand           | 0.5 – 1.5           |
| TBM (open face)     | Rock/Coarse soils   | 0.5 – 2.0           |
| NATM / SEM          | Soft Ground         | 1.0 – 3.0           |
| Hand Excavation     | Very Soft Soils     | Up to 5%+           |
""")

# Input sliders
st.sidebar.header("🔧 Input Parameters")
V_L = st.sidebar.slider("Volume Loss (V_L %) ", 0.5, 5.0, 1.5, 0.1)
D = st.sidebar.slider("Tunnel Diameter (D) [m]", 3.0, 12.0, 6.0, 0.5)
z0 = st.sidebar.slider("Depth to Tunnel Axis (z₀) [m]", 5.0, 50.0, 20.0, 1.0)
k = st.sidebar.slider("Trough Width Parameter (k)", 0.2, 0.8, 0.5, 0.05)

# Compute settlement
i = k * z0
y = np.linspace(-3*i, 3*i, 400)

def gaussian_settlement(y, V_L_percent, D, z0, k):
    i = k * z0
    V_L = V_L_percent / 100.0
    coefficient = (V_L * np.pi * D**2) / (4 * i * np.sqrt(2 * np.pi))
    exponent = -y**2 / (2 * i**2)
    S_y_mm = -coefficient * np.exp(exponent) * 1000  # mm
    return S_y_mm

settlements = gaussian_settlement(y, V_L, D, z0, k)

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(y, settlements, 'b-', linewidth=2)

# Use a LaTeX-safe label
label_text = rf"Inflection Point\n$i = kz_0 = {i:.1f}\,\mathrm{{m}}$"
ax.axvline(i, color='r', linestyle='--', label=label_text)
ax.axvline(-i, color='r', linestyle='--')

ax.set_xlabel(r'Horizontal Distance from Centerline, $y$ (m)', labelpad=10)
ax.set_ylabel(r'Settlement, $S_y$ (mm)', labelpad=10)
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_title(r'Transverse Settlement Trough', pad=15)

st.pyplot(fig)


# Optional: show computed i value
st.markdown(f"**Computed Inflection Distance:** $i = k \\cdot z_0 = {k:.2f} \\times {z0:.1f} = {i:.2f}$ m")

