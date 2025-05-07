import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Streamlit page config
st.set_page_config(page_title="3D Tunnel Settlement Calculator", layout="centered")

# Title
st.title("3D Gaussian Settlement Trough for TBM Tunnelling")

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

# Sidebar inputs
st.sidebar.header("🔧 Input Parameters")
V_L = st.sidebar.slider("Volume Loss (V_L %) ", 0.5, 5.0, 1.5, 0.1)
D = st.sidebar.slider("Tunnel Diameter (D) [m]", 3.0, 12.0, 6.0, 0.5)
z0 = st.sidebar.slider("Depth to Tunnel Axis (z₀) [m]", 5.0, 50.0, 20.0, 1.0)
k = st.sidebar.slider("Trough Width Parameter (k)", 0.2, 0.8, 0.5, 0.05)
lambda_x = st.sidebar.slider("Longitudinal Decay (λ) [m]", 1.0, 3.0, 2.0, 0.1)

# Compute i, volume loss
i = k * z0
V_exc = (np.pi / 4) * D**2  # m³/m
V_loss = V_L / 100 * V_exc  # m³/m

# Settlement profile (transverse)
y = np.linspace(-3*i, 3*i, 400)

def gaussian_settlement(y, V_L_percent, D, z0, k):
    i = k * z0
    V_L = V_L_percent / 100.0
    coefficient = (V_L * np.pi * D**2) / (4 * i * np.sqrt(2 * np.pi))
    exponent = -y**2 / (2 * i**2)
    S_y_mm = -coefficient * np.exp(exponent) * 1000  # mm
    return S_y_mm

settlements = gaussian_settlement(y, V_L, D, z0, k)
S_max = np.min(settlements)

# Transverse plot
fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(y, settlements, 'b-', linewidth=2)
ax1.axvline(i, color='r', linestyle='--', label=rf"Inflection Point $i = {i:.1f}$ m")
ax1.axvline(-i, color='r', linestyle='--')
ax1.set_xlabel(r'Horizontal Distance from Centerline, $y$ (m)')
ax1.set_ylabel(r'Settlement, $S_y$ (mm)')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_title("Transverse Settlement Trough")
st.pyplot(fig1)

# Display volume info
st.markdown(f"**Computed Inflection Distance:** $i = k \\cdot z_0 = {k:.2f} \\times {z0:.1f} = {i:.2f}$ m")
st.markdown(f"**Excavated Volume per metre:** {V_exc:.2f} m³/m")
st.markdown(f"**Volume Loss:** {V_L:.1f}% → {V_loss:.3f} m³/m")

# 3D Gaussian settlement
x = np.linspace(-4*lambda_x, 4*lambda_x, 200)
X, Y = np.meshgrid(x, y)
S_3D = S_max * np.exp(-X**2 / (2 * lambda_x**2)) * np.exp(-Y**2 / (2 * i**2))

# 3D plot
fig2 = plt.figure(figsize=(10, 6))
ax2 = fig2.add_subplot(111, projection='3d')
surf = ax2.plot_surface(X, Y, S_3D, cmap='viridis', edgecolor='none')
ax2.set_title("3D Gaussian Settlement Trough")
ax2.set_xlabel("Distance Along Tunnel, x [m]")
ax2.set_ylabel("Transverse Distance, y [m]")
ax2.set_zlabel("Settlement S(x, y) [mm]")
st.pyplot(fig2)
