import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def plot_arc_beam_tbm_rect(R, theta_deg, w, E_GPa, b_mm, h_mm):
    # Convert dimensions to meters
    b = b_mm / 1000
    h = h_mm / 1000

    # Moment of inertia for rectangular cross-section (per 1 m width)
    I = (b * h**3) / 12
    I_mm4 = I * 1e12

    # Arc properties
    theta = np.radians(theta_deg)
    L = R * theta
    s = np.linspace(0, L, 500)

    # Arc coordinates
    angle = np.linspace(0, theta, 500)
    x_coords = R * (1 - np.cos(angle))     # horizontal
    y_coords = R * np.sin(angle)           # vertical

    # Shear force and moment
    V = w * (L / 2 - s)
    M = w * s * (L - s) / 2

    # Deflection
    E = E_GPa * 1e9
    w_Nm = w * 1e3
    deflection_m = (w_Nm * s * (L**3 - 2 * L * s**2 + s**3)) / (24 * E * I)
    deflection_mm = deflection_m * 1000

    # Plotting
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=False)

    # 1. Arc and UDL
    axs[0].plot(x_coords, y_coords, 'k', lw=6)
    axs[0].plot([x_coords[0]], [y_coords[0]], 'g^', markersize=15)
    axs[0].plot([x_coords[-1]], [y_coords[-1]], 'go', markersize=15)
    for i in np.linspace(0.05, 0.95, 8):
        idx = int(i * len(x_coords))
        axs[0].arrow(x_coords[idx], y_coords[idx] + 0.1, 0, -0.08,
                     head_width=0.05, head_length=0.02, color='blue')
    axs[0].text(x_coords[len(x_coords)//2] - 0.5, y_coords[len(y_coords)//2] + 0.25,
                f'w = {w:.1f} kN/m', fontsize=12)
    axs[0].axis('equal')
    axs[0].set_title("TBM Tunnel Segment (Arc) with UDL", fontsize=14, pad=30)
    axs[0].axis('off')

    # 2. Shear force
    axs[1].plot(s, V, 'b')
    axs[1].axhline(0, color='black', lw=0.5)
    axs[1].set_ylabel("Shear (kN)")
    axs[1].set_title("Shear Force Diagram")

    # 3. Bending moment
    axs[2].plot(s, M, 'r')
    axs[2].axhline(0, color='black', lw=0.5)
    axs[2].set_ylabel("Moment (kN·m)")
    axs[2].set_title("Bending Moment Diagram")

    # 4. Deflection
    axs[3].plot(s, deflection_mm, 'purple')
    axs[3].axhline(0, color='black', lw=0.5)
    axs[3].set_ylabel("Deflection (mm)")
    axs[3].set_xlabel("Arc Length (m)")
    axs[3].set_title("Deformation Shape")

    plt.tight_layout()
    st.pyplot(fig)

    # Output computed I
    st.markdown("### TBM Segment Moment of Inertia (Rectangular Approximation)")
    st.markdown(f"**Width (b):** {b_mm:.1f} mm")
    st.markdown(f"**Height (h):** {h_mm:.1f} mm")
    st.latex(r"I = \frac{b h^3}{12}")
    st.markdown(f"**Computed I:** {I:.4e} m⁴ = {I_mm4:.2f} mm⁴")


# Streamlit App Layout
st.title("TBM Tunnel Segment: Arc Beam Analysis")

st.sidebar.header("Input Parameters")

R = st.sidebar.slider("Tunnel Radius (m)", 2.5, 5.0, 3.6, 0.1)
theta_deg = st.sidebar.slider("Arc Angle (°)", 30, 180, 180, 5)
w = st.sidebar.slider("Uniformly Distributed Load UDL (kN/m)", 1.0, 50.0, 20.0, 1.0)
E_GPa = st.sidebar.slider("Elastic Modulus E (GPa)", 5.0, 100.0, 100.0, 1.0)
b_mm = st.sidebar.slider("Segment Length b (mm)", 500, 2000, 1000, 50)
h_mm = st.sidebar.slider("Segment Thickness h (mm)", 200, 500, 200, 10)

plot_arc_beam_tbm_rect(R, theta_deg, w, E_GPa, b_mm, h_mm)

st.markdown("---")
st.markdown("📘 **For technical design details, please refer to specific guidelines such as _Guidelines for the Design of Segmental Tunnel Linings_ (ITA WG2 Report).**")

