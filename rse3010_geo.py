"""RSE3010 Geotechnical Platform – the companion application of the book
*Underground Design and Construction*.

One Streamlit application, one sidebar, one shared ground model, organised
along the design chain of the book: ground model -> intact rock -> rock mass
-> convergence–confinement -> ground movement -> lining, with construction
vibration and the rock-slope tools of the course alongside.

Run:     streamlit run rse3010_geo.py
Deploy:  Streamlit Community Cloud, entry point rse3010_geo.py, Python 3.12.

Calculations live in the ``geo_platform`` package (no Streamlit inside), so
they can be tested on their own; this file draws the pages.
"""

from __future__ import annotations

import datetime
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from geo_platform import ccm, figures, ground, lining, report, rock, settlement, style, ui

APP_NAME = "RSE3010 Geotechnical Platform"
PLATFORM_VERSION = "1.0"
BOOK_TITLE = "Underground Design and Construction"
REPO_URL = "https://github.com/qbzhang666/RSE3010"

PAGE_KEY = "page"
PAGES: dict[str, str] = {
    "overview": "Overview",
    "ground": "Ground model",
    "intact": "Intact rock strength",
    "rockmass": "Rock mass parameters",
    "ccm": "Convergence–confinement",
    "settlement": "Ground movement",
    "lining": "Segmental lining",
    "vibration": "Construction vibration",
    "smr": "Slope mass rating",
    "kinematic": "Kinematics and planar sliding",
    "slope_lem": "Wedge and circular failure",
    "equations": "Equations and references",
    "report": "Design report",
}
NAV_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Start", ("overview", "ground")),
    ("Ground and parameters", ("intact", "rockmass")),
    ("Underground design", ("ccm", "settlement", "lining")),
    ("Construction", ("vibration",)),
    ("Rock slopes", ("smr", "kinematic", "slope_lem")),
    ("Reference and report", ("equations", "report")),
)
ALL_PAGES = tuple(p for _, group in NAV_GROUPS for p in group)

# Widget keys of these pages are re-written at the start of every run, so their
# values survive a visit to another page (Streamlit removes the state of a
# widget that is not drawn in a run).
PERSIST_PREFIXES = ("g_", "int_", "rm_", "ccm_", "set_", "lin_", "vib_", "smr_", "kin_", "lem_", "rep_")
# Buttons, download buttons and table editors cannot have their state set.
NOT_SETTABLE = ("btn_", "_ed", "_dl", "smr_joints", "smr_faces", "kin_joints", "vib_receivers")


# ---------------------------------------------------------------------------
# State and navigation
# ---------------------------------------------------------------------------


def init_state() -> None:
    state = st.session_state
    if "g_sigma_ci" not in state:
        ground.load_preset(state, ground.DEFAULT_PRESET)
    for key in list(state.keys()):
        if key.startswith(PERSIST_PREFIXES) and not any(tag in key for tag in NOT_SETTABLE):
            state[key] = state[key]
    if state.get(PAGE_KEY) not in ALL_PAGES:
        state[PAGE_KEY] = "overview"


def go_to_page(page_id: str) -> None:
    st.session_state[PAGE_KEY] = page_id if page_id in ALL_PAGES else "overview"


def _signature(gm: ground.GroundModel) -> tuple:
    """The ground-model values a design page depends on, to tell when its report section is out of date."""
    return tuple(round(float(getattr(gm, k)), 9) for k in ground.KEYS) + (gm.phi_adopted, gm.c_adopted, gm.em_adopted)


def _fit_labels(m1: rock.MCFit, m2: rock.MCFit) -> list:
    return [("Method 1, Mohr circles", m1), (r"Method 2, $\sigma_1$ against $\sigma_3$", m2)]


def sidebar(gm: ground.GroundModel) -> str:
    page = st.session_state[PAGE_KEY]
    with st.sidebar:
        st.title(APP_NAME)
        st.caption(f"Ground model: {gm.preset}" if gm.preset else "Ground model: user-defined")
        for title, group in NAV_GROUPS:
            st.markdown(f"**{title}**")
            for page_id in group:
                st.button(PAGES[page_id], key=f"btn_nav_{page_id}", on_click=go_to_page, args=(page_id,),
                          type="primary" if page_id == page else "secondary", width="stretch")
        st.caption(f"Version {PLATFORM_VERSION}. Companion to the book *{BOOK_TITLE}*. "
                   f"Source and licence: [{REPO_URL.split('//')[1]}]({REPO_URL}).")
    return page


def app_style() -> None:
    st.markdown(
        f"""
        <style>
        h1 {{ color: {style.MALACHITE}; }}
        h2, h3 {{ color: {style.INK}; }}
        div[data-testid="stMetric"] {{
            background: {style.WHITE}; border: 1px solid {style.GREY_LIGHT};
            border-left: 3px solid {style.MALACHITE}; border-radius: 6px; padding: 0.5rem 0.75rem;
        }}
        section[data-testid="stSidebar"] .stButton button {{ justify-content: flex-start; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------


def page_overview(gm: ground.GroundModel) -> None:
    ui.page_header(
        APP_NAME,
        f"companion to *{BOOK_TITLE}*; RSE3010 Mine Geotechnical Engineering",
        "Calculations of the book that are tedious by hand and instructive when varied, in one application. "
        "Set the ground once on the **Ground model** page; the parameter and design pages read it, so a change "
        "in GSI is followed through to the rock mass strength, the plastic zone and the support load.",
    )
    st.subheader("How the pages connect")
    st.markdown(
        "1. **Ground model** – intact strength, $m_i$, GSI, disturbance, depth and stress ratio, or a data set of the book.\n"
        "2. **Intact rock strength** (Chapter 11) – Hoek–Brown and Mohr–Coulomb for intact rock, tensile strength, "
        "and the equivalent Mohr–Coulomb line over a stated range.\n"
        "3. **Rock mass parameters** (Chapter 12) – generalised Hoek–Brown, rock mass strength, five modulus relations.\n"
        "4. **Convergence–confinement** (Chapter 16) – ground reaction, longitudinal profile, support and equilibrium.\n"
        "5. **Ground movement** (Chapters 17 and 26) – settlement trough, its development with the face, horizontal "
        "strain and the first-stage screening of buildings.\n"
        "6. **Segmental lining** (Chapter 19) – a segment as a curved member under a transient load, and the "
        "compressibility and flexibility of a ring in the ground.\n"
        "7. **Construction vibration** (Chapters 17, 22 and 26) and the **rock-slope** tools of the course."
    )
    st.subheader("Where each page sits in the book")
    table = pd.DataFrame([
        ("Ground model", "Chapters 8, 10-12", "In-situ stress; RMR, Q and GSI; the four-rock data set"),
        ("Intact rock strength", "Chapter 11", "Hoek–Brown, Mohr–Coulomb, Griffith and tension"),
        ("Rock mass parameters", "Chapter 12", "Generalised Hoek–Brown, modulus, equivalent Mohr–Coulomb"),
        ("Convergence–confinement", "Chapter 16; Appendix F", "Ground reaction, LDP, support, factor of safety"),
        ("Ground movement", "Chapters 17 and 26; Appendix H", "Gaussian trough, volume loss, screening"),
        ("Segmental lining", "Chapter 19; Appendix F", "Transient load cases; ring stiffness ratios"),
        ("Construction vibration", "Chapters 17, 22 and 26", "Screening of a drive beneath buildings"),
        ("Rock slopes", "Chapter 9; course Weeks 10-12", "SMR, kinematic screening, limit equilibrium"),
    ], columns=["Page", "Book", "Content"])
    st.dataframe(table, hide_index=True, width="stretch")
    st.info(
        "This is a teaching and screening tool. It contains no judgement and returns an answer whether or not "
        "its assumptions apply; every page states those assumptions. Results are for checking and for learning "
        "how a design responds to its inputs, not for design without independent verification."
    )


# ---------------------------------------------------------------------------
# Ground model
# ---------------------------------------------------------------------------


def _apply_preset() -> None:
    ground.load_preset(st.session_state, st.session_state["set_preset"])


def page_ground(gm: ground.GroundModel) -> None:
    ui.page_header(
        "Ground model", "Chapters 8 and 10-12",
        "One set of ground and geometry inputs for the whole platform. Load a data set of the book or enter "
        "your own; the parameter and design pages use these values.",
    )
    labels = list(ground.PRESETS)
    if "set_preset" not in st.session_state:
        st.session_state["set_preset"] = gm.preset if gm.preset in labels else ground.DEFAULT_PRESET
    col_a, col_b = st.columns([3, 1])
    col_a.selectbox("Data set", labels, key="set_preset")
    col_b.button("Load data set", key="btn_load_preset", on_click=_apply_preset, width="stretch")
    if gm.note:
        st.caption(gm.note)

    st.subheader("Intact rock and rock mass")
    c1, c2, c3, c4 = st.columns(4)
    c1.number_input("Intact strength, σ_ci (MPa)", min_value=0.5, max_value=400.0, step=1.0, key="g_sigma_ci")
    c2.number_input("Hoek–Brown constant, m_i (-)", min_value=1.0, max_value=40.0, step=0.5, key="g_mi")
    c3.number_input("Intact modulus, E_i (GPa)", min_value=0.05, max_value=120.0, step=0.5, key="g_ei")
    c4.number_input("Poisson's ratio, ν (-)", min_value=0.05, max_value=0.49, step=0.01, key="g_nu")
    c1, c2, c3, c4 = st.columns(4)
    c1.number_input("Geological Strength Index, GSI (-)", min_value=5.0, max_value=100.0, step=1.0, key="g_gsi")
    c2.number_input("Disturbance factor, D (-)", min_value=0.0, max_value=1.0, step=0.1, key="g_d")
    c3.number_input("Unit weight, γ (kN/m³)", min_value=15.0, max_value=32.0, step=0.5, key="g_gamma")
    with c4:
        with st.popover("GSI from a rating"):
            st.session_state.setdefault("g_tool_rmr", 68.0)
            st.session_state.setdefault("g_tool_q", 10.0)
            rmr = st.number_input("Dry RMR'89 without orientation adjustment", 23.0, 100.0, key="g_tool_rmr")
            st.write(f"GSI = RMR'89 − 5 = **{rock.gsi_from_rmr89_dry(rmr):.0f}**")
            qp = st.number_input("Modified quality index, Q' = (RQD/J_n)(J_r/J_a)", 0.01, 400.0, key="g_tool_q")
            st.write(f"GSI = 9 ln Q' + 44 = **{rock.gsi_from_q_prime(qp):.0f}**")
        with st.popover("Typical m_i"):
            st.dataframe(pd.DataFrame(rock.MI_TYPICAL.items(), columns=["Rock type", "m_i"]),
                         hide_index=True, height=320)

    st.subheader("Depth, stress and opening")
    c1, c2, c3 = st.columns(3)
    c1.number_input("Depth to tunnel axis, z (m)", min_value=1.0, max_value=3000.0, step=10.0, key="g_depth")
    c2.number_input("In-situ stress ratio, k = σ_h/σ_v (-)", min_value=0.1, max_value=5.0, step=0.1, key="g_k")
    c3.number_input("Tunnel radius, a (m)", min_value=0.5, max_value=15.0, step=0.25, key="g_radius")

    gm = ground.from_state(st.session_state)
    st.subheader("Derived quantities")
    ins = gm.insitu
    ui.metrics([
        ("σ_v = γz", ui.fmt(ins.sigma_v, 2, "MPa")),
        ("σ_h,k = kσ_v", ui.fmt(ins.sigma_h, 2, "MPa")),
        ("σ₁ in situ", ui.fmt(ins.sigma_1, 2, "MPa")),
        ("σ₃ in situ", ui.fmt(ins.sigma_3, 2, "MPa")),
        ("m_b", ui.fmt(gm.hb.mb, 3)),
        ("s", f"{gm.hb.s:.5f}"),
        ("a_HB", ui.fmt(gm.hb.a, 3)),
        ("σ_cm = σ_ci sᵃ", ui.fmt(gm.sigma_cm, 2, "MPa")),
    ])
    if gm.phi_adopted is not None:
        st.caption(
            f"Adopted in the book's worked example: φ = {gm.phi_adopted:g}°, c = {gm.c_adopted:.3g} MPa"
            + (f", E_m = {gm.em_adopted * 1000:.0f} MPa" if gm.em_adopted else "")
            + ". The convergence–confinement page offers these as the default source. "
            "Editing the inputs above does not change them."
        )
    ui.assumptions([
        "The in-situ stress is described by the vertical stress γz and one horizontal stress kσ_v (Chapter 11).",
        "The convergence–confinement page uses a hydrostatic field p₀ = γz.",
        "GSI from a rating uses the correlations of Chapter 10; the chart estimate is preferred in poor ground.",
    ])


# ---------------------------------------------------------------------------
# Intact rock (Chapter 11)
# ---------------------------------------------------------------------------


def _envelope_figure(s3_line, s1_hb, fits, s3_pts, s1_pts, title_hb: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s3_line, y=s1_hb, name=title_hb, mode="lines",
                             line=dict(color=style.MALACHITE, width=2.2)))
    for idx, (label, fit) in enumerate(fits):
        st_ = style.series_style(idx + 1, "plotly")
        fig.add_trace(go.Scatter(x=s3_line, y=fit.intercept + fit.slope * s3_line if label.startswith("Method 2")
                                 else rock.mc_sigma1(s3_line, fit.c, fit.phi_deg),
                                 name=label, mode="lines", line=dict(color=st_["color"], dash=st_["dash"])))
    fig.add_trace(go.Scatter(x=s3_pts, y=s1_pts, name="Points used in the fit", mode="markers",
                             marker=dict(color=style.INK, size=8, symbol="circle-open")))
    fig.update_layout(style.plotly_layout(
        xaxis_title=style.axis_label("minor principal stress", "σ₃", "MPa"),
        yaxis_title=style.axis_label("major principal stress at failure", "σ₁", "MPa"),
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0)")))
    return fig


def _mohr_figure(s3_pts, s1_pts, fit: rock.MCFit, tension: float):
    with plt.rc_context(style.matplotlib_rcparams()):
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        theta = np.linspace(0, np.pi, 200)
        for s3, s1 in zip(s3_pts, s1_pts):
            c, r = (s1 + s3) / 2, (s1 - s3) / 2
            ax.plot(c + r * np.cos(theta), r * np.sin(theta), color=style.GREY_DARK, lw=0.9)
        smax = float(max(s1_pts)) * 1.05
        sn = np.linspace(min(tension, 0) * 1.1, smax, 50)
        ax.plot(sn, fit.c + sn * math.tan(math.radians(fit.phi_deg)), color=style.MALACHITE, lw=1.6,
                label=rf"$\tau = c + \sigma_n\tan\phi$, $c$ = {fit.c:.2f} MPa, $\phi$ = {fit.phi_deg:.1f}$^\circ$")
        ax.axvline(tension, **{k: v for k, v in style.reference_style().items() if k != "marker"}, lw=1.0,
                   label=f"Hoek–Brown tensile strength {tension:.2f} MPa")
        ax.set_xlabel(style.axis_label("normal stress", r"\sigma_n", "MPa", math=True))
        ax.set_ylabel(style.axis_label("shear stress", r"\tau", "MPa", math=True))
        ax.set_ylim(bottom=0)
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(loc="upper left")
    return fig


def page_intact(gm: ground.GroundModel) -> None:
    ui.page_header(
        "Intact rock strength", "Chapter 11",
        "The intact Hoek–Brown criterion, the tensile strength it implies, and the equivalent Mohr–Coulomb "
        "line fitted over a stated range of confinement by the two methods of the chapter.",
    )
    ins = gm.insitu
    st.markdown(f"Rock from the ground model: **σ_ci = {gm.sigma_ci:g} MPa, m_i = {gm.mi:g}**; "
                f"in-situ σ₃ = {ins.sigma_3:.2f} MPa, σ₁ = {ins.sigma_1:.2f} MPa.")
    c1, c2, c3 = st.columns(3)
    if "int_s3max" not in st.session_state:
        st.session_state["int_s3max"] = round(ins.sigma3_fit_max, 2)
    s3max = c1.number_input("Upper limit of the fitting range, σ₃,fit,max (MPa)", 0.1, 500.0, step=0.5,
                            key="int_s3max", help="Chapter 11 takes max(σ_v, σ_h,k).")
    st.session_state.setdefault("int_npts", 6)
    npts = c2.number_input("Points in the fit", 3, 50, key="int_npts")
    c3.button("Reset range to max(σ_v, σ_h,k)", key="btn_int_reset",
              on_click=lambda: st.session_state.update(int_s3max=round(ground.from_state(st.session_state).insitu.sigma3_fit_max, 2)))

    s3_pts, s1_pts = rock.hb_points(gm.sigma_ci, gm.mi, 1.0, 0.5, s3max, int(npts))
    m1 = rock.fit_mohr_circles(s3_pts, s1_pts)
    m2 = rock.fit_principal_line(s3_pts, s1_pts)
    t_hb = rock.hb_intact_tension(gm.sigma_ci, gm.mi)
    t_hm = rock.hoek_martin_tension(gm.sigma_ci, gm.mi)
    s1_is = float(rock.hb_sigma1(ins.sigma_3, gm.sigma_ci, gm.mi))

    ui.metrics([
        ("σ₁ᶠ at in-situ σ₃", ui.fmt(s1_is, 1, "MPa"), "Intact Hoek–Brown strength at the minor in-situ stress"),
        ("Capacity / demand", ui.fmt(s1_is / ins.sigma_1, 2), "σ₁ᶠ / σ₁ in situ"),
        ("Tension, Hoek–Brown", ui.fmt(t_hb, 2, "MPa")),
        ("Tension, Hoek–Martin", ui.fmt(t_hm, 2, "MPa")),
        ("Method 1: φ / c", f"{m1.phi_deg:.1f}° / {m1.c:.1f}", "Regression of R on C; c in MPa"),
        ("Method 2: φ / c", f"{m2.phi_deg:.1f}° / {m2.c:.1f}", "Regression of σ₁ on σ₃; c in MPa"),
        ("Implied UCS of the line", ui.fmt(m2.intercept, 1, "MPa")),
        ("Implied tension of the line", ui.fmt(rock.mc_tension(m2.c, m2.phi_deg), 1, "MPa")),
    ])
    s3_line = np.linspace(0, max(s3max * 1.25, ins.sigma_1 * 0.2), 200)
    tab1, tab2, tab3 = st.tabs(["Envelope", "Mohr circles", "Fitted points"])
    with tab1:
        ui.show_plotly(_envelope_figure(s3_line, rock.hb_sigma1(s3_line, gm.sigma_ci, gm.mi), [
            ("Method 1, Mohr circles", m1), ("Method 2, σ₁ against σ₃", m2)], s3_pts, s1_pts, "Intact Hoek–Brown"))
    with tab2:
        ui.show_matplotlib(_mohr_figure(s3_pts, s1_pts, m1, t_hb))
    with tab3:
        df = pd.DataFrame({"σ₃ (MPa)": s3_pts, "σ₁ᶠ (MPa)": s1_pts,
                           "C (MPa)": (s1_pts + s3_pts) / 2, "R (MPa)": (s1_pts - s3_pts) / 2}).round(2)
        st.dataframe(df, hide_index=True, width="stretch")
        ui.download_table(df, "intact_hb_points.csv", key="dl_int_points")

    with st.expander("Fit Mohr–Coulomb to your own triaxial results"):
        data = st.data_editor(pd.DataFrame({"σ₃ (MPa)": [0.0, 5.0, 10.0, 20.0], "σ₁ (MPa)": [110.0, 160.0, 195.0, 255.0]}),
                              num_rows="dynamic", key="int_tri_ed", width="stretch")
        clean = data.dropna()
        if len(clean) >= 2:
            f1 = rock.fit_mohr_circles(clean.iloc[:, 0], clean.iloc[:, 1])
            f2 = rock.fit_principal_line(clean.iloc[:, 0], clean.iloc[:, 1])
            ui.metrics([("Method 1: φ / c (MPa)", f"{f1.phi_deg:.1f}° / {f1.c:.2f}"),
                        ("Method 2: φ / c (MPa)", f"{f2.phi_deg:.1f}° / {f2.c:.2f}"),
                        ("R² (method 2)", ui.fmt(f2.r2, 3))], columns=3)
    notes = [
        "Intact Hoek–Brown with s = 1 and a = 0.5; compression positive.",
        "The equivalent Mohr–Coulomb parameters belong to the fitting range, not to the rock; a narrower range "
        "gives a higher φ and a lower c.",
        "A straight line cannot follow the curved envelope at both ends: the tension it implies is usually far "
        "above the measured value.",
    ]
    ui.assumptions(notes)
    n = report.num
    report.record(st.session_state, report.Section(
        key="intact", title="Intact rock strength", chapters="Chapter 11",
        purpose="The intact Hoek–Brown criterion, the tensile strength it implies, and the equivalent "
                "Mohr–Coulomb line over the stated range of confinement, fitted by the two methods of the chapter.",
        inputs=[
            ("Uniaxial compressive strength of intact rock", r"\sigma_{ci}", n(gm.sigma_ci, 1), "MPa"),
            ("Hoek–Brown constant of intact rock", "m_i", n(gm.mi, 1), "-"),
            ("Minor principal stress in situ", r"\sigma_3", n(ins.sigma_3, 2), "MPa"),
            ("Major principal stress in situ", r"\sigma_1", n(ins.sigma_1, 2), "MPa"),
            ("Upper limit of the fitting range", r"\sigma_{3,max}", n(s3max, 2), "MPa"),
            ("Points in the fit", "n", f"{int(npts)}", "-"),
        ],
        equations=[EQUATIONS["rock"][0], EQUATIONS["rock"][2], EQUATIONS["rock"][4]],
        results=[
            ("Strength at the in-situ minor stress", r"\sigma_1^f", n(s1_is, 1), "MPa"),
            ("Capacity / demand", r"\sigma_1^f/\sigma_1", n(s1_is / ins.sigma_1, 2), "-"),
            ("Tensile strength, Hoek–Brown", r"\sigma_t", n(t_hb, 2), "MPa"),
            ("Tensile strength, Hoek and Martin (2014)", r"\sigma_t", n(t_hm, 2), "MPa"),
            ("Method 1 (R on C): friction angle", r"\phi", n(m1.phi_deg, 1), "°"),
            ("Method 1: cohesion", "c", n(m1.c, 2), "MPa"),
            ("Method 2 (σ₁ on σ₃): friction angle", r"\phi", n(m2.phi_deg, 1), "°"),
            ("Method 2: cohesion", "c", n(m2.c, 2), "MPa"),
            ("Uniaxial strength implied by the line", r"\sigma_{c,MC}", n(m2.intercept, 1), "MPa"),
            ("Tensile strength implied by the line", r"\sigma_{t,MC}", n(rock.mc_tension(m2.c, m2.phi_deg), 1), "MPa"),
        ],
        checks=[(f"The intact strength at the in-situ minor stress exceeds the major in-situ stress "
                 f"(capacity / demand {s1_is / ins.sigma_1:.2f}).", bool(s1_is >= ins.sigma_1))],
        figures=[
            ("envelope", "Intact Hoek–Brown envelope, the fitting points and the two equivalent Mohr–Coulomb lines.",
             lambda: figures.envelope(gm.sigma_ci, gm.mi, 1.0, 0.5, s3max, _fit_labels(m1, m2), s3_pts, s1_pts,
                                      "Intact Hoek–Brown")),
            ("mohr", "Mohr circles of the fitting points with the Method 1 line.",
             lambda: _mohr_figure(s3_pts, s1_pts, m1, t_hb)),
        ],
        assumptions=notes, signature=_signature(gm)))


# ---------------------------------------------------------------------------
# Rock mass (Chapter 12)
# ---------------------------------------------------------------------------


def page_rockmass(gm: ground.GroundModel) -> None:
    ui.page_header(
        "Rock mass parameters", "Chapter 12",
        "The generalised Hoek–Brown criterion from GSI, m_i and D: strength at in-situ confinement, tensile "
        "strength, five estimates of the deformation modulus, and the equivalent Mohr–Coulomb parameters.",
    )
    ins = gm.insitu
    c1, c2, c3 = st.columns(3)
    if "rm_s3max" not in st.session_state:
        st.session_state["rm_s3max"] = round(ins.sigma3_fit_max, 2)
    s3max = c1.number_input("Upper limit of the fitting range, σ₃,fit,max (MPa)", 0.1, 500.0, step=0.5, key="rm_s3max")
    st.session_state.setdefault("rm_npts", 6)
    npts = c2.number_input("Points in the fit", 3, 50, key="rm_npts")
    if "rm_rmr" not in st.session_state:
        st.session_state["rm_rmr"] = float(gm.gsi + 5)
    rmr = c3.number_input("RMR (for the two RMR-based modulus relations)", 0.0, 100.0, step=1.0, key="rm_rmr",
                          help="Defaults to GSI + 5.")

    hb = gm.hb
    s1m = float(rock.hb_sigma1(ins.sigma_3, gm.sigma_ci, hb.mb, hb.s, hb.a))
    ui.metrics([
        ("m_b", ui.fmt(hb.mb, 3)), ("s", f"{hb.s:.5f}"), ("a_HB", ui.fmt(hb.a, 3)),
        ("σ_cm = σ_ci sᵃ", ui.fmt(gm.sigma_cm, 2, "MPa")),
        ("σ₁ₘᶠ at in-situ σ₃", ui.fmt(s1m, 1, "MPa")),
        ("Capacity / demand", ui.fmt(s1m / ins.sigma_1, 2), "σ₁ₘᶠ / σ₁ in situ"),
        ("σ_tm = −sσ_ci/m_b", ui.fmt(rock.rock_mass_tension(gm.sigma_ci, hb), 3, "MPa")),
        ("Global strength σ'_cm", ui.fmt(rock.global_strength(gm.sigma_ci, hb), 2, "MPa"),
         "Hoek, Carranza-Torres and Corkum (2002)"),
    ])
    s3_pts, s1_pts = rock.hb_points(gm.sigma_ci, hb.mb, hb.s, hb.a, s3max, int(npts))
    m1 = rock.fit_mohr_circles(s3_pts, s1_pts)
    m2 = rock.fit_principal_line(s3_pts, s1_pts)
    phi02, c02 = rock.hoek_2002_mc(gm.sigma_ci, hb, s3max)
    tabs = st.tabs(["Strength envelope", "Equivalent Mohr–Coulomb", "Deformation modulus", "Effect of disturbance"])
    with tabs[0]:
        s3_line = np.linspace(0, max(s3max * 1.25, 1.0), 200)
        fig = _envelope_figure(s3_line, rock.hb_sigma1(s3_line, gm.sigma_ci, hb.mb, hb.s, hb.a),
                               [("Method 1, Mohr circles", m1), ("Method 2, σ₁ against σ₃", m2)],
                               s3_pts, s1_pts, "Rock mass Hoek–Brown")
        fig.add_trace(go.Scatter(x=s3_line, y=rock.hb_sigma1(s3_line, gm.sigma_ci, gm.mi), name="Intact rock",
                                 mode="lines", line=dict(color=style.GREY, dash="dot")))
        ui.show_plotly(fig)
    with tabs[1]:
        df = pd.DataFrame([
            ("Method 1: regression of R on C", m1.phi_deg, m1.c, m1.r2),
            ("Method 2: regression of σ₁ on σ₃", m2.phi_deg, m2.c, m2.r2),
            ("Hoek et al. (2002) closed form", phi02, c02, float("nan")),
        ], columns=["Method", "φ (°)", "c (MPa)", "R²"]).round(3)
        st.dataframe(df, hide_index=True, width="stretch")
        st.caption(f"All three over 0 ≤ σ₃ ≤ {s3max:g} MPa. The closed form integrates the envelope; the two "
                   "regressions use the evenly spaced points.")
    with tabs[2]:
        em = pd.DataFrame([
            ("Hoek and Diederichs (2006), generalised", rock.modulus_hd_generalised(gm.ei, gm.gsi, gm.d), "E_i, GSI, D"),
            ("Hoek and Diederichs (2006), simplified", rock.modulus_hd_simplified(gm.gsi, gm.d), "GSI, D"),
            ("Hoek, Carranza-Torres and Corkum (2002)", rock.modulus_hoek_2002(gm.sigma_ci, gm.gsi, gm.d), "σ_ci, GSI, D"),
            ("Bieniawski (1978): 2RMR − 100", rock.modulus_bieniawski(rmr) if rmr > 50 else float("nan"), "RMR > 50"),
            ("Serafim and Pereira (1983)", rock.modulus_serafim_pereira(rmr), "RMR"),
        ], columns=["Relation", "E_m (GPa)", "Uses"])
        em["E_m (GPa)"] = em["E_m (GPa)"].round(1)
        st.dataframe(em, hide_index=True, width="stretch")
        st.caption("The generalised relation is the primary estimate where E_i, GSI and D are known; the others "
                   "are sensitivity checks (Chapter 12). They were derived from different data sets and do not agree.")
    with tabs[3]:
        rows = []
        for d in (0.0, 0.5, 0.8, 1.0):
            h = rock.hb_constants(gm.gsi, gm.mi, d)
            rows.append((d, h.mb, h.s, h.a, rock.rock_mass_ucs(gm.sigma_ci, h),
                         float(rock.hb_sigma1(ins.sigma_3, gm.sigma_ci, h.mb, h.s, h.a)),
                         rock.modulus_hd_generalised(gm.ei, gm.gsi, d)))
        st.dataframe(pd.DataFrame(rows, columns=["D", "m_b", "s", "a_HB", "σ_cm (MPa)", "σ₁ₘᶠ (MPa)", "E_m (GPa)"]).round(4),
                     hide_index=True, width="stretch")
        st.caption("D applies to the damaged annulus around the opening, typically one to two metres for a blasted "
                   "tunnel, and not to the whole rock mass.")
    notes = [
        "Generalised Hoek–Brown criterion of Chapter 12; GSI and D as entered on the Ground model page.",
        "The in-situ confinement is the minor stress of the two-stress description.",
        "Equivalent Mohr–Coulomb parameters belong to the stated fitting range.",
    ]
    ui.assumptions(notes)
    n = report.num
    em_table = em.assign(**{"E_m (GPa)": em["E_m (GPa)"].map(lambda v: n(v, 1))})
    report.record(st.session_state, report.Section(
        key="rockmass", title="Rock mass parameters", chapters="Chapter 12",
        purpose="The generalised Hoek–Brown criterion from GSI, m_i and D: strength at the in-situ confinement, "
                "tensile strength, the deformation modulus and the equivalent Mohr–Coulomb parameters.",
        inputs=[
            ("Uniaxial compressive strength of intact rock", r"\sigma_{ci}", n(gm.sigma_ci, 1), "MPa"),
            ("Hoek–Brown constant of intact rock", "m_i", n(gm.mi, 1), "-"),
            ("Geological Strength Index", r"\mathrm{GSI}", n(gm.gsi, 0), "-"),
            ("Disturbance factor", "D", n(gm.d, 1), "-"),
            ("Modulus of intact rock", "E_i", n(gm.ei, 2), "GPa"),
            ("Rock mass rating, for the RMR-based moduli", r"\mathrm{RMR}", n(rmr, 0), "-"),
            ("Minor principal stress in situ", r"\sigma_3", n(ins.sigma_3, 2), "MPa"),
            ("Major principal stress in situ", r"\sigma_1", n(ins.sigma_1, 2), "MPa"),
            ("Upper limit of the fitting range", r"\sigma_{3,max}", n(s3max, 2), "MPa"),
        ],
        equations=EQUATIONS["rock"],
        results=[
            ("Hoek–Brown constant of the rock mass", "m_b", n(hb.mb, 3), "-"),
            ("Hoek–Brown constant", "s", f"{hb.s:.5f}", "-"),
            ("Hoek–Brown exponent", "a_{HB}", n(hb.a, 3), "-"),
            ("Uniaxial strength of the rock mass", r"\sigma_{cm}", n(gm.sigma_cm, 2), "MPa"),
            ("Strength at the in-situ minor stress", r"\sigma_{1m}^f", n(s1m, 1), "MPa"),
            ("Capacity / demand", r"\sigma_{1m}^f/\sigma_1", n(s1m / ins.sigma_1, 2), "-"),
            ("Tensile strength of the rock mass", r"\sigma_{tm}", n(rock.rock_mass_tension(gm.sigma_ci, hb), 3), "MPa"),
            ("Global strength, Hoek et al. (2002)", r"\sigma'_{cm}", n(rock.global_strength(gm.sigma_ci, hb), 2), "MPa"),
            ("Method 1: friction angle / cohesion", r"\phi,\ c", f"{m1.phi_deg:.1f} / {m1.c:.2f}", "° / MPa"),
            ("Method 2: friction angle / cohesion", r"\phi,\ c", f"{m2.phi_deg:.1f} / {m2.c:.2f}", "° / MPa"),
            ("Hoek et al. (2002): friction angle / cohesion", r"\phi,\ c", f"{phi02:.1f} / {c02:.2f}", "° / MPa"),
        ],
        checks=[(f"The rock mass strength at the in-situ minor stress exceeds the major in-situ stress "
                 f"(capacity / demand {s1m / ins.sigma_1:.2f}).", bool(s1m >= ins.sigma_1))],
        tables=[("Estimates of the rock mass deformation modulus.", em_table)],
        figures=[("envelope", "Rock mass Hoek–Brown envelope, the fitting points, the two equivalent Mohr–Coulomb "
                  "lines and the intact envelope.",
                  lambda: figures.envelope(gm.sigma_ci, hb.mb, hb.s, hb.a, s3max, _fit_labels(m1, m2), s3_pts, s1_pts,
                                           "Rock mass Hoek–Brown", intact_mi=gm.mi))],
        assumptions=notes, signature=_signature(gm)))


# ---------------------------------------------------------------------------
# Convergence–confinement (Chapter 16)
# ---------------------------------------------------------------------------


def _ccm_parameters(gm: ground.GroundModel) -> tuple[float, float, float, str]:
    """phi (deg), c (MPa), E_m (GPa) and a description of their source, from the page widgets."""
    sources = ["Equivalent Mohr–Coulomb over 0 ≤ σ₃ ≤ p₀ (method 2)",
               "Hoek et al. (2002) closed form, tunnel range", "Enter φ and c"]
    if gm.phi_adopted is not None:
        sources.insert(0, "Adopted in the book's worked example")
    if st.session_state.get("ccm_mc_source") not in sources:
        st.session_state["ccm_mc_source"] = sources[0]
    c1, c2, c3 = st.columns(3)
    src = c1.selectbox("Mohr–Coulomb parameters", sources, key="ccm_mc_source")
    p0 = gm.p0
    if src.startswith("Adopted"):
        phi, c = gm.phi_adopted, gm.c_adopted
    elif src.startswith("Equivalent"):
        s3, s1 = rock.hb_points(gm.sigma_ci, gm.hb.mb, gm.hb.s, gm.hb.a, p0, 6)
        f = rock.fit_principal_line(s3, s1)
        phi, c = f.phi_deg, f.c
    elif src.startswith("Hoek"):
        phi, c = rock.hoek_2002_mc(gm.sigma_ci, gm.hb, rock.tunnel_sig3_max(gm.sigma_ci, gm.hb, p0))
    else:
        if "ccm_phi" not in st.session_state:
            st.session_state["ccm_phi"], st.session_state["ccm_c"] = 30.0, 1.0
        phi = c2.number_input("Friction angle, φ (°)", 5.0, 70.0, step=0.5, key="ccm_phi")
        c = c3.number_input("Cohesion, c (MPa)", 0.0, 50.0, step=0.05, key="ccm_c")
    em_sources = ["Hoek and Diederichs (2006) from the ground model", "Enter E_m"]
    if gm.em_adopted:
        em_sources.insert(0, "Adopted in the book's worked example")
    if st.session_state.get("ccm_em_source") not in em_sources:
        st.session_state["ccm_em_source"] = em_sources[0]
    src_em = c1.selectbox("Rock mass modulus", em_sources, key="ccm_em_source")
    if src_em.startswith("Adopted"):
        em = gm.em_adopted
    elif src_em.startswith("Hoek"):
        em = rock.modulus_hd_generalised(gm.ei, gm.gsi, gm.d)
    else:
        if "ccm_em" not in st.session_state:
            st.session_state["ccm_em"] = round(float(gm.em_gpa), 2)
        em = c2.number_input("E_m (GPa)", 0.01, 100.0, step=0.1, key="ccm_em")
    return float(phi), float(c), float(em), src


def _support_inputs(a: float) -> list[ccm.Support]:
    st.markdown("**Support installed together** (stiffness adds; capacity at the first element's limit)")
    if "ccm_supports" not in st.session_state:
        st.session_state["ccm_supports"] = ["Concrete or shotcrete ring"]
    chosen = st.multiselect("Elements", ["Concrete or shotcrete ring", "Rock bolts", "Steel sets", "Other (enter p_sm and k_s)"],
                            key="ccm_supports")
    supports: list[ccm.Support] = []
    if "Concrete or shotcrete ring" in chosen:
        c1, c2, c3, c4 = st.columns(4)
        for k, v in (("ccm_t", 0.5), ("ccm_fcc", 60.0), ("ccm_ec", 40.0), ("ccm_nuc", 0.25)):
            st.session_state.setdefault(k, v)
        t = c1.number_input("Ring thickness, t (m)", 0.03, 2.0, step=0.025, key="ccm_t")
        fcc = c2.number_input("Concrete strength, σ_cc (MPa)", 5.0, 120.0, step=1.0, key="ccm_fcc")
        ec = c3.number_input("Concrete modulus, E_c (GPa)", 5.0, 60.0, step=1.0, key="ccm_ec")
        nuc = c4.number_input("Poisson's ratio, ν_c (-)", 0.1, 0.35, step=0.01, key="ccm_nuc")
        supports.append(ccm.support_ring(a, t, fcc, ec, nuc, f"Ring {t:g} m"))
    if "Rock bolts" in chosen:
        c1, c2 = st.columns(2)
        st.session_state.setdefault("ccm_bolt", list(ccm.ROCKBOLTS)[1])
        st.session_state.setdefault("ccm_bolt_s", 1.5)
        kind = c1.selectbox("Bolt", list(ccm.ROCKBOLTS), key="ccm_bolt")
        sp = c2.number_input("Pattern spacing (m)", 0.5, 4.0, step=0.1, key="ccm_bolt_s")
        supports.append(ccm.support_bolts(a, kind, sp))
    if "Steel sets" in chosen:
        c1, c2 = st.columns(2)
        st.session_state.setdefault("ccm_set", list(ccm.STEEL_SETS)[1])
        st.session_state.setdefault("ccm_set_s", 1.0)
        kind = c1.selectbox("Set", list(ccm.STEEL_SETS), key="ccm_set")
        sp = c2.number_input("Longitudinal spacing (m)", 0.5, 3.0, step=0.1, key="ccm_set_s")
        supports.append(ccm.support_steel_sets(a, kind, sp))
    if "Other (enter p_sm and k_s)" in chosen:
        c1, c2 = st.columns(2)
        st.session_state.setdefault("ccm_psm", 1.0)
        st.session_state.setdefault("ccm_ks", 100.0)
        psm = c1.number_input("Capacity, p_sm (MPa)", 0.01, 50.0, step=0.1, key="ccm_psm")
        ks = c2.number_input("Stiffness, k_s (MPa/m)", 1.0, 10000.0, step=10.0, key="ccm_ks")
        supports.append(ccm.Support("Other", psm, ks, psm / ks * 1000.0))
    if len(supports) > 1:
        st.dataframe(pd.DataFrame([(s.name, s.psm, s.ks, s.usm) for s in supports],
                                  columns=["Element", "p_sm (MPa)", "k_s (MPa/m)", "u_sm (mm)"]).round(3),
                     hide_index=True, width="stretch")
    return supports


def _ccm_figure(res: ccm.CCMResult) -> go.Figure:
    g = res.ground
    p, u = ccm.grc_curve(g, res.method)
    finite = np.isfinite(u)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=u[finite], y=p[finite], name="Ground reaction curve", mode="lines",
                             line=dict(color=style.MALACHITE, width=2.4)))
    if res.support is not None:
        s = res.support
        ux = [res.us0, res.us0 + s.usm, max(res.us0 + s.usm, float(np.nanmax(u[finite])) * 1.05)]
        fig.add_trace(go.Scatter(x=ux, y=[0, s.psm, s.psm], name="Support characteristic curve", mode="lines",
                                 line=dict(color=style.CARMINE, width=2.2)))
        if res.eq and res.eq.p_eq is not None and res.eq.u_eq is not None:
            fig.add_trace(go.Scatter(x=[res.eq.u_eq], y=[res.eq.p_eq], name="Equilibrium", mode="markers",
                                     marker=dict(color=style.INK, size=11, symbol="circle-open", line=dict(width=2))))
    if not g.elastic:
        fig.add_hline(y=g.p_cr, line=dict(color=style.GREY, dash="dash"),
                      annotation_text=f"<i>p</i><sub>cr</sub> = {g.p_cr:.2f} MPa", annotation_position="top right")
    fig.add_vline(x=res.us0, line=dict(color=style.GREY, dash="dot"),
                  annotation_text=f"<i>u</i><sub>s0</sub> = {res.us0:.1f} mm", annotation_position="top left")
    fig.update_layout(style.plotly_layout(
        xaxis_title=style.axis_label("radial wall displacement", "<i>u</i><sub>r</sub>", "mm"),
        yaxis_title=style.axis_label("support pressure", "<i>p</i><sub>i</sub>", "MPa"),
        legend=dict(x=0.55, y=0.95)))
    return fig


def _ldp_figure(res: ccm.CCMResult) -> go.Figure:
    g = res.ground
    rstar = res.rp0 / g.a if math.isfinite(res.rp0) else 20.0
    x = np.linspace(-4 * g.a, 12 * g.a, 300)
    fig = go.Figure()
    for idx, m in enumerate(ccm.LDP_METHODS):
        y = [ccm.ldp(v / g.a, rstar, m) * res.u_max for v in x]
        sty = style.series_style(idx, "plotly") if m == res.ldp_method else {"color": style.GREY, "dash": "dot"}
        fig.add_trace(go.Scatter(x=x, y=y, name=m, mode="lines",
                                 line=dict(color=sty["color"], dash=sty["dash"], width=2.2 if m == res.ldp_method else 1.2)))
    fig.add_vline(x=0, line=dict(color=style.INK, width=1), annotation_text="Face")
    if math.isfinite(res.install_distance):
        fig.add_trace(go.Scatter(x=[res.install_distance], y=[res.us0], mode="markers", name="Support installed",
                                 marker=dict(color=style.CARMINE, size=10)))
    fig.update_layout(style.plotly_layout(
        xaxis_title=style.axis_label("distance behind the face", "X", "m"),
        yaxis_title=style.axis_label("radial wall displacement", "<i>u</i><sub>r</sub>", "mm"),
        legend=dict(x=0.45, y=0.35)))
    return fig


def _section_figure(res: ccm.CCMResult):
    g = res.ground
    with plt.rc_context(style.matplotlib_rcparams()):
        fig, ax = plt.subplots(figsize=(4.2, 4.2))
        lim = 1.0
        if math.isfinite(res.rp0) and not g.elastic:
            ax.add_patch(plt.Circle((0, 0), res.rp0, color=style.CARMINE_TINT, zorder=1, label="Plastic zone, unsupported"))
            ax.add_patch(plt.Circle((0, 0), res.rp0, fill=False, color=style.CARMINE, lw=1.0, ls="--", zorder=2))
            lim = res.rp0
        if math.isfinite(res.rp_eq) and res.rp_eq > g.a:
            ax.add_patch(plt.Circle((0, 0), res.rp_eq, color=style.MALACHITE_TINT, zorder=3, label="Plastic zone, supported"))
            ax.add_patch(plt.Circle((0, 0), res.rp_eq, fill=False, color=style.MALACHITE, lw=1.0, zorder=4))
        ax.add_patch(plt.Circle((0, 0), g.a, color=style.WHITE, ec=style.INK, lw=1.4, zorder=5))
        lim = max(lim, g.a) * 1.15
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xlabel(style.axis_label("distance", "x", "m"))
        ax.set_ylabel(style.axis_label("distance", "y", "m"))
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper right", fontsize=7)
        else:
            ax.text(0, -0.95 * lim, "The ground stays elastic", ha="center", fontsize=8, color=style.GREY_DARK)
    return fig


def page_ccm(gm: ground.GroundModel) -> None:
    ui.page_header(
        "Convergence–confinement", "Chapter 16; Appendix F",
        "Ground reaction curve for a circular tunnel in a hydrostatic field, the longitudinal deformation "
        "profile, the support characteristic curve and their equilibrium.",
    )
    st.markdown(f"From the ground model: **a = {gm.radius:g} m, p₀ = γz = {gm.p0:.2f} MPa, ν = {gm.nu:g}** "
                f"({gm.preset}).")
    phi, c, em, src = _ccm_parameters(gm)
    g = ccm.Ground(gm.p0, gm.radius, gm.nu, em, phi, c)
    c1, c2, c3 = st.columns(3)
    st.session_state.setdefault("ccm_method", ccm.GRC_METHODS[0])
    st.session_state.setdefault("ccm_ldp", ccm.LDP_METHODS[0])
    st.session_state.setdefault("ccm_install", "Distance behind the face (m)")
    st.session_state.setdefault("ccm_install_value", 3.0)
    method = c1.selectbox("Ground reaction curve", ccm.GRC_METHODS, key="ccm_method")
    ldp_method = c2.selectbox("Longitudinal deformation profile", ccm.LDP_METHODS, key="ccm_ldp")
    install = c3.selectbox("Support installed at", ["Distance behind the face (m)", "Wall displacement u_s0 (mm)",
                                                     "Radial strain u_s0/a (%)"], key="ccm_install")
    value = c3.number_input("Value", 0.0, 1000.0, step=0.5, key="ccm_install_value")
    supports = _support_inputs(gm.radius)
    mode = {"Distance": "distance", "Wall": "us0", "Radial": "strain"}[install.split()[0]]
    res = ccm.analyse(g, supports, method=method, ldp_method=ldp_method, install=mode, value=value)

    eq = res.eq
    ui.metrics([
        ("Friction angle φ", f"{phi:.2f}°", src),
        ("Cohesion c", ui.fmt(c, 3, "MPa"), src),
        ("E_m", ui.fmt(em * 1000, 0, "MPa")),
        ("σ_cm,MC", ui.fmt(g.scm_mc, 2, "MPa")),
        ("Critical pressure p_cr", ui.fmt(g.p_cr, 2, "MPa") if not g.elastic else "elastic"),
        ("R_p / a, unsupported", ui.fmt(res.rp0 / g.a, 3) if math.isfinite(res.rp0) else "unbounded"),
        ("u_max", ui.fmt(res.u_max, 1, "mm")),
        ("u_s0 at installation", ui.fmt(res.us0, 1, "mm"), f"u* = {res.ustar:.3f}, {res.install_distance:.2f} m behind the face"),
    ])
    if res.support is not None and eq is not None:
        ui.metrics([
            ("Support p_sm", ui.fmt(res.support.psm, 2, "MPa")),
            ("Support k_s (MPa/m)", ui.fmt(res.support.ks, 0)),
            ("Equilibrium p_eq", ui.fmt(eq.p_eq, 2, "MPa")),
            ("Factor of safety p_sm/p_eq", ui.fmt(eq.fos, 2)),
        ])
        if eq.note:
            st.warning(eq.note) if eq.yielded else st.info(eq.note)
        elif eq.fos is not None:
            ui.status_line(eq.fos >= 1.5, f"Factor of safety {eq.fos:.2f}: the support reaches equilibrium within "
                           "its capacity, above the usual 1.5 for a permanent tunnel.",
                           f"Factor of safety {eq.fos:.2f}: below the usual 1.5 for a permanent tunnel.")
    tabs = st.tabs(["Ground and support", "Longitudinal profile", "Plastic zone", "Results table"])
    with tabs[0]:
        ui.show_plotly(_ccm_figure(res))
    with tabs[1]:
        ui.show_plotly(_ldp_figure(res))
    with tabs[2]:
        ui.show_matplotlib(_section_figure(res))
    with tabs[3]:
        rows = [("p₀", g.p0, "MPa"), ("φ", phi, "°"), ("c", c, "MPa"), ("E_m", em * 1000, "MPa"),
                ("k_φ", g.k_phi, "-"), ("σ_cm,MC", g.scm_mc, "MPa"), ("p_cr", g.p_cr, "MPa"),
                ("R_p unsupported", res.rp0, "m"), ("u_max", res.u_max, "mm"), ("u_s0", res.us0, "mm")]
        if res.support is not None and eq is not None:
            rows += [("p_sm", res.support.psm, "MPa"), ("k_s", res.support.ks, "MPa/m"), ("u_sm", res.support.usm, "mm"),
                     ("p_eq", eq.p_eq, "MPa"), ("u_eq", eq.u_eq, "mm"), ("FoS", eq.fos, "-"), ("R_p supported", res.rp_eq, "m")]
        df = pd.DataFrame(rows, columns=["Quantity", "Value", "Unit"])
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce").round(3)
        st.dataframe(df, hide_index=True, width="stretch")
        ui.download_table(df, "ccm_results.csv", key="dl_ccm")
    notes = [
        "Circular opening, hydrostatic in-situ stress p₀ = γz, plane strain, isotropic homogeneous ground.",
        "Elastic-perfectly plastic Mohr–Coulomb ground with zero dilation; a dilatant flow rule increases the displacement.",
        "The two ground reaction curves coincide at ν = 0.5; Salençon's is the form derived in the book.",
        "The longitudinal profiles are written with X* = X/a and R* = R_p/a; software that normalises by the diameter "
        "writes the exponents differently.",
        "Elements installed together: stiffness adds, and the combined capacity is reached at the first element's limit.",
        "The result is optimistic in poor ground: gravity, non-circular deformation, creep and block failure are not represented.",
    ]
    ui.assumptions(notes)
    n = report.num
    inputs = [("In-situ stress, γz", "p_0", n(g.p0, 2), "MPa"), ("Tunnel radius", "a", n(g.a, 2), "m"),
              ("Poisson's ratio of the rock mass", r"\nu", n(g.nu, 2), "-"),
              ("Rock mass modulus", "E_m", n(em * 1000, 0), "MPa"),
              ("Friction angle", r"\phi", n(phi, 2), "°"), ("Cohesion", "c", n(c, 3), "MPa")]
    for sup in supports:
        inputs += [(f"{sup.name}: capacity", "p_{sm}", n(sup.psm, 2), "MPa"),
                   (f"{sup.name}: stiffness", "k_s", n(sup.ks, 0), "MPa/m"),
                   (f"{sup.name}: displacement at capacity", "u_{sm}", n(sup.usm, 2), "mm")]
    results = [("Passive coefficient", r"k_\phi", n(g.k_phi, 3), "-"),
               ("Rock mass strength, Mohr–Coulomb", r"\sigma_{cm,MC}", n(g.scm_mc, 2), "MPa"),
               ("Critical support pressure", "p_{cr}", "elastic" if g.elastic else n(g.p_cr, 2), "MPa"),
               ("Plastic radius, unsupported", "R_p", n(res.rp0, 2), "m"),
               ("Wall displacement, unsupported", "u_{max}", n(res.u_max, 1), "mm"),
               ("Distance of installation behind the face", "X", n(res.install_distance, 2), "m"),
               ("Normalised displacement at installation", "u^*", n(res.ustar, 3), "-"),
               ("Wall displacement at installation", "u_{s0}", n(res.us0, 1), "mm")]
    checks = []
    if res.support is not None and eq is not None:
        results += [("Capacity of the support", "p_{sm}", n(res.support.psm, 2), "MPa"),
                    ("Stiffness of the support", "k_s", n(res.support.ks, 0), "MPa/m"),
                    ("Equilibrium pressure", "p_{eq}", n(eq.p_eq, 2), "MPa"),
                    ("Equilibrium displacement", "u_{eq}", n(eq.u_eq, 1), "mm"),
                    ("Factor of safety", r"p_{sm}/p_{eq}", n(eq.fos, 2), "-"),
                    ("Plastic radius, supported", "R_p", n(res.rp_eq, 2), "m")]
        if eq.note:
            checks.append((eq.note, False if eq.yielded else None))
        elif eq.fos is not None:
            checks.append((f"Factor of safety {eq.fos:.2f} against the usual minimum of 1.5 for a permanent tunnel.",
                           bool(eq.fos >= 1.5)))
    else:
        checks.append(("No support is selected: only the ground reaction is reported.", None))
    report.record(st.session_state, report.Section(
        key="ccm", title="Convergence–confinement", chapters="Chapter 16; Appendix F",
        purpose=(f"Ground reaction curve ({method}), longitudinal deformation profile ({ldp_method}) and support "
                 f"characteristic curve of a circular tunnel in a hydrostatic field. Mohr–Coulomb parameters: {src}. "
                 f"Rock mass modulus: {st.session_state.get('ccm_em_source', '')}. Support installed at "
                 f"{install.split(' (')[0].lower()} = {value:g}."),
        inputs=inputs, equations=EQUATIONS["ccm"], results=results, checks=checks,
        figures=[("grc", "Ground reaction curve, support characteristic curve and their equilibrium.",
                  lambda: figures.ground_support(res)),
                 ("ldp", "Longitudinal deformation profiles; the one used is drawn in colour.",
                  lambda: figures.longitudinal_profile(res)),
                 ("plastic", "Plastic zone around the opening, unsupported and supported.",
                  lambda: _section_figure(res))],
        assumptions=notes, signature=_signature(gm)))


# ---------------------------------------------------------------------------
# Ground movement (Chapters 17 and 26)
# ---------------------------------------------------------------------------


def page_settlement(gm: ground.GroundModel) -> None:
    ui.page_header(
        "Ground movement", "Chapters 17 and 26; Appendix H",
        "Settlement trough from volume loss, its development as the face passes, horizontal strain, and the "
        "first-stage screening of buildings.",
    )
    defaults = {"set_d": 2 * gm.radius, "set_z0": float(gm.depth) if gm.depth <= 60 else 20.0,
                "set_vl": 1.0, "set_k": 0.5, "set_twin": False, "set_spacing": 15.0, "set_vl2": 1.0}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    c1, c2, c3, c4 = st.columns(4)
    d = c1.number_input("Excavated diameter, D (m)", 1.0, 20.0, step=0.1, key="set_d")
    z0 = c2.number_input("Depth to axis, z₀ (m)", 2.0, 100.0, step=0.5, key="set_z0")
    vl = c3.number_input("Volume loss, V_L (%)", 0.05, 10.0, step=0.05, key="set_vl")
    k = c4.number_input("Trough width parameter, K (-)", 0.2, 1.0, step=0.05, key="set_k",
                        help="Soft clay 0.5-0.7; stiff clay 0.4-0.5; granular soil 0.25-0.45; weak rock 0.25-0.4 (Appendix G).")
    twin = st.checkbox("Second, parallel bore", key="set_twin")
    bores = [settlement.Bore(0.0, z0, d, vl, k)]
    if twin:
        c1, c2 = st.columns(2)
        spacing = c1.number_input("Axis spacing (m)", 2.0, 100.0, step=0.5, key="set_spacing")
        vl2 = c2.number_input("Volume loss of the second bore (%)", 0.05, 10.0, step=0.05, key="set_vl2",
                              help="Often higher than the first, through ground the first has disturbed.")
        bores.append(settlement.Bore(spacing, z0, d, vl2, k))
    b = bores[0]
    span = max(4 * b.i, 25.0)
    y = np.linspace(-span, span + (bores[-1].offset if twin else 0.0), 801)
    s = settlement.transverse(bores, y)
    hd, eps = settlement.horizontal(bores, y)
    slope_v = settlement.slope(bores, y)
    ui.metrics([
        ("Trough width i = Kz₀", ui.fmt(b.i, 2, "m")),
        ("Trough volume V_s", ui.fmt(b.vs, 3, "m³/m")),
        ("S_max, one bore", ui.fmt(b.smax_mm, 1, "mm")),
        ("S_max, all bores", ui.fmt(float(s.max()), 1, "mm")),
        ("Maximum slope", f"1 in {1 / max(abs(slope_v).max(), 1e-12):,.0f}"),
        ("Maximum tensile strain", ui.fmt(float(max(eps.max(), 0)), 0, "με")),
        ("Cover / diameter", ui.fmt((z0 - d / 2) / d, 2), "The Gaussian form rests on cases with cover of at least one diameter."),
    ])
    if z0 - d / 2 < d:
        st.warning("Cover above the crown is less than one diameter: the trough becomes narrower and deeper than the "
                   "Gaussian form predicts, and the peak is underestimated (Chapter 17).")
    tabs = st.tabs(["Transverse trough", "Development with the face", "Plan view", "Building screening"])
    with tabs[0]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y, y=-s, name="Settlement", mode="lines", line=dict(color=style.MALACHITE, width=2.4)))
        fig.add_trace(go.Scatter(x=y, y=eps / 100.0, name="Horizontal strain / 100 (με, tension +)", mode="lines",
                                 line=dict(color=style.CARMINE, dash="dash")))
        for bb in bores:
            for side in (-1, 1):
                fig.add_vline(x=bb.offset + side * bb.i, line=dict(color=style.GREY, dash="dot"))
        fig.update_layout(style.plotly_layout(
            xaxis_title=style.axis_label("offset from the first axis", "y", "m"),
            yaxis_title=style.axis_label("settlement (down negative)", "S", "mm"), legend=dict(x=0.02, y=0.15)))
        ui.show_plotly(fig)
        st.caption("Dotted lines: points of inflection at ±i. Strain is compressive over the centre and tensile "
                   "beyond the point of inflection.")
        df = pd.DataFrame({"y (m)": y, "S (mm)": s, "u_h (mm)": hd, "ε_h (με)": eps, "slope (-)": slope_v}).round(4)
        ui.download_table(df, "transverse_trough.csv", key="dl_set_trough")
    with tabs[1]:
        x = np.linspace(-4 * b.i, 4 * b.i, 300)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=-b.smax_mm * settlement.longitudinal_factor(x, 0.0, b.i), mode="lines",
                                 name="Above the axis", line=dict(color=style.MALACHITE, width=2.4)))
        fig.add_vline(x=0, line=dict(color=style.INK, width=1), annotation_text="Face")
        fig.update_layout(style.plotly_layout(
            xaxis_title=style.axis_label("position relative to the face (drive towards +x)", "x", "m"),
            yaxis_title=style.axis_label("settlement", "S", "mm")))
        ui.show_plotly(fig)
        st.caption("Cumulative normal profile: half the final settlement at the face and the whole of it about two "
                   "trough widths behind. A building is hogged as the face approaches and sagged after it passes.")
    with tabs[2]:
        st.session_state.setdefault("set_face", 0.0)
        st.session_state.setdefault("set_lag", 50.0)
        c1, c2 = st.columns(2)
        face = c1.slider("Face of the first bore, x_f (m)", -100.0, 100.0, step=5.0, key="set_face")
        lag = c2.slider("Second bore behind the first (m)", 0.0, 200.0, step=5.0, key="set_lag", disabled=not twin)
        xg = np.linspace(-120, 120, 161)
        yg = np.linspace(y.min(), y.max(), 121)
        field = settlement.plan_field(bores, [face, face - lag][:len(bores)], xg, yg)
        fig = go.Figure(go.Contour(x=xg, y=yg, z=field, colorscale=[[0, style.WHITE], [0.5, style.MALACHITE_TINT], [1, style.MALACHITE]],
                                   colorbar=dict(title="S (mm)"), contours=dict(showlabels=True)))
        fig.update_layout(style.plotly_layout(xaxis_title=style.axis_label("along the drive", "x", "m"),
                                              yaxis_title=style.axis_label("offset", "y", "m")))
        ui.show_plotly(fig)
    with tabs[3]:
        st.markdown("First stage (Chapter 26): a building is excluded only where the predicted settlement is "
                    "below 10 mm **and** the slope below 1 in 500 over its footprint.")
        start = pd.DataFrame({"Building": ["A", "B", "C"], "Near edge y (m)": [-30.0, 5.0, 25.0],
                              "Far edge y (m)": [-12.0, 20.0, 45.0]})
        data = st.data_editor(start, num_rows="dynamic", key="set_buildings_ed", width="stretch")
        rows = []
        for _, row in data.dropna().iterrows():
            r = settlement.screen_building(bores, float(row.iloc[1]), float(row.iloc[2]))
            rows.append((row.iloc[0], r["max_settlement_mm"], 1 / max(r["max_slope"], 1e-12),
                         r["max_tensile_strain_microstrain"], "excluded" if r["excluded"] else "assess further"))
        out = pd.DataFrame(rows, columns=["Building", "Max settlement (mm)", "Max slope (1 in)",
                                          "Max tensile strain (με)", "First stage"])
        st.dataframe(out.style.format({"Max settlement (mm)": "{:.1f}", "Max slope (1 in)": "{:,.0f}",
                                       "Max tensile strain (με)": "{:.0f}"})
                     .map(lambda v: f"color: {style.CARMINE}" if v == "assess further" else f"color: {style.MALACHITE}",
                          subset=["First stage"]),
                     hide_index=True, width="stretch")
    notes = [
        "Greenfield Gaussian trough; i = Kz₀ with the same K for the settlement and the horizontal movement.",
        "Volume loss in per cent of the excavated area; D is the bored diameter, not the internal diameter.",
        "Horizontal movement towards the axis, u_h = yS/z₀ (O'Reilly and New 1982).",
        "Two bores are superposed; interaction beyond superposition is not represented.",
        "Screening uses greenfield movement; building stiffness, which usually reduces the deformation, is a later stage.",
    ]
    ui.assumptions(notes)
    n = report.num
    inputs = [("Excavated diameter", "D", n(d, 2), "m"), ("Depth to the axis", "z_0", n(z0, 1), "m"),
              ("Volume loss", "V_L", n(vl, 2), "%"), ("Trough width parameter", "K", n(k, 2), "-")]
    if twin:
        inputs += [("Axis spacing of the second bore", "s", n(spacing, 1), "m"),
                   ("Volume loss of the second bore", "V_{L2}", n(vl2, 2), "%")]
    checks = [("Cover above the crown is at least one diameter, the range of the Gaussian form.",
               bool(z0 - d / 2 >= d))]
    for _, row in out.iterrows():
        checks.append((f"Building {row['Building']} is excluded at the first stage (settlement below 10 mm and "
                       f"slope below 1 in 500): settlement {row['Max settlement (mm)']:.1f} mm, slope 1 in "
                       f"{row['Max slope (1 in)']:,.0f}.", row["First stage"] == "excluded"))
    report.record(st.session_state, report.Section(
        key="settlement", title="Ground movement", chapters="Chapters 17 and 26; Appendix H",
        purpose="Greenfield settlement trough from volume loss, its development as the face passes, horizontal "
                "strain and the first-stage screening of buildings.",
        inputs=inputs, equations=EQUATIONS["settlement"],
        results=[("Trough width", "i = Kz_0", n(b.i, 2), "m"), ("Trough volume, one bore", "V_s", n(b.vs, 3), "m³/m"),
                 ("Maximum settlement, one bore", "S_{max}", n(b.smax_mm, 1), "mm"),
                 ("Maximum settlement, all bores", "S_{max}", n(float(s.max()), 1), "mm"),
                 ("Maximum slope", "", f"1 in {1 / max(abs(slope_v).max(), 1e-12):,.0f}", "-"),
                 ("Maximum tensile strain", r"\varepsilon_h", n(float(max(eps.max(), 0)), 0), "με"),
                 ("Cover / diameter", "C/D", n((z0 - d / 2) / d, 2), "-")],
        checks=checks,
        tables=[("First-stage screening of the buildings.", report.text_frame(out, 1))] if len(out) else [],
        figures=[("trough", "Transverse settlement trough and horizontal strain; dotted lines at ±i.",
                  lambda: figures.transverse_trough(bores, y, s, eps)),
                 ("face", "Settlement above the axis as the face passes.", lambda: figures.face_development(b))],
        assumptions=notes, signature=_signature(gm)))


# ---------------------------------------------------------------------------
# Segmental lining (Chapter 19)
# ---------------------------------------------------------------------------


def _segment_figure(act: dict):
    sx = act["s"]
    with plt.rc_context(style.matplotlib_rcparams()):
        fig, axes = plt.subplots(4, 1, figsize=(7.2, 8.0))
        axes[0].plot(act["x"], act["y"] - act["y"].min(), color=style.INK, lw=4)
        axes[0].plot(act["x"][[0, -1]], [0, 0], "^", color=style.GREY_DARK, ms=10)
        axes[0].set_aspect("equal", adjustable="datalim")
        axes[0].set_ylabel(style.axis_label("height", "", "m"))
        for ax, key, label, col in ((axes[1], "N", style.axis_label("axial force", "N", "kN"), style.MALACHITE),
                                    (axes[2], "V", style.axis_label("shear force", "V", "kN"), style.GREY_DARK),
                                    (axes[3], "M", style.axis_label("bending moment", "M", "kNm"), style.CARMINE)):
            ax.plot(sx, act[key], color=col)
            ax.axhline(0, color=style.GREY_LIGHT, lw=0.8)
            ax.set_ylabel(label)
        axes[3].set_xlabel(style.axis_label("arc length from the pinned end", "s", "m"))
        fig.tight_layout()
    return fig


def page_lining(gm: ground.GroundModel) -> None:
    ui.page_header(
        "Segmental lining", "Chapter 19; Appendix F",
        "A segment under a transient load case as a curved member, and the compressibility and flexibility "
        "of a complete ring in the ground.",
    )
    tab1, tab2 = st.tabs(["Segment under a transient load", "Ring in the ground"])
    with tab1:
        defaults = {"lin_r": 3.4, "lin_angle": 67.5, "lin_b": 1.5, "lin_h": 0.3, "lin_gc": 25.0, "lin_w": 0.0,
                    "lin_e": 35.0, "lin_fct": 3.0}
        for key, value in defaults.items():
            st.session_state.setdefault(key, value)
        c1, c2, c3, c4 = st.columns(4)
        r = c1.number_input("Radius to centroid, R (m)", 1.0, 10.0, step=0.05, key="lin_r")
        ang = c2.number_input("Segment angle, θ (°)", 10.0, 180.0, step=2.5, key="lin_angle")
        b = c3.number_input("Ring length, b (m)", 0.5, 2.5, step=0.1, key="lin_b")
        h = c4.number_input("Thickness, h (m)", 0.15, 0.8, step=0.01, key="lin_h")
        c1, c2, c3, c4 = st.columns(4)
        gc = c1.number_input("Concrete unit weight (kN/m³)", 20.0, 27.0, step=0.5, key="lin_gc")
        w = c2.number_input("Additional load (kN/m of arc)", 0.0, 500.0, step=1.0, key="lin_w")
        e = c3.number_input("Concrete modulus at the age of the case (GPa)", 5.0, 50.0, step=1.0, key="lin_e")
        fct = c4.number_input("Tensile strength at that age (MPa)", 0.5, 8.0, step=0.1, key="lin_fct")
        seg = lining.Segment(r, ang, b, h, gc, w, e)
        act = lining.segment_actions(seg)
        sts = lining.segment_stresses(seg, act)
        delta = lining.crown_deflection_mm(seg, act)
        tension = float(max((-sts["intrados"]).max(), (-sts["extrados"]).max(), 0.0))
        ui.metrics([
            ("Chord", ui.fmt(seg.chord, 2, "m"), f"Rise of the arc above its chord: {seg.rise:.3f} m."),
            ("Load q", ui.fmt(seg.q, 2, "kN/m")),
            ("Support reaction", ui.fmt(float(act["reaction"][0]), 2, "kN")),
            ("Maximum moment", ui.fmt(float(act["M"].max()), 2, "kNm")),
            ("Axial force at the supports", ui.fmt(float(act["N"][0]), 2, "kN")),
            ("Largest tensile stress", ui.fmt(tension, 2, "MPa")),
            ("Crown deflection", ui.fmt(delta, 3, "mm")),
            ("Tension / strength", ui.fmt(tension / fct, 2)),
        ])
        ui.status_line(tension <= fct, "Tensile stress within the tensile strength at this age.",
                       "Tensile stress exceeds the tensile strength at this age: the segment would crack in this case.")
        ui.show_matplotlib(_segment_figure(act))
        st.caption("Compression positive. The moment is positive when the intrados is in tension. The top panel "
                   "shows the segment with its pinned and sliding supports; it is drawn to scale.")
    with tab2:
        defaults = {"lin_ring_a": gm.radius, "lin_ring_t": 0.3, "lin_ring_el": 35.0, "lin_ring_nul": 0.2,
                    "lin_ring_eg": round(float(gm.em_gpa) * 1000, 0), "lin_ring_nug": gm.nu, "lin_ring_jf": 1.0,
                    "lin_ring_p": round(gm.p0 * 1000 * 0.5, 0)}
        for key, value in defaults.items():
            st.session_state.setdefault(key, value)
        c1, c2, c3, c4 = st.columns(4)
        a = c1.number_input("Ring radius, a (m)", 1.0, 10.0, step=0.05, key="lin_ring_a")
        t = c2.number_input("Thickness, t (m)", 0.1, 1.0, step=0.01, key="lin_ring_t")
        el = c3.number_input("Lining modulus, E_l (GPa)", 5.0, 50.0, step=1.0, key="lin_ring_el")
        nul = c4.number_input("Lining Poisson's ratio, ν_l (-)", 0.1, 0.3, step=0.01, key="lin_ring_nul")
        c1, c2, c3, c4 = st.columns(4)
        eg = c1.number_input("Ground modulus, E (MPa)", 1.0, 100000.0, step=10.0, key="lin_ring_eg")
        nug = c2.number_input("Ground Poisson's ratio, ν (-)", 0.1, 0.49, step=0.01, key="lin_ring_nug")
        jf = c3.number_input("Joint reduction of I (-)", 0.1, 1.0, step=0.05, key="lin_ring_jf",
                             help="I_eff / I; 1 for a monolithic ring. Muir Wood's reduction accounts for joints.")
        p = c4.number_input("Uniform radial pressure, p (kPa)", 0.0, 50000.0, step=10.0, key="lin_ring_p")
        ratios = lining.ring_ratios(lining.Ring(a, t, el, nul, eg, nug, jf))
        thrust = lining.hoop_thrust(p, a)
        ui.metrics([
            ("Compressibility ratio, C", ui.fmt(ratios["C"], 3)),
            ("Flexibility ratio, F", ui.fmt(ratios["F"], 1)),
            ("Hoop thrust N = pa", ui.fmt(thrust, 0, "kN/m")),
            ("Mean stress N/t", ui.fmt(thrust / t / 1000, 2, "MPa")),
        ])
        f = ratios["F"]
        if f > 30:
            st.success("High flexibility ratio: the ring conforms to the ground, carries thrust with little moment, "
                       "and in seismic ovaling its distortion exceeds the free field, approaching that of an "
                       "unlined opening, 4(1 - nu) times the free field (Chapter 20).")
        elif f > 5:
            st.info("Intermediate flexibility ratio: both thrust and moment matter; check the ring with a "
                    "closed-form or numerical interaction analysis.")
        else:
            st.warning("Low flexibility ratio: the ring is stiff relative to the ground and attracts large moments.")
        st.caption("C governs how much of the ground load appears as thrust, F how much appears as moment. The thrust "
                   "pa is the thin-ring value under a uniform pressure; the distortional part of the load needs the "
                   "coefficient functions of the continuum solutions (Appendix F).")
    notes = [
        "Segment: circular arc on a pin and a roller at its ends, loaded by self-weight and a uniform vertical load "
        "per metre of arc; a statically determinate transient case such as handling or stacking.",
        "Deflection by virtual work from bending alone; axial and shear deformation are neglected.",
        "Ring: elastic ring in elastic ground; the ratios are per metre of tunnel.",
        "Joints, the build sequence and the bursting zones under the jacks are outside this page (Chapter 19).",
    ]
    ui.assumptions(notes)
    n = report.num
    report.record(st.session_state, report.Section(
        key="lining", title="Segmental lining", chapters="Chapter 19; Appendix F",
        purpose="A segment under a transient load case as a curved member, and the compressibility and flexibility "
                "of a complete ring in the ground.",
        inputs=[("Segment: radius to the centroid", "R", n(r, 2), "m"), ("Segment angle", r"\theta", n(ang, 1), "°"),
                ("Ring length", "b", n(b, 2), "m"), ("Thickness", "h", n(h, 3), "m"),
                ("Concrete unit weight", r"\gamma_c", n(gc, 1), "kN/m³"),
                ("Additional load per metre of arc", "w", n(w, 1), "kN/m"),
                ("Concrete modulus at the age of the case", "E", n(e, 1), "GPa"),
                ("Tensile strength at that age", "f_{ct}", n(fct, 2), "MPa"),
                ("Ring: radius", "a", n(a, 2), "m"), ("Ring: thickness", "t", n(t, 3), "m"),
                ("Lining modulus", "E_l", n(el, 1), "GPa"), ("Lining Poisson's ratio", r"\nu_l", n(nul, 2), "-"),
                ("Ground modulus", "E", n(eg, 0), "MPa"), ("Ground Poisson's ratio", r"\nu", n(nug, 2), "-"),
                ("Joint reduction of the second moment of area", "I_{eff}/I", n(jf, 2), "-"),
                ("Uniform radial pressure", "p", n(p, 0), "kPa")],
        equations=[r"M(\varphi)=V_AR\left(\sin\varphi+\sin\tfrac{\theta}{2}\right)-qR^2\left[\sin\varphi"
                   r"\left(\varphi+\tfrac{\theta}{2}\right)+\cos\varphi-\cos\tfrac{\theta}{2}\right]",
                   *EQUATIONS["lining"], r"N=p\,a"],
        results=[("Chord of the segment", "", n(seg.chord, 2), "m"), ("Rise of the arc", "", n(seg.rise, 3), "m"),
                 ("Load per metre of arc", "q", n(seg.q, 2), "kN/m"),
                 ("Support reaction", "V_A", n(float(act["reaction"][0]), 2), "kN"),
                 ("Maximum bending moment", "M_{max}", n(float(act["M"].max()), 2), "kNm"),
                 ("Axial force at the supports", "N", n(float(act["N"][0]), 2), "kN"),
                 ("Largest tensile stress", r"\sigma_t", n(tension, 2), "MPa"),
                 ("Crown deflection", r"\delta", n(delta, 3), "mm"),
                 ("Ring compressibility ratio", "C", n(ratios["C"], 3), "-"),
                 ("Ring flexibility ratio", "F", n(ratios["F"], 1), "-"),
                 ("Hoop thrust", "N", n(thrust, 0), "kN/m"),
                 ("Mean hoop stress", "N/t", n(thrust / t / 1000, 2), "MPa")],
        checks=[(f"Largest tensile stress {tension:.2f} MPa against the tensile strength {fct:.2f} MPa at the age "
                 "of the case.", bool(tension <= fct)),
                (("High" if f > 30 else "Intermediate" if f > 5 else "Low") + f" flexibility ratio, F = {f:.1f}: "
                 + ("the ring conforms to the ground and carries thrust with little moment." if f > 30 else
                    "both thrust and moment matter." if f > 5 else
                    "the ring is stiff relative to the ground and attracts large moments."), None)],
        figures=[("segment", "The segment with its supports, and its axial force, shear force and bending moment.",
                  lambda: _segment_figure(act))],
        assumptions=notes, signature=_signature(gm)))


# ---------------------------------------------------------------------------
# Equations and references
# ---------------------------------------------------------------------------


EQUATIONS: dict[str, list[str]] = {
    "rock": [
        r"\sigma_1^f=\sigma_3+\sigma_{ci}\left(m_b\frac{\sigma_3}{\sigma_{ci}}+s\right)^{a_{HB}}",
        r"m_b=m_i\,e^{\frac{\mathrm{GSI}-100}{28-14D}},\quad s=e^{\frac{\mathrm{GSI}-100}{9-3D}},"
        r"\quad a_{HB}=\tfrac12+\tfrac16\left(e^{-\mathrm{GSI}/15}-e^{-20/3}\right)",
        r"\sigma_{cm}=\sigma_{ci}\,s^{a_{HB}},\qquad \sigma_{tm}=-\frac{s\,\sigma_{ci}}{m_b},\qquad "
        r"\sigma_t=-\frac{\sigma_{ci}}{8.62+0.7m_i}",
        r"E_m=E_i\left[0.02+\frac{1-D/2}{1+e^{(60+15D-\mathrm{GSI})/11}}\right]",
        r"R=A+BC:\ \phi=\arcsin B,\ c=\frac{A}{\cos\phi};\qquad "
        r"\sigma_1=a+b\,\sigma_3:\ \sin\phi=\frac{b-1}{b+1},\ c=\frac{a(1-\sin\phi)}{2\cos\phi}",
    ],
    "ccm": [
        r"p_{cr}=\frac{2p_0-\sigma_{cm,MC}}{1+k_\phi},\quad "
        r"R_p=a\left[\frac{2\left(p_0(k_\phi-1)+\sigma_{cm,MC}\right)}{(1+k_\phi)\left(p_i(k_\phi-1)+\sigma_{cm,MC}\right)}\right]^{\frac{1}{k_\phi-1}}",
        r"u_r=\frac{(1+\nu)(p_0-p_i)\,a}{E_m}\ (p_i\ge p_{cr}),\qquad u_r=u_r(p_{cr})\left(\frac{R_p}{a}\right)^2\ (p_i<p_{cr})",
        r"u^*=1-\left[1-\tfrac13e^{-0.15R^*}\right]e^{-\frac{3X^*}{2R^*}},\qquad "
        r"u^*=0.25+0.75\left[1-\left(\frac{0.75}{0.75+X^*}\right)^2\right]",
        r"p_{sm}=\frac{\sigma_{cc}t}{a},\quad k_s=\frac{E_c t}{(1-\nu_c^2)a^2},\quad "
        r"\mathrm{FoS}=\frac{p_{sm}}{p_{eq}}",
    ],
    "settlement": [
        r"S=S_{max}\,e^{-y^2/2i^2}\,\Phi\!\left(\frac{x_f-x}{i}\right),\quad i=Kz_0,\quad "
        r"S_{max}=\frac{V_L}{100}\frac{\pi D^2}{4}\frac{1}{i\sqrt{2\pi}}",
        r"u_h=\frac{y}{z_0}S,\qquad \varepsilon_h=\frac{S}{z_0}\left(\frac{y^2}{i^2}-1\right)",
    ],
    "lining": [
        r"C=\frac{E\,a\,(1-\nu_l^2)}{E_lA_l(1-\nu^2)},\qquad F=\frac{E\,a^3(1-\nu_l^2)}{E_lI_l(1-\nu^2)}",
    ],
}
EQUATION_GROUPS = (("Intact rock and rock mass (Chapters 11 and 12)", "rock"),
                   ("Convergence–confinement (Chapter 16; Appendix F)", "ccm"),
                   ("Ground movement (Chapters 17 and 26)", "settlement"),
                   ("Lining (Chapter 19; Appendix F)", "lining"))

REFERENCES = [
    "Hoek E, Brown ET (2019) The Hoek–Brown failure criterion and GSI – 2018 edition. J Rock Mech Geotech Eng 11(3):445-463",
    "Hoek E, Carranza-Torres C, Corkum B (2002) Hoek–Brown failure criterion – 2002 edition. In: Proceedings of NARMS-TAC, Toronto, pp 267-273",
    "Hoek E, Diederichs MS (2006) Empirical estimation of rock mass modulus. Int J Rock Mech Min Sci 43(2):203-215",
    "Hoek E, Martin CD (2014) Fracture initiation and propagation in intact rock – a review. J Rock Mech Geotech Eng 6(4):287-300",
    "Carranza-Torres C, Fairhurst C (2000) Application of the convergence–confinement method of tunnel design to rock masses "
    "that satisfy the Hoek–Brown failure criterion. Tunn Undergr Space Technol 15(2):187-213",
    "Panet M (1995) Le calcul des tunnels par la méthode convergence–confinement. Presses de l'École Nationale des Ponts et Chaussées, Paris",
    "Vlachopoulos N, Diederichs MS (2009) Improved longitudinal displacement profiles for convergence confinement analysis "
    "of deep tunnels. Rock Mech Rock Eng 42(2):131-146",
    "Mair RJ, Taylor RN (1997) Bored tunnelling in the urban environment. In: Proceedings of the 14th International "
    "Conference on Soil Mechanics and Foundation Engineering, Hamburg, pp 2353-2385",
    "O'Reilly MP, New BM (1982) Settlements above tunnels in the United Kingdom – their magnitude and prediction. "
    "In: Tunnelling '82. Institution of Mining and Metallurgy, London, pp 173-181",
    "Muir Wood AM (1975) The circular tunnel in elastic ground. Géotechnique 25(1):115-127",
    "Einstein HH, Schwartz CW (1979) Simplified analysis for tunnel supports. J Geotech Eng Div 105(GT4):499-518",
]


def page_equations(gm: ground.GroundModel) -> None:
    ui.page_header("Equations and references", "the chapters named below",
                   "The expressions every page uses, as printed in the book, and the sources they come from.")
    for idx, (title, key) in enumerate(EQUATION_GROUPS):
        with st.expander(title, expanded=idx == 0):
            for eq in EQUATIONS[key]:
                st.latex(eq)
    st.subheader("References")
    st.markdown("\n".join(f"- {ref}" for ref in REFERENCES))


# ---------------------------------------------------------------------------
# Design report
# ---------------------------------------------------------------------------

REPORT_ORDER = ("intact", "rockmass", "ccm", "settlement", "lining", "vibration", "smr", "kinematic", "slope_lem")


def _ground_section(gm: ground.GroundModel) -> report.Section:
    n = report.num
    ins = gm.insitu
    results = [("Vertical stress", r"\sigma_v", n(ins.sigma_v, 2), "MPa"),
               ("Horizontal stress", r"\sigma_h", n(ins.sigma_h, 2), "MPa"),
               ("Major principal stress in situ", r"\sigma_1", n(ins.sigma_1, 2), "MPa"),
               ("Minor principal stress in situ", r"\sigma_3", n(ins.sigma_3, 2), "MPa"),
               ("Hoek–Brown constant of the rock mass", "m_b", n(gm.hb.mb, 3), "-"),
               ("Hoek–Brown constant", "s", f"{gm.hb.s:.5f}", "-"),
               ("Hoek–Brown exponent", "a_{HB}", n(gm.hb.a, 3), "-"),
               ("Uniaxial strength of the rock mass", r"\sigma_{cm}", n(gm.sigma_cm, 2), "MPa"),
               ("Rock mass modulus, Hoek and Diederichs (2006)", "E_m",
                n(rock.modulus_hd_generalised(gm.ei, gm.gsi, gm.d), 2), "GPa")]
    if gm.phi_adopted is not None:
        results += [("Friction angle adopted in the book's worked example", r"\phi", n(gm.phi_adopted, 2), "°"),
                    ("Cohesion adopted in the book's worked example", "c", n(gm.c_adopted, 3), "MPa")]
    if gm.em_adopted:
        results.append(("Rock mass modulus adopted in the book's worked example", "E_m", n(gm.em_adopted, 2), "GPa"))
    return report.Section(
        key="ground", title="Ground model", chapters="Chapters 8 and 10-12",
        purpose=((f"Data set: {gm.preset}. " if gm.preset else "User-defined ground. ") + (gm.note or "")).strip(),
        inputs=[("Uniaxial compressive strength of intact rock", r"\sigma_{ci}", n(gm.sigma_ci, 1), "MPa"),
                ("Hoek–Brown constant of intact rock", "m_i", n(gm.mi, 1), "-"),
                ("Modulus of intact rock", "E_i", n(gm.ei, 2), "GPa"),
                ("Poisson's ratio", r"\nu", n(gm.nu, 2), "-"),
                ("Geological Strength Index", r"\mathrm{GSI}", n(gm.gsi, 0), "-"),
                ("Disturbance factor", "D", n(gm.d, 1), "-"),
                ("Unit weight", r"\gamma", n(gm.gamma, 1), "kN/m³"),
                ("Depth to the tunnel axis", "z", n(gm.depth, 0), "m"),
                ("In-situ stress ratio", "k", n(gm.k, 2), "-"),
                ("Tunnel radius", "a", n(gm.radius, 2), "m")],
        equations=[r"\sigma_v=\gamma z,\qquad \sigma_h=k\,\sigma_v", EQUATIONS["rock"][1], EQUATIONS["rock"][3]],
        results=results,
        assumptions=["The in-situ stress is described by the vertical stress γz and one horizontal stress kσ_v.",
                     "The convergence–confinement calculation uses a hydrostatic field p₀ = γz."])


def page_report(gm: ground.GroundModel) -> None:
    ui.page_header(
        "Design report", "the design pages of this platform",
        "A PDF report of the current design: the inputs, equations, results, checks and figures of each page, "
        "in the order of the design chain, set in LaTeX.",
    )
    for key, value in (("rep_project", "Tunnel design calculations"), ("rep_author", ""), ("rep_notes", "")):
        st.session_state.setdefault(key, value)
    c1, c2 = st.columns([3, 2])
    project = c1.text_input("Title", key="rep_project")
    author = c2.text_input("Prepared by", key="rep_author")
    notes = st.text_area("Note to the reader (optional)", key="rep_notes", height=80)

    st.subheader("Contents")
    st.caption("The ground model is always included. A design page enters the report once it has been opened in "
               "this session, with the inputs it had when it was last open.")
    sections = report.recorded(st.session_state)
    sig = _signature(gm)
    chosen = [_ground_section(gm)]
    stale_titles = []
    for key in REPORT_ORDER:
        sec = sections.get(key)
        c1, c2, c3 = st.columns([3, 5, 1])
        if sec is None:
            c1.markdown(PAGES[key])
            c2.caption("Not opened in this session.")
        else:
            st.session_state.setdefault(f"rep_inc_{key}", True)
            include = c1.checkbox(PAGES[key], key=f"rep_inc_{key}")
            stale = sec.signature is not None and sec.signature != sig
            if stale:
                c2.caption(":red[The ground model has changed since this page was last opened: open it again to "
                           "bring it up to date.]")
            else:
                counts = ((len(sec.results), "result"), (len(sec.checks), "check"), (len(sec.figures), "figure"))
                c2.caption(", ".join(f"{k} {word}{'' if k == 1 else 's'}" for k, word in counts) + ".")
            if include:
                chosen.append(sec)
                if stale:
                    stale_titles.append(PAGES[key])
        c3.button("Open", key=f"btn_rep_open_{key}", on_click=go_to_page, args=(key,), width="stretch")
    if stale_titles:
        st.warning("Out of date with the ground model: " + ", ".join(stale_titles)
                   + ". Open these pages again before building the report.")

    stem = report.safe_stem(project)
    if st.button("Build the report", type="primary", key="btn_rep_build"):
        info = report.ReportInfo(project=project.strip() or "Design calculations", author=author,
                                 date=datetime.date.today().strftime("%d %B %Y"),
                                 platform=f"{APP_NAME} {PLATFORM_VERSION}",
                                 ground_label=gm.preset or "user-defined", notes=notes)
        with st.spinner("Setting the report in LaTeX and compiling it"):
            source, assets = report.build_document(info, chosen, REFERENCES)
            pdf, error = report.compile_pdf(source, assets, stem)
        st.session_state["_rep_out"] = {"pdf": pdf, "error": error, "zip": report.latex_pack(source, assets, stem),
                                        "stem": stem, "tex": source, "count": len(chosen)}
    out = st.session_state.get("_rep_out")
    if out:
        c1, c2 = st.columns(2)
        if out["pdf"]:
            c1.download_button("Download the report (PDF)", out["pdf"], f"{out['stem']}.pdf", "application/pdf",
                               key="btn_rep_pdf", type="primary", width="stretch")
        else:
            c1.error(out["error"])
        c2.download_button("Download the LaTeX source and figures (.zip)", out["zip"], f"{out['stem']}_latex.zip",
                           "application/zip", key="btn_rep_zip", width="stretch")
        st.caption(f"{out['count']} sections. The .zip compiles on its own with pdflatex, so the report can be "
                   "edited and extended before submission.")
        with st.expander("LaTeX source"):
            st.code(out["tex"], language="latex")



# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def _external(module: str, function: str):
    """Pages written in their own modules; a missing module shows a notice instead of failing the app."""
    def render(gm: ground.GroundModel) -> None:
        import importlib
        try:
            mod = importlib.import_module(f"geo_platform.{module}")
        except ModuleNotFoundError:
            st.info("This page is being prepared.")
            return
        getattr(mod, function)({"ground": gm})
    return render


RENDERERS = {
    "overview": page_overview,
    "ground": page_ground,
    "intact": page_intact,
    "rockmass": page_rockmass,
    "ccm": page_ccm,
    "settlement": page_settlement,
    "lining": page_lining,
    "vibration": _external("vibration_page", "render_vibration"),
    "smr": _external("slope_pages", "render_smr"),
    "kinematic": _external("slope_pages", "render_kinematic"),
    "slope_lem": _external("slope_pages", "render_slope_lem"),
    "equations": page_equations,
    "report": page_report,
}


def main() -> None:
    st.set_page_config(page_title=APP_NAME, page_icon="⛰️", layout="wide", initial_sidebar_state="expanded",
                       menu_items={"About": f"{APP_NAME} {PLATFORM_VERSION}: companion application of the book "
                                            f"{BOOK_TITLE}. Teaching and screening calculations; not a design tool."})
    init_state()
    app_style()
    gm = ground.from_state(st.session_state)
    page = sidebar(gm)
    RENDERERS[page](gm)


if __name__ == "__main__":
    main()
