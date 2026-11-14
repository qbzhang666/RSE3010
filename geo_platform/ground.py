"""The shared ground model: one set of inputs read by every design page.

The inputs live in ``st.session_state`` under the ``g_`` prefix, so a value set
on the Ground model page is the value the intact rock, rock mass and
convergence--confinement pages use. Presets reproduce the data sets of the
book. Nothing here draws a widget; the Ground model page does that.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import rock


@dataclass(frozen=True)
class Preset:
    label: str
    source: str
    sigma_ci: float          # MPa
    mi: float
    ei: float                # GPa
    gsi: float
    d: float
    nu: float
    gamma: float             # kN/m3
    depth: float             # m, to the tunnel axis
    k: float                 # in-situ stress ratio
    radius: float            # m, tunnel radius
    # Values adopted in a worked example of the book, used by the CCM page when present.
    phi_adopted: float | None = None
    c_adopted: float | None = None
    em_adopted: float | None = None   # GPa
    note: str = ""


PRESETS: dict[str, Preset] = {p.label: p for p in (
    Preset("Sandstone (Chapters 10-12)", "Chapters 10-12 data set", 110, 17, 30, 63, 0.0, 0.25, 27, 450, 1.8, 5.0,
           note="GSI 63 is the value the book derives from the RMR rating in Chapter 10."),
    Preset("Granite (Chapters 10-12)", "Chapters 10-12 data set", 265, 32, 80, 70, 0.0, 0.20, 27, 780, 3.2, 5.0,
           note="GSI 70 is illustrative: the book rates only the sandstone."),
    Preset("Mudstone (Chapters 10-12)", "Chapters 10-12 data set", 27, 4, 7.6, 30, 0.0, 0.30, 27, 100, 1.0, 5.0,
           note="GSI 30 is illustrative: the book rates only the sandstone."),
    Preset("Limestone (Chapters 10-12)", "Chapters 10-12 data set", 85, 12, 20, 50, 0.0, 0.25, 27, 80, 2.0, 5.0,
           note="GSI 50 is illustrative: the book rates only the sandstone."),
    Preset("Deep tunnel at 650 m, good rock (Chapter 16)", "Chapter 16 worked example", 72, 7, 26, 60, 0.0, 0.22,
           26, 650, 1.0, 4.75, phi_adopted=31.5, c_adopted=2.85, em_adopted=13.52,
           note="Friction angle, cohesion and E_m are the values tabulated in the book's worked example."),
    Preset("Deep tunnel at 650 m, poor rock (Chapter 16)", "Chapter 16 worked example", 72, 7, 26, 30, 0.0, 0.22,
           26, 650, 1.0, 4.75, phi_adopted=23.8, c_adopted=1.72, em_adopted=2.12,
           note="Friction angle, cohesion and E_m are the values tabulated in the book's worked example."),
    Preset("Shallow motorway tunnel in very weak ground (Chapter 16)", "Chapter 16, after Vrakas and Anagnostou (2014)",
           7, 10, 1.0, 15, 0.0, 0.30, 27, 60, 1.0, 6.0, phi_adopted=25.64, c_adopted=None, em_adopted=0.353,
           note="Rock mass strength 0.47 MPa and modulus 353 MPa are the values adopted in the book."),
)}

DEFAULT_PRESET = "Deep tunnel at 650 m, poor rock (Chapter 16)"

KEYS = ("sigma_ci", "mi", "ei", "gsi", "d", "nu", "gamma", "depth", "k", "radius")


def motorway_cohesion(phi_deg: float = 25.64, scm_mc: float = 0.47) -> float:
    """Cohesion that gives the adopted unconfined strength: c = scm (1 - sin phi)/(2 cos phi)."""
    import math
    phi = math.radians(phi_deg)
    return scm_mc * (1 - math.sin(phi)) / (2 * math.cos(phi))


def preset_values(label: str) -> dict:
    p = PRESETS[label]
    c = p.c_adopted
    if c is None and p.phi_adopted is not None:
        c = motorway_cohesion(p.phi_adopted)
    return {
        "sigma_ci": p.sigma_ci, "mi": p.mi, "ei": p.ei, "gsi": p.gsi, "d": p.d, "nu": p.nu,
        "gamma": p.gamma, "depth": p.depth, "k": p.k, "radius": p.radius,
        "phi_adopted": p.phi_adopted, "c_adopted": c, "em_adopted": p.em_adopted,
        "preset": label, "note": p.note,
    }


@dataclass
class GroundModel:
    """The current inputs and the quantities every page derives from them."""
    preset: str
    sigma_ci: float
    mi: float
    ei: float
    gsi: float
    d: float
    nu: float
    gamma: float
    depth: float
    k: float
    radius: float
    phi_adopted: float | None = None
    c_adopted: float | None = None
    em_adopted: float | None = None
    note: str = ""
    hb: rock.HBConstants = field(init=False)
    insitu: rock.InSitu = field(init=False)

    def __post_init__(self) -> None:
        self.hb = rock.hb_constants(self.gsi, self.mi, self.d)
        self.insitu = rock.insitu_stresses(self.depth, self.k, self.gamma)

    @property
    def p0(self) -> float:
        """Hydrostatic field for the CCM: the vertical stress gamma z (MPa)."""
        return self.gamma * self.depth / 1000.0

    @property
    def em_gpa(self) -> float:
        return self.em_adopted if self.em_adopted else rock.modulus_hd_generalised(self.ei, self.gsi, self.d)

    @property
    def sigma_cm(self) -> float:
        return rock.rock_mass_ucs(self.sigma_ci, self.hb)


def from_state(state) -> GroundModel:
    """Build the ground model from ``st.session_state`` (or any mapping with g_ keys)."""
    values = preset_values(DEFAULT_PRESET)
    for key in KEYS + ("phi_adopted", "c_adopted", "em_adopted", "preset", "note"):
        if f"g_{key}" in state:
            values[key] = state[f"g_{key}"]
    return GroundModel(**{k: values[k] for k in (
        "preset", "sigma_ci", "mi", "ei", "gsi", "d", "nu", "gamma", "depth", "k", "radius",
        "phi_adopted", "c_adopted", "em_adopted", "note")})


def load_preset(state, label: str) -> None:
    """Write a preset into the state, including the widget keys of the Ground model page."""
    for key, value in preset_values(label).items():
        state[f"g_{key}"] = value
