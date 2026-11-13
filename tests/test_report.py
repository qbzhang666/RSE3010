"""Tests of the design report: the LaTeX text conversion, the sections the pages
record, and (where pdflatex is installed) a compiled PDF.

    python -m pytest tests/test_report.py -q
"""

from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from geo_platform import report  # noqa: E402

DESIGN_PAGES = ["ccm", "intact", "rockmass", "settlement", "lining", "vibration", "smr", "kinematic", "slope_lem"]


def test_tex_converts_the_platform_symbols():
    assert report.tex("σ_ci") == r"$\sigma_{ci}$"
    assert report.tex("z₀ = 20 m") == r"$z_{0}$ = 20 m"
    assert report.tex("σ₁ᶠ at in-situ σ₃") == r"$\sigma _{1}^{f}$ at in-situ $\sigma _{3}$"
    assert report.tex("50% & 1_000 #") == r"50\% \& 1\_000 \#"
    assert report.tex("23.8°") == r"23.8$^{\circ}$"
    assert report.tex("kN/m³") == r"kN/m$^{3}$"
    assert report.tex("u_s0 and a_HB, cm") == r"$u_{s0}$ and $a_{HB}$, cm"


def test_number_format():
    assert report.num(126.92, 1) == "126.9"
    assert report.num(float("nan")) == "--"
    assert report.num(None) == "--"
    assert report.num(12345.6, 0) == "12 346"


def _session_through_every_page():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(os.path.join(ROOT, "rse3010_geo.py"), default_timeout=120)
    for page in DESIGN_PAGES:
        at.session_state["page"] = page
        at.run()
        assert not at.exception, (page, at.exception)
    return at


def test_every_design_page_records_a_section():
    at = _session_through_every_page()
    sections = at.session_state[report.STATE_KEY]
    assert set(sections) == set(DESIGN_PAGES)
    ccm = sections["ccm"]
    assert dict((q, v) for q, _, v, _ in ccm.results)["Factor of safety"] == "2.78"
    at.session_state["page"] = "report"
    at.run()
    assert not at.exception, at.exception
    assert not at.warning


def test_values_survive_a_visit_to_another_page():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(os.path.join(ROOT, "rse3010_geo.py"), default_timeout=120)
    at.session_state["page"] = "kinematic"
    at.run()
    at.number_input(key="kin_H").set_value(25.0).run()
    at.session_state["page"] = "overview"
    at.run()
    at.session_state["page"] = "kinematic"
    at.run()
    assert at.number_input(key="kin_H").value == 25.0
    assert not at.exception, at.exception


def test_document_builds_and_compiles(tmp_path):
    import datetime

    at = _session_through_every_page()
    sections = at.session_state[report.STATE_KEY]
    import importlib.util

    spec = importlib.util.spec_from_file_location("rse3010_geo", os.path.join(ROOT, "rse3010_geo.py"))
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    from geo_platform import ground

    gm = ground.from_state({})
    chosen = [app._ground_section(gm)] + [sections[k] for k in app.REPORT_ORDER if k in sections]
    info = report.ReportInfo("Test report", "A student", datetime.date(2026, 11, 13).strftime("%d %B %Y"),
                             "RSE3010 Geotechnical Platform 1.0", gm.preset)
    source, assets = report.build_document(info, chosen, app.REFERENCES)
    assert source.count(r"\section{") == len(chosen)
    assert len(assets) >= 10 and all(data.startswith(b"%PDF") for data in assets.values())
    assert report.latex_pack(source, assets)[:2] == b"PK"
    if report.find_pdflatex() is None:
        pytest.skip("pdflatex is not installed here")
    pdf, error = report.compile_pdf(source, assets, "test_report", timeout=300)
    assert error is None, error
    assert pdf[:4] == b"%PDF"
    (tmp_path / "test_report.pdf").write_bytes(pdf)
