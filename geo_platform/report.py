"""Design calculation report: sections recorded by the pages, set in LaTeX and
compiled to PDF.

Each design page, after it has computed its results, records a ``Section``
in ``st.session_state`` (``record``). The report page assembles the recorded
sections, in the order of the design chain, into one LaTeX document
(``build_document``) and compiles it with pdflatex (``compile_pdf``), as the
pumped hydro platform does. The figures are drawn again with matplotlib for
print and embedded as PDF, so they stay vector.

Streamlit is not imported here, so the assembly can be tested on its own.
"""

from __future__ import annotations

import io
import math
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

STATE_KEY = "_report_sections"

# A row of an inputs or results table: (quantity, symbol as LaTeX math, value, unit).
Row = tuple[str, str, str, str]


@dataclass
class Section:
    key: str
    title: str
    chapters: str
    purpose: str
    inputs: list[Row] = field(default_factory=list)
    equations: list[str] = field(default_factory=list)
    results: list[Row] = field(default_factory=list)
    # (statement, verdict): True satisfied, False not satisfied, None information only.
    checks: list[tuple[str, bool | None]] = field(default_factory=list)
    # (file stem, caption, builder returning a matplotlib figure)
    figures: list[tuple[str, str, Callable]] = field(default_factory=list)
    # (caption, DataFrame with values already formatted as text)
    tables: list[tuple[str, object]] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    # Ground-model values the section was computed with; None if it does not depend on them.
    signature: tuple | None = None


def record(state, section: Section) -> None:
    """Store (or replace) a page's section for the report."""
    sections = state.get(STATE_KEY)
    if not isinstance(sections, dict):
        sections = {}
    sections[section.key] = section
    state[STATE_KEY] = sections


def recorded(state) -> dict[str, Section]:
    sections = state.get(STATE_KEY)
    return sections if isinstance(sections, dict) else {}


# ---------------------------------------------------------------------------
# Numbers and text
# ---------------------------------------------------------------------------


def num(value, digits: int = 2) -> str:
    """A number for a report table; an en dash where the value does not exist."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "--"
    if not math.isfinite(x):
        return "--"
    text = f"{x:,.{digits}f}"
    return text.replace(",", "\u2009") if abs(x) >= 10000 else text.replace(",", "")


_GREEK = {
    "α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta", "ε": r"\varepsilon", "η": r"\eta",
    "θ": r"\theta", "λ": r"\lambda", "μ": r"\mu", "ν": r"\nu", "π": r"\pi", "ρ": r"\rho", "σ": r"\sigma",
    "τ": r"\tau", "φ": r"\phi", "ψ": r"\psi", "ω": r"\omega", "Δ": r"\Delta", "Φ": r"\Phi", "Σ": r"\Sigma",
}
_MATH_SYMBOLS = {
    "±": r"\pm", "≤": r"\le", "≥": r"\ge", "×": r"\times", "·": r"\cdot", "√": r"\surd", "→": r"\rightarrow",
    "≈": r"\approx", "∞": r"\infty", "−": "-", "°": r"^{\circ}", "′": r"'", "µ": r"\mu ",
}
_SUPERSCRIPTS = {"²": "2", "³": "3", "¹": "1", "⁴": "4", "ᵃ": "a", "ᶠ": "f", "ⁿ": "n"}
_SUBSCRIPTS = {"₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4", "ₘ": "m", "ₐ": "a", "ₑ": "e", "ᵢ": "i", "ₚ": "p", "ₛ": "s"}
_TEXT_SPECIAL = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
    "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    "\u2009": r"\,", "\u202f": r"\,", "\u00a0": "~",
}
_SUB_TEXT = r"[A-Za-z0-9]+(?:,[A-Za-z0-9]+)*"
_GREEK_CLASS = "".join(_GREEK)
_SUBSCRIPTED = re.compile(rf"([A-Za-z{_GREEK_CLASS}])_({_SUB_TEXT})")


def tex(text) -> str:
    """Plain text with the platform's Unicode symbols, as LaTeX for pdflatex.

    Greek letters, primes, degree signs, Unicode sub- and superscripts and the
    underscore notation of the pages (``σ_ci``, ``u_s0``) become inline math;
    the LaTeX special characters are escaped.
    """
    s = str(text)
    parts: list[list] = []          # [is_math, text]

    def add(is_math: bool, piece: str) -> None:
        if parts and parts[-1][0] == is_math:
            parts[-1][1] += piece
        else:
            parts.append([is_math, piece])

    i = 0
    while i < len(s):
        m = _SUBSCRIPTED.match(s, i)
        if m and (m.group(1) in _GREEK or i == 0 or not s[i - 1].isalpha()):
            base = _GREEK.get(m.group(1), m.group(1))
            add(True, f"{base}_{{{m.group(2)}}}")
            i = m.end()
            continue
        ch = s[i]
        if ch in _GREEK:
            add(True, _GREEK[ch] + " ")
        elif ch in _MATH_SYMBOLS:
            add(True, _MATH_SYMBOLS[ch])
        elif ch in _SUPERSCRIPTS:
            add(True, f"^{{{_SUPERSCRIPTS[ch]}}}")
        elif ch in _SUBSCRIPTS:
            # Carry a single preceding letter into the math, so that z₀ reads $z_{0}$.
            if parts and not parts[-1][0] and parts[-1][1][-1:].isalpha() and \
                    (len(parts[-1][1]) == 1 or not parts[-1][1][-2].isalpha()):
                letter = parts[-1][1][-1]
                parts[-1][1] = parts[-1][1][:-1]
                if not parts[-1][1]:
                    parts.pop()
                add(True, letter)
            add(True, f"_{{{_SUBSCRIPTS[ch]}}}")
        else:
            add(False, _TEXT_SPECIAL.get(ch, ch))
        i += 1
    return "".join(f"${p}$" if is_math else p for is_math, p in parts)


# ---------------------------------------------------------------------------
# LaTeX blocks
# ---------------------------------------------------------------------------

PREAMBLE = r"""\documentclass[a4paper,11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath,amssymb}
\usepackage{array}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{float}
\usepackage{xcolor}
\usepackage[margin=2.3cm,headheight=14pt]{geometry}
\usepackage{fancyhdr}
\usepackage{lastpage}
\usepackage{caption}
\usepackage{enumitem}
\usepackage{titlesec}
\usepackage[hidelinks]{hyperref}
\usepackage{microtype}
\usepackage{needspace}

\definecolor{Malachite}{HTML}{2F6B57}
\definecolor{Carmine}{HTML}{8C2B3E}
\definecolor{GreyDark}{HTML}{5A5A5A}

\titleformat{\section}{\large\bfseries\color{Malachite}}{\thesection}{0.8em}{}
\titleformat{\subsection}{\normalsize\bfseries}{\thesubsection}{0.8em}{}
\titlespacing*{\section}{0pt}{14pt}{6pt}
\titlespacing*{\subsection}{0pt}{10pt}{4pt}
\captionsetup{font=small,labelfont={bf,color=Malachite}}
\setlist{topsep=2pt,itemsep=1pt,parsep=0pt}
\setlength{\parindent}{0pt}
\setlength{\parskip}{4pt}
\renewcommand{\arraystretch}{1.12}
\newcommand{\satisfied}{\textcolor{Malachite}{\textbf{Satisfied}}}
\newcommand{\notsatisfied}{\textcolor{Carmine}{\textbf{Not satisfied}}}
\newcommand{\information}{\textcolor{GreyDark}{\textbf{Note}}}
"""


def _row_table(rows: Iterable[Row]) -> str:
    lines = [r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.47\linewidth}>{\raggedright\arraybackslash}p{0.17\linewidth}r>{\raggedright\arraybackslash}p{0.13\linewidth}@{}}",
             r"\toprule", r"Quantity & Symbol & Value & Unit \\", r"\midrule"]
    for quantity, symbol, value, unit in rows:
        sym = f"${symbol}$" if symbol else ""
        lines.append(f"{tex(quantity)} & {sym} & {tex(value)} & {tex(unit)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


_NUMERIC = re.compile(r"^\s*[-+]?[\d ,]*\.?\d+\s*$|^\s*--\s*$")


def _frame_table(df, caption: str) -> str:
    columns = list(df.columns)
    # Numbers right-aligned, text left-aligned.
    spec = "".join("r" if all(_NUMERIC.match(str(v)) for v in df[c]) else "l" for c in columns)
    wide = len(columns) >= 8
    lines = [r"\begin{table}[H]", r"\centering", r"\small" if not wide else r"\footnotesize",
             f"\\caption{{{tex(caption)}}}"]
    if wide:
        # shrink to the text width if wider; never enlarge
        lines.append(r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%")
    lines += [f"\\begin{{tabular}}{{@{{}}{spec}@{{}}}}", r"\toprule",
              " & ".join(tex(c) for c in columns) + r" \\", r"\midrule"]
    for _, row in df.iterrows():
        lines.append(" & ".join(tex(v) for v in row.tolist()) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}" + ("}" if wide else ""), r"\end{table}"]
    return "\n".join(lines)


def text_frame(df, digits: int = 2):
    """A copy of a table with every value as report text: numbers rounded, booleans as yes or no."""
    import numbers

    def cell(value):
        if isinstance(value, bool) or type(value).__name__ == "bool_":
            return "yes" if value else "no"
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "--"
        if isinstance(value, numbers.Number):
            return num(value, digits)
        return str(value)

    return df.apply(lambda col: col.map(cell))


def _itemize(items: Iterable[str]) -> str:
    items = list(items)
    if not items:
        return ""
    return "\\begin{itemize}\n" + "\n".join(f"\\item {tex(x)}" for x in items) + "\n\\end{itemize}"


def _verdict(ok: bool | None) -> str:
    return r"\information" if ok is None else (r"\satisfied" if ok else r"\notsatisfied")


def _checks(checks) -> str:
    if not checks:
        return ""
    return ("\\begin{itemize}\n"
            + "\n".join(f"\\item {_verdict(ok)}: {tex(text)}" for text, ok in checks)
            + "\n\\end{itemize}")


def figure_pdf(builder: Callable) -> bytes | None:
    """Draw a figure with its builder and return it as PDF bytes."""
    import matplotlib.pyplot as plt

    try:
        fig = builder()
    except Exception:          # a figure that cannot be drawn is left out, not fatal
        return None
    if fig is None:
        return None
    buffer = io.BytesIO()
    fig.savefig(buffer, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def section_latex(section: Section, assets: dict[str, bytes]) -> str:
    """One section of the report; its figures are added to ``assets``."""
    out = [f"\\section{{{tex(section.title)}}}",
           f"\\textit{{Book: {tex(section.chapters)}.}} {tex(section.purpose)}"]
    if section.inputs:
        out += [r"\subsection{Inputs}", r"\begin{center}", _row_table(section.inputs), r"\end{center}"]
    if section.equations:
        out.append(r"\subsection{Equations}")
        out += [f"\\[ {eq} \\]" for eq in section.equations]
    if section.results:
        out += [r"\subsection{Results}", r"\begin{center}", _row_table(section.results), r"\end{center}"]
    if section.checks:
        out += [r"\subsection{Checks}", _checks(section.checks)]
    for caption, df in section.tables:
        out.append(_frame_table(df, caption))
    for stem, caption, builder in section.figures:
        data = figure_pdf(builder)
        if data is None:
            continue
        name = f"{section.key}_{stem}.pdf"
        assets[name] = data
        out += [r"\begin{figure}[H]", r"\centering",
                f"\\includegraphics[width=0.82\\linewidth,height=0.42\\textheight,keepaspectratio]{{{name}}}",
                f"\\caption{{{tex(caption)}}}", r"\end{figure}"]
    if section.assumptions:
        out += [r"\subsection{Assumptions and limits}", _itemize(section.assumptions)]
    # Keep a heading with the start of what follows it.
    return ("\n\n".join(out)
            .replace(r"\section{", r"\needspace{12\baselineskip}\section{")
            .replace(r"\subsection{", r"\needspace{8\baselineskip}\subsection{"))


@dataclass
class ReportInfo:
    project: str
    author: str
    date: str
    platform: str
    ground_label: str
    notes: str = ""


def build_document(info: ReportInfo, sections: list[Section], references: list[str]) -> tuple[str, dict[str, bytes]]:
    """The LaTeX source of the report and the figure files it includes."""
    assets: dict[str, bytes] = {}
    body = [section_latex(s, assets) for s in sections]

    summary_parts = []
    for s in sections:
        verdicts = [(text, ok) for text, ok in s.checks if ok is not None]
        if verdicts:
            summary_parts.append(f"\\needspace{{3\\baselineskip}}\\textbf{{{tex(s.title)}}}\n" + _checks(verdicts))
    summary = ""
    if summary_parts:
        summary = r"\section*{Summary of checks}" + "\n" + "\n\n".join(summary_parts)

    notes = f"\n\n{tex(info.notes)}" if info.notes.strip() else ""
    head = f"""
\\pagestyle{{fancy}}
\\fancyhf{{}}
\\lhead{{\\small {tex(info.platform)}}}
\\rhead{{\\small {tex(info.project)}}}
\\cfoot{{\\small Page \\thepage\\ of \\pageref{{LastPage}}}}
\\renewcommand{{\\headrulewidth}}{{0.4pt}}

\\begin{{document}}

\\begin{{center}}
{{\\LARGE\\bfseries\\color{{Malachite}} Design calculation report}}\\\\[6pt]
{{\\large {tex(info.project)}}}\\\\[4pt]
{{\\small {tex(('Prepared by ' + info.author + ', ') if info.author.strip() else '')}{tex(info.date)}}}
\\end{{center}}

\\noindent\\textcolor{{Malachite}}{{\\rule{{\\linewidth}}{{0.6pt}}}}

\\textbf{{Basis.}} The calculations were made with the {tex(info.platform)}, the companion application of the
book \\textit{{Underground Design and Construction}}. Ground model: {tex(info.ground_label)}. They are screening
and teaching calculations: each states its assumptions, and each returns an answer whether or not those
assumptions apply. Check the results independently before relying on them.{notes}

\\noindent\\textcolor{{Malachite}}{{\\rule{{\\linewidth}}{{0.6pt}}}}
"""
    references = list(references)
    for s in sections:
        references += [r for r in s.references if r not in references]
    refs = ""
    if references:
        refs = "\\section*{References}\n\\begin{itemize}[leftmargin=1.2em,label={}]\n" + \
               "\n".join(f"\\item {tex(r)}" for r in references) + "\n\\end{itemize}"
    source = PREAMBLE + head + summary + "\n\n" + "\n\n".join(body) + "\n\n" + refs + "\n\\end{document}\n"
    return source, assets


# ---------------------------------------------------------------------------
# Compilation and packaging
# ---------------------------------------------------------------------------


def find_pdflatex() -> str | None:
    """pdflatex on the PATH, or in the usual MiKTeX and TeX Live folders on Windows."""
    for name in ("pdflatex", "pdflatex.exe"):
        found = shutil.which(name)
        if found:
            return found
    candidates: list[Path] = []
    for env in ("LOCALAPPDATA", "ProgramFiles", "ProgramFiles(x86)"):
        root = os.environ.get(env)
        if not root:
            continue
        base = Path(root) / ("Programs" if env == "LOCALAPPDATA" else "") / "MiKTeX" / "miktex" / "bin"
        candidates += [base / "x64" / "pdflatex.exe", base / "pdflatex.exe"]
    texlive = Path("C:/texlive")
    if texlive.exists():
        candidates += sorted(texlive.glob("*/bin/windows/pdflatex.exe"), reverse=True)
    for path in candidates:
        if path.exists():
            return str(path)
    return None


def compile_pdf(source: str, assets: dict[str, bytes], stem: str = "design_report",
                timeout: int = 120) -> tuple[bytes | None, str | None]:
    """Compile the report; returns (PDF bytes, None) or (None, the reason it failed)."""
    pdflatex = find_pdflatex()
    if pdflatex is None:
        return None, ("pdflatex was not found on this machine. Download the LaTeX pack instead and compile "
                      "it with any TeX distribution (MiKTeX, TeX Live or Overleaf).")
    with tempfile.TemporaryDirectory(prefix="rse3010_report_") as tmp:
        folder = Path(tmp)
        (folder / f"{stem}.tex").write_text(source, encoding="utf-8")
        for name, data in assets.items():
            (folder / name).write_bytes(data)
        command = [pdflatex, "-interaction=nonstopmode", "-halt-on-error", f"{stem}.tex"]
        completed = None
        try:
            for _ in range(2):          # the second pass resolves the page count
                completed = subprocess.run(command, cwd=folder, capture_output=True, timeout=timeout)
                if completed.returncode != 0:
                    break
        except subprocess.TimeoutExpired:
            return None, f"pdflatex did not finish within {timeout} s."
        except OSError as exc:
            return None, f"pdflatex could not be started: {exc}"
        pdf = folder / f"{stem}.pdf"
        if completed is not None and completed.returncode == 0 and pdf.exists() and pdf.stat().st_size > 0:
            return pdf.read_bytes(), None
        log = folder / f"{stem}.log"
        text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
        if not text and completed is not None:
            text = (completed.stdout or b"").decode("utf-8", errors="replace")
        errors = [ln for ln in text.splitlines() if ln.startswith("!")]
        detail = "\n".join(errors[:5]) or text[-1500:]
        return None, f"pdflatex stopped with an error:\n{detail}"


def latex_pack(source: str, assets: dict[str, bytes], stem: str = "design_report") -> bytes:
    """The .tex source and its figures in one zip, which compiles on its own."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as pack:
        pack.writestr(f"{stem}.tex", source)
        for name, data in assets.items():
            pack.writestr(name, data)
    return buffer.getvalue()


def safe_stem(text: str, fallback: str = "design_report") -> str:
    stem = re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_").lower()
    return stem[:60] or fallback
