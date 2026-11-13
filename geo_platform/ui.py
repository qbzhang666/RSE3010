"""Shared Streamlit building blocks for the pages of the platform.

Every page uses the same few pieces, so the pages read alike:

* :func:`page_header` -- title, the book chapters the page serves, one-line purpose;
* :func:`assumptions` -- the conditions under which the result holds, shown on
  every calculation page, collapsed by default;
* :func:`metrics` -- a row of results;
* :func:`show_matplotlib` and :func:`show_plotly` -- figures in the house style;
* :func:`download_table` -- a CSV of what is on screen.

Widget keys are namespaced by page (``f"{page_id}_{name}"``) so that two pages
never share a widget, while the shared ground model lives under the ``g_``
prefix in ``st.session_state`` (see ``geo_platform.ground``).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import pandas as pd
import streamlit as st

from . import style


def page_header(title: str, chapters: str, purpose: str) -> None:
    """Title, the book chapters the page serves, and what the page is for."""
    st.title(title)
    st.caption(f"Book: {chapters}")
    st.markdown(purpose)


def assumptions(lines: Iterable[str], expanded: bool = False) -> None:
    """The conditions under which the page's result holds."""
    with st.expander("Assumptions and limits of this calculation", expanded=expanded):
        st.markdown("\n".join(f"- {line}" for line in lines))


def metrics(items: Sequence[tuple[str, str] | tuple[str, str, str]], columns: int = 4) -> None:
    """A row (or rows) of results; each item is (label, value[, help])."""
    items = list(items)
    for start in range(0, len(items), columns):
        cols = st.columns(columns)
        for col, item in zip(cols, items[start:start + columns]):
            label, value = item[0], item[1]
            help_text = item[2] if len(item) > 2 else None
            col.metric(label, value, help=help_text)


def status_line(ok: bool, good: str, bad: str) -> None:
    """One-line verdict in the two meaning colours."""
    if ok:
        st.success(good)
    else:
        st.error(bad)


def show_matplotlib(fig) -> None:
    """Draw a matplotlib figure and release it."""
    import matplotlib.pyplot as plt

    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def show_plotly(fig) -> None:
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


def download_table(df: pd.DataFrame, filename: str, label: str = "Download table (CSV)", key: str | None = None) -> None:
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=filename,
                       mime="text/csv", key=key or f"dl_{filename}")


def fmt(value: float, digits: int = 2, unit: str = "") -> str:
    """Number for a metric card; dashes for a value that does not exist."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "–"
    if number != number or number in (float("inf"), float("-inf")):
        return "–"
    text = f"{number:,.{digits}f}"
    return f"{text} {unit}".strip()


__all__ = [
    "page_header", "assumptions", "metrics", "status_line", "show_matplotlib",
    "show_plotly", "download_table", "fmt", "style",
]
