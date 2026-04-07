"""Helper functions for rendering metric displays in Streamlit."""

from __future__ import annotations

import streamlit as st


def render_metric_row(metrics: list[dict]) -> None:
    """Render a row of ``st.metric`` cards.

    Each dict in *metrics* should have keys ``label``, ``value``, and
    optionally ``delta`` and ``delta_color``.
    """
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        col.metric(
            label=m["label"],
            value=m["value"],
            delta=m.get("delta"),
            delta_color=m.get("delta_color", "normal"),
        )


def empty_state(title: str, body: str, icon: str = "ℹ️") -> None:
    """Display a styled empty-state message with guidance."""
    st.info(f"**{title}**\n\n{body}", icon=icon)
