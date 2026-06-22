"""
Friendly key-value renderer for pipeline parameters.
Used instead of raw `st.json` to present small config dicts in a compact,
human-readable form.
"""
import streamlit as st

from app.i18n import t


def render_params(params: dict, expanded: bool = False) -> None:
    """Render a shallow dict of parameters as a two-column key/value list.

    Falls back to `st.json` when the object is not a plain dict.
    """
    if not params:
        st.caption(t("(no parameters)"))
        return

    if not isinstance(params, dict):
        st.json(params, expanded=expanded)
        return

    # Two-column layout: keys on the left, values on the right.
    for k, v in params.items():
        cols = st.columns([1, 3])
        cols[0].markdown(f"**{k}**")
        cols[1].write(v)
