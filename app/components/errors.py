"""
Error display helpers.

Replaces the pattern `st.error(msg); st.exception(exc)` used in upload.py and
elsewhere. Showing raw tracebacks scares non-technical users (the intended
audience here is environmental scientists, not developers); these helpers
keep the surface message clean while still allowing technical inspection
on demand via an expander.

Usage
-----
    from app.components.errors import show_error, show_warning

    try:
        ...
    except Exception as e:
        show_error(
            "Could not read the uploaded file. Make sure it is a valid Excel/CSV.",
            exc=e,
        )
"""

from __future__ import annotations

import streamlit as st

from app.i18n import t


def show_error(message: str, exc: Exception | None = None) -> None:
    """Show a user-friendly error message.

    The plain-language `message` is shown as a normal Streamlit error.
    If `exc` is provided, an expandable section labelled "Technical details"
    is added underneath; it contains the exception type, message, and the
    full traceback. Casual users can ignore it; developers can open it.
    """
    st.error(message)
    if exc is not None:
        with st.expander(t("Technical details (for debugging)")):
            st.code(f"{type(exc).__name__}: {exc}", language="text")
            st.exception(exc)


def show_warning(message: str, detail: str | None = None) -> None:
    """Show a warning with an optional inline detail (no traceback)."""
    st.warning(message)
    if detail:
        st.caption(detail)
