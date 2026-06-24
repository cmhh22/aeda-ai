"""
Page header component.

Used at the top of every page for visual consistency. Replaces the ad-hoc
mix of `st.header() / st.title() / st.write() / st.caption()` calls that
each page currently uses with a single, themed pattern.

Usage
-----
    from app.components.page_header import page_header

    def render():
        page_header(
            title="Upload & Configure",
            description="Upload your environmental dataset and configure the analysis.",
            icon=":material/upload:",
        )
        # ... rest of the page ...
"""

from __future__ import annotations

import streamlit as st


def page_header(
    title: str,
    description: str | None = None,
    icon: str | None = None,
) -> None:
    """Render a standard page header.

    Parameters
    ----------
    title : str
        The page title (e.g. "Upload & Configure").
    description : str, optional
        A short subtitle shown below the title in caption style. Keep it under
        ~120 characters so it fits on one line on most screens.
    icon : str, optional
        A Material icon name (e.g. ":material/upload:") or None.
        Rendered next to the title using Streamlit's native Material icon support.
    """
    if icon:
        st.markdown(
            f'<div class="page-header-row">'
            f'<span class="page-header-icon">{icon}</span>'
            f'<h1 class="page-header-title">{title}</h1>'
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.title(title)

    if description:
        st.caption(description)

    st.divider()