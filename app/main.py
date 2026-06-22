"""
AEDA-AI — Streamlit interface, main entry point.

Run with:
    streamlit run app/main.py

Responsibilities of this module:
- Bootstrap sys.path so both `aeda.*` and `app.*` are importable.
- Configure the Streamlit page (title, icon, layout).
- Apply the AEDA-AI theme (palette + CSS layer).
- Initialize session state.
- Render the sidebar (branding + navigation + dataset status).
- Route the selected page to its render() function.
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so that both `aeda.*` and `app.*`
# imports work regardless of how Streamlit is invoked (from the project root
# or from inside the app/ directory).
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from app.theme import apply_theme
from app.i18n import t, language_selector

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AEDA-AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
if "results" not in st.session_state:
    st.session_state.results = None
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "filename" not in st.session_state:
    st.session_state.filename = None
# Persisted run context — used by the Advanced Configuration page to re-run
# the pipeline on the same data with different settings, without forcing the
# user to upload the file again.
if "run_context" not in st.session_state:
    st.session_state.run_context = None
if "lang" not in st.session_state:
    st.session_state.lang = "es"

# ---------------------------------------------------------------------------
# Pages registry — single source of truth for navigation.
# Each entry: (label shown in sidebar, dotted import path of the page module)
# ---------------------------------------------------------------------------
PAGES = [
    ("Upload & Configure", "app.views.upload"),
    ("Analysis Plan", "app.views.plan"),
    ("Results", "app.views.results"),
    ("Depth Profiles", "app.views.depth"),
    ("Audit", "app.views.audit"),
    ("Advanced Configuration", "app.views.advanced"),
]


def _render_sidebar() -> str:
    """Render the sidebar (branding + navigation + status block).

    Returns the label of the currently selected page.
    """
    # Branding: 🔬 + AEDA-AI in a single styled row
    st.sidebar.markdown(
        '<div class="sidebar-brand">'
        '<span class="brand-icon">🔬</span>'
        '<span class="brand-name">AEDA-AI</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.caption(t("Automated EDA for environmental data"))
    st.sidebar.divider()

    # Navigation
    page_label = st.sidebar.radio(
        t("Navigation"),
        options=[label for label, _ in PAGES],
        format_func=lambda label: t(label),
        label_visibility="collapsed",
    )

    # Status block
    st.sidebar.divider()
    _render_status_block()

    # Language selector (compact, bottom corner)
    language_selector()

    return page_label


def _render_status_block() -> None:
    """Show dataset status in the sidebar — empty state or current dataset."""
    results = st.session_state.results
    if results is None:
        st.sidebar.markdown(
            f'<div class="status-empty">{t("No dataset loaded")}</div>',
            unsafe_allow_html=True,
        )
        return

    filename = st.session_state.filename or "—"
    st.sidebar.markdown(f"**{t('Current dataset')}**")
    st.sidebar.caption(f"📄 {filename}")

    cols = st.sidebar.columns(2)
    cols[0].metric(t("Samples"), results.raw_data.shape[0])
    cols[1].metric(t("Variables"), results.raw_data.shape[1])

    if results.clustering is not None:
        st.sidebar.metric(t("Clusters"), results.clustering.n_clusters)


def main() -> None:
    page_label = _render_sidebar()

    # Route to the selected page. Imports are lazy to keep page-load fast.
    page_module_path = dict(PAGES)[page_label]
    import importlib
    module = importlib.import_module(page_module_path)
    module.render()


if __name__ == "__main__":
    main()
