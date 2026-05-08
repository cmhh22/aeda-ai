"""
AEDA-AI — Streamlit Interface

Main entry point for the web interface. Run with:
    streamlit run app/main.py
"""

import sys
import os

# Ensure the project root is on sys.path so both `app` and `aeda` packages
# are importable regardless of the working directory when streamlit is launched.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

st.set_page_config(
    page_title="AEDA-AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Session state initialization
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
    st.session_state.run_context = None  # dict with tmp_path, sheet_name, exclude_cols, settings


def main():
    # Sidebar navigation
    st.sidebar.title("AEDA-AI")
    st.sidebar.caption("Automated Exploratory Data Analysis\nfor Environmental Data")
    st.sidebar.divider()

    page = st.sidebar.radio(
        "Navigation",
        options=[
            "Upload & Configure",
            "Analysis Plan",
            "Results",
            "Depth Profiles",
            "Audit",
            "Advanced Configuration",
        ],
        label_visibility="collapsed",
    )

    # Show status in sidebar
    if st.session_state.results is not None:
        st.sidebar.divider()
        st.sidebar.success(f"Dataset loaded: **{st.session_state.filename}**")
        r = st.session_state.results
        st.sidebar.metric("Samples", r.raw_data.shape[0])
        st.sidebar.metric("Variables", r.raw_data.shape[1])
        if r.clustering:
            st.sidebar.metric("Clusters found", r.clustering.n_clusters)
    else:
        st.sidebar.divider()
        st.sidebar.info("No dataset loaded yet.")

    # Route to page
    if page == "Upload & Configure":
        from app.pages.upload import render
        render()
    elif page == "Analysis Plan":
        from app.pages.plan import render
        render()
    elif page == "Results":
        from app.pages.results import render
        render()
    elif page == "Depth Profiles":
        from app.pages.depth import render
        render()
    elif page == "Audit":
        from app.pages.audit import render
        render()
    elif page == "Advanced Configuration":
        from app.pages.advanced import render
        render()


if __name__ == "__main__":
    main()
