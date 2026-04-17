"""
AEDA-AI — Streamlit Interface

Main entry point for the web interface. Run with:
    streamlit run app/main.py
"""

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


def main():
    # Sidebar navigation
    st.sidebar.title("AEDA-AI")
    st.sidebar.caption("Automated Exploratory Data Analysis\nfor Environmental Data")
    st.sidebar.divider()

    page = st.sidebar.radio(
        "Navigation",
        options=["Upload & Configure", "Analysis Plan", "Results", "Depth Profiles"],
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


if __name__ == "__main__":
    main()
