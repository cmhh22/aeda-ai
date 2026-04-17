"""
Page 4: Depth Profiles

Dedicated page for depth-profile visualizations. Users select variables and
sites to display concentration-vs-depth plots, which in sediment studies
represent temporal trends (deeper = older).
"""

import streamlit as st


def render():
    st.header("Depth profiles")

    results = st.session_state.get("results")
    if results is None:
        st.info("Run an analysis first from the Upload page.")
        return

    raw_df = results.raw_data
    info = results.dataset_info

    # Check if depth column exists
    if info is None or info.depth_col is None:
        st.warning("No depth column detected in this dataset. Depth profiles require a column like 'Profundidad' or 'Depth'.")
        return

    depth_col = info.depth_col
    site_col = info.site_col
    numeric_cols = sorted(raw_df.select_dtypes(include="number").columns.tolist())

    # Remove depth itself and other metadata from variable options
    variable_options = [c for c in numeric_cols if c != depth_col]

    from aeda.viz.profiles import depth_profile, depth_profile_grid

    # ---- Mode selection ----
    mode = st.radio(
        "View mode",
        options=["Single variable", "Multi-variable grid"],
        horizontal=True,
    )

    if mode == "Single variable":
        _render_single(raw_df, variable_options, depth_col, site_col)
    else:
        _render_grid(raw_df, variable_options, depth_col, site_col)


def _render_single(df, variable_options, depth_col, site_col):
    """Render a single-variable depth profile with site and core selection."""
    from aeda.viz.profiles import depth_profile

    col1, col2 = st.columns([2, 1])

    with col1:
        variable = st.selectbox(
            "Variable",
            options=variable_options,
            index=variable_options.index("Pb") if "Pb" in variable_options else 0,
        )

    with col2:
        # Optional core column detection
        core_col = None
        possible_core_cols = [c for c in df.columns if c.lower() in ("core", "perfil", "profile")]
        if possible_core_cols:
            use_core = st.checkbox(f"Separate by core ({possible_core_cols[0]})", value=True)
            if use_core:
                core_col = possible_core_cols[0]

    # Site filter
    if site_col and site_col in df.columns:
        all_sites = sorted(df[site_col].unique())
        selected_sites = st.multiselect(
            "Sites to display",
            options=all_sites,
            default=all_sites,
            help="Deselect sites to simplify the plot.",
        )
        filtered_df = df[df[site_col].isin(selected_sites)] if selected_sites else df
    else:
        filtered_df = df

    fig = depth_profile(
        filtered_df,
        variable=variable,
        depth_col=depth_col,
        site_col=site_col,
        core_col=core_col,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Quick stats
    with st.expander("Variable statistics"):
        stats = filtered_df.groupby(site_col)[variable].describe() if site_col else filtered_df[variable].describe()
        st.dataframe(stats, use_container_width=True)


def _render_grid(df, variable_options, depth_col, site_col):
    """Render a grid of depth profiles for multiple variables."""
    from aeda.viz.profiles import depth_profile_grid

    # Preset groups for convenience
    plan = st.session_state.results.plan if st.session_state.results else None

    presets = {"Custom selection": []}
    if plan and plan.profile.heavy_metal_cols:
        available_hm = [c for c in plan.profile.heavy_metal_cols if c in variable_options]
        if available_hm:
            presets["Heavy metals"] = available_hm
    if plan and plan.profile.major_element_cols:
        available_major = [c for c in plan.profile.major_element_cols if c in variable_options]
        if available_major:
            presets["Major elements"] = available_major
    if plan and plan.profile.sediment_indicator_cols:
        available_sed = [c for c in plan.profile.sediment_indicator_cols if c in variable_options]
        if available_sed:
            presets["Sediment indicators"] = available_sed

    col1, col2 = st.columns([1, 2])
    with col1:
        preset = st.selectbox("Preset", options=list(presets.keys()))

    default_vars = presets[preset] if preset != "Custom selection" else variable_options[:6]

    with col2:
        variables = st.multiselect(
            "Variables to plot",
            options=variable_options,
            default=default_vars,
            help="Select 2-9 variables for the grid.",
        )

    if len(variables) < 2:
        st.info("Select at least 2 variables.")
        return

    n_cols = st.slider("Columns in grid", min_value=2, max_value=4, value=min(3, len(variables)))

    fig = depth_profile_grid(
        df,
        variables=variables,
        depth_col=depth_col,
        site_col=site_col,
        n_cols=n_cols,
    )
    st.plotly_chart(fig, use_container_width=True)
