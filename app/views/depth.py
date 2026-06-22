"""
Page 4: Depth Profiles

Dedicated page for depth-profile visualizations. Users select variables and
sites to display concentration-vs-depth plots, which in sediment studies
represent temporal trends (deeper = older).
"""

import streamlit as st

from app.i18n import t


def render():
    from app.components.page_header import page_header

    page_header(
        title=t("Depth Profiles"),
        description=t("Concentration vs. depth — sediment cores read as temporal series."),
        icon="🌊",
    )

    st.caption(
        t(
            "In sediment cores, **deeper = older**. The plots show how concentration "
            "of each variable changes through time, revealing historical contamination trends "
            "and geochemical changes."
        )
    )

    results = st.session_state.get("results")
    if results is None:
        st.info(t("Run an analysis first from the Upload page."))
        return

    raw_df = results.raw_data
    info = results.dataset_info

    # Check if depth column exists
    if info is None or info.depth_col is None:
        st.warning(t("No depth column detected in this dataset. Depth profiles require a column like 'Profundidad' or 'Depth'."))
        return

    depth_col = info.depth_col
    site_col = info.site_col
    numeric_cols = sorted(raw_df.select_dtypes(include="number").columns.tolist())

    # Filter the variable list shown to the user:
    # - Drop the depth column itself (used as Y axis).
    # - Drop common metadata columns (coordinates, row numbers).
    # - Drop "U_*" columns (analytical uncertainty / measurement error,
    #   not concentration values). These are confusing next to the
    #   actual measurements.
    METADATA_COLS = {
        "No", "N", "ID", "Id", "Sample", "Order", "Row",
        "Latitud", "Longitud", "Latitude", "Longitude",
        "Lat", "Lon", "Lng",
    }
    variable_options = [
        c for c in numeric_cols
        if c != depth_col
        and c not in METADATA_COLS
        and not c.startswith("U_")
    ]

    from aeda.viz.profiles import depth_profile, depth_profile_grid

    # ---- Mode selection ----
    # NOTE: option values stay in English (used in logic below); the label
    # shown to the user is translated via format_func.
    mode = st.radio(
        t("View mode"),
        options=["Single variable", "Multi-variable grid"],
        format_func=lambda o: t(o),
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
            t("Variable"),
            options=variable_options,
            index=variable_options.index("Pb") if "Pb" in variable_options else 0,
        )

    with col2:
        # Optional core column detection
        core_col = None
        possible_core_cols = [c for c in df.columns if c.lower() in ("core", "perfil", "profile")]
        if possible_core_cols:
            use_core = st.checkbox(
                t("Separate by core ({core})").format(core=possible_core_cols[0]),
                value=True,
                help=t(
                    "When a site has multiple sediment cores (e.g. Core A, "
                    "Core B), this draws each core as a separate line. "
                    "Useful to check reproducibility between cores at the "
                    "same site."
                ),
            )
            if use_core:
                core_col = possible_core_cols[0]

    # Site filter
    if site_col and site_col in df.columns:
        all_sites = sorted(df[site_col].unique())
        selected_sites = st.multiselect(
            t("Sites to display"),
            options=all_sites,
            default=all_sites,
            help=t("Deselect sites to simplify the plot."),
        )
        filtered_df = df[df[site_col].isin(selected_sites)] if selected_sites else df
    else:
        filtered_df = df

    # Include units from the dataset info when available
    info = st.session_state.get("results").dataset_info if st.session_state.get("results") else None
    units = info.units if info and hasattr(info, "units") else None
    fig = depth_profile(
        filtered_df,
        variable=variable,
        depth_col=depth_col,
        site_col=site_col,
        core_col=core_col,
        units=units,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Quick stats
    with st.expander(t("Variable statistics")):
        st.caption(t("Per-site descriptive statistics for the selected column."))
        stats = filtered_df.groupby(site_col)[variable].describe() if site_col else filtered_df[variable].describe()
        st.dataframe(stats, use_container_width=True)


def _render_grid(df, variable_options, depth_col, site_col, units=None):
    """Render a grid of depth profiles for multiple variables."""
    from aeda.viz.profiles import depth_profile_grid

    st.caption(
        t(
            "Compare several variables side-by-side. Each panel is **read top-to-bottom: "
            "0 cm is the most recent sediment, deeper rows are older**. A line that rises "
            "(toward the surface) means the concentration has increased over time at that "
            "site; a line that stays flat means the chemistry is stable. "
            "**Tip:** start with 3–4 variables to keep the figure readable, then add more."
        )
    )

    # Preset groups for convenience.
    # NOTE: preset keys stay in English (used in logic below); shown translated
    # via format_func on the selectbox.
    plan = st.session_state.results.plan if st.session_state.results else None

    presets = {}
    if plan and plan.profile.heavy_metal_cols:
        available_hm = [c for c in plan.profile.heavy_metal_cols if c in variable_options]
        if available_hm:
            presets["Heavy metals"] = available_hm
    if plan and plan.profile.major_element_cols:
        available_major = [c for c in plan.profile.major_element_cols if c in variable_options]
        if available_major:
            presets["Major elements"] = available_major
    if plan and plan.profile.ancillary_cols:
        available_sed = [c for c in plan.profile.ancillary_cols if c in variable_options]
        if available_sed:
            presets["Ancillary variables"] = available_sed
    presets["Custom selection"] = []

    col1, col2 = st.columns([1, 2])
    with col1:
        preset = st.selectbox(
            t("Preset"),
            options=list(presets.keys()),
            format_func=lambda o: t(o),
            help=t(
                "Pre-built variable groups based on your dataset's geochemistry. "
                "Pick **Custom** to choose any combination."
            ),
        )

    # For Heavy metals we start with the 4 NOAA priority metals to keep the
    # grid readable; the user can still add more from the multiselect.
    if preset == "Heavy metals":
        priority = [m for m in ["Pb", "Zn", "Cu", "As"] if m in presets[preset]]
        default_vars = priority if priority else presets[preset][:4]
    elif preset == "Custom selection":
        default_vars = variable_options[:4]
    else:
        default_vars = presets[preset][:6]

    with col2:
        variables = st.multiselect(
            t("Variables to plot"),
            options=variable_options,
            default=default_vars,
            help=t("2–9 variables work well; more than that and the panels get crowded."),
        )

    if len(variables) < 2:
        st.info(t("Select at least 2 variables."))
        return

    n_cols = st.slider(
        t("Columns in grid"), min_value=2, max_value=4,
        value=min(3, len(variables)),
        help=t("Fewer columns = wider panels = easier to compare individual sites."),
    )

    info = st.session_state.get("results").dataset_info if st.session_state.get("results") else None
    units = units or (info.units if info and hasattr(info, "units") else None)
    fig = depth_profile_grid(
        df,
        variables=variables,
        depth_col=depth_col,
        site_col=site_col,
        n_cols=n_cols,
        units=units,
    )
    st.plotly_chart(fig, use_container_width=True)
