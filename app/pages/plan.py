"""
Page 2: Analysis Plan

Shows the auto-selector's dataset profile, recommendations, warnings,
and analysis scales. This is the "brain" of AEDA-AI made visible —
the user can see exactly WHY the system chose each method.
"""

import streamlit as st


def render():
    st.header("Analysis Plan")

    results = st.session_state.get("results")
    if results is None or results.plan is None:
        st.info("Run an analysis first from the Upload page.")
        return

    plan = results.plan
    profile = plan.profile

    # ---- Dataset profile ----
    st.subheader("Dataset profile")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Samples", profile.n_samples)
    col2.metric("Variables", profile.n_features)
    col3.metric("Effective dimensions", profile.effective_dimensionality)
    col4.metric("Samples/features ratio", f"{profile.ratio_samples_features:.1f}")

    # Distribution and structure metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Skewed variables", f"{profile.pct_skewed:.0f}%")
    col2.metric("Missing data", f"{profile.pct_missing:.1f}%")
    col3.metric("Correlated pairs (|r|>0.7)", profile.high_correlation_pairs)
    col4.metric("Outlier samples (IQR)", f"{profile.pct_outliers_iqr:.0f}%")

    # ---- Geochemistry detected ----
    if profile.has_major_elements or profile.has_trace_elements:
        st.subheader("Geochemistry detected")
        col1, col2 = st.columns(2)
        with col1:
            if profile.major_element_cols:
                st.markdown(f"**Major elements** ({len(profile.major_element_cols)})")
                st.write(", ".join(profile.major_element_cols))
            if profile.heavy_metal_cols:
                st.markdown(f"**Heavy metals** ({len(profile.heavy_metal_cols)})")
                st.write(", ".join(profile.heavy_metal_cols))
        with col2:
            if profile.trace_element_cols:
                st.markdown(f"**Trace elements** ({len(profile.trace_element_cols)})")
                st.write(", ".join(profile.trace_element_cols))
            if profile.granulometry_cols:
                st.markdown("**Granulometry**")
                st.write(", ".join(profile.granulometry_cols))

        if profile.mixed_units_detected:
            st.warning("Mixed units detected (% and mg/kg). Scaling is mandatory before multivariate analysis.")

    # ---- Warnings ----
    if plan.warnings:
        st.subheader("Warnings")
        for w in plan.warnings:
            st.warning(w)

    # ---- Analysis scales ----
    if plan.analysis_scales:
        st.subheader("Recommended analysis scales")
        for scale in plan.analysis_scales:
            icon = "★" if scale.recommended else "○"
            st.markdown(f"**{icon} {scale.name}** — {scale.description}")
            if scale.reason:
                st.caption(scale.reason)

    # ---- Recommendations by category ----
    st.subheader("Method recommendations")

    categories = [
        ("preprocessing", "Preprocessing"),
        ("dimensionality", "Dimensionality reduction"),
        ("clustering", "Clustering"),
        ("anomaly", "Anomaly detection"),
        ("correlation", "Correlations"),
        ("feature_analysis", "Feature analysis"),
    ]

    for cat_key, cat_name in categories:
        recs = plan.get_by_category(cat_key)
        if not recs:
            continue

        with st.expander(f"**{cat_name}** ({len(recs)} methods)", expanded=(cat_key in ("preprocessing", "clustering"))):
            for rec in recs:
                priority = "Primary" if rec.priority == 1 else "Alternative"
                confidence = rec.confidence.value if hasattr(rec.confidence, "value") else str(rec.confidence)

                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{rec.method}**")
                    st.write(rec.reason)
                with col2:
                    st.caption(f"Priority: {priority}")
                    st.caption(f"Confidence: {confidence}")

                if rec.evidence:
                    for ev in rec.evidence:
                        st.caption(f"  · {ev}")

                if rec.params:
                    st.code(str(rec.params), language=None)

                st.divider()

    # ---- Validation report ----
    if results.validation:
        with st.expander("Validation report"):
            v = results.validation
            st.write(f"Completeness: {v.completeness_pct:.1f}%")
            st.write(f"Issues found: {len(v.issues)}")
            for issue in v.issues:
                severity = issue.severity.value.upper()
                if severity == "ERROR":
                    st.error(f"**{issue.column}**: {issue.message}")
                elif severity == "WARNING":
                    st.warning(f"**{issue.column}**: {issue.message}")
                else:
                    st.info(f"**{issue.column}**: {issue.message}")
