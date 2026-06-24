"""
Page 2: Analysis Plan

Shows the auto-selector's dataset profile, recommendations, warnings,
and analysis scales. This is the "brain" of AEDA-AI made visible —
the user can see exactly WHY the system chose each method.
"""

import streamlit as st

from app.i18n import t


def render():
    from app.components.page_header import page_header

    page_header(
        title=t("Analysis Plan"),
        description=t("What the system decided to do with your dataset, and why."),
        icon=":material/explore:",
    )

    results = st.session_state.get("results")
    if results is None or results.plan is None:
        st.info(t("Run an analysis first from the Upload page."))
        return

    plan = results.plan
    profile = plan.profile

    # ---- Executive summary ----
    st.markdown(
        t(
            "Your dataset contains **{n} samples** and **{f} variables**. "
            "After preprocessing and dimensionality reduction, the system will work with approximately "
            "**{d} dimensions**. The following analysis was tailored to your data's "
            "statistical properties and suspected geochemical composition."
        ).format(
            n=profile.n_samples,
            f=profile.n_features,
            d=profile.effective_dimensionality,
        )
    )
    st.divider()

    # ---- Dataset profile ----
    st.subheader(t("Dataset profile"))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(t("Samples"), profile.n_samples)
    col2.metric(t("Variables"), profile.n_features)
    col3.metric(t("Effective dimensions"), profile.effective_dimensionality)
    col4.metric(t("Samples/features ratio"), f"{profile.ratio_samples_features:.1f}")

    # Distribution and structure metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(t("Skewed variables"), f"{profile.pct_skewed:.0f}%")
    col2.metric(t("Missing data"), f"{profile.pct_missing:.1f}%")
    col3.metric(t("Correlated pairs (|r|>0.6)"), profile.high_correlation_pairs)
    col4.metric(t("Outlier samples (IQR)"), f"{profile.pct_outliers_iqr:.0f}%")

    # ---- Geochemistry detected ----
    if profile.has_major_elements or profile.has_trace_elements:
        st.subheader(t("Geochemistry detected"))
        col1, col2 = st.columns(2)
        with col1:
            if profile.major_element_cols:
                st.markdown(f"**{t('Major elements')}** ({len(profile.major_element_cols)})")
                st.write(", ".join(profile.major_element_cols))
            if profile.heavy_metal_cols:
                st.markdown(f"**{t('Heavy metals')}** ({len(profile.heavy_metal_cols)})")
                st.write(", ".join(profile.heavy_metal_cols))
        with col2:
            if profile.trace_element_cols:
                st.markdown(f"**{t('Trace elements')}** ({len(profile.trace_element_cols)})")
                st.write(", ".join(profile.trace_element_cols))
            if profile.granulometry_cols:
                st.markdown(f"**{t('Granulometry')}**")
                st.write(", ".join(profile.granulometry_cols))

        if profile.mixed_units_detected:
            st.warning(t("Mixed units detected (% and mg/kg). Scaling is mandatory before multivariate analysis."))

    # ---- Warnings ----
    # NOTE: warning messages come from the expert engine (backend) and are shown as-is.
    if plan.warnings:
        st.subheader(t("Warnings"))
        for w in plan.warnings:
            st.warning(w)

    # ---- Analysis scales ----
    if plan.analysis_scales:
        st.subheader(t("Recommended analysis scales"))
        for scale in plan.analysis_scales:
            icon = "★" if scale.recommended else "○"
            st.markdown(f"**{icon} {scale.name}** — {scale.description}")
            if scale.reason:
                st.caption(scale.reason)

    # ---- Recommendations by category ----
    st.subheader(t("Method recommendations"))

    # (cat_key is the logic key; cat_name is the English display label, translated via t()).
    categories = [
        ("preprocessing", "Preprocessing"),
        ("dimensionality", "Dimensionality reduction"),
        ("clustering", "Clustering"),
        ("spatial", "Spatial analysis"),
        ("anomaly", "Anomaly detection"),
        ("correlation", "Correlations"),
        ("feature_analysis", "Feature analysis"),
    ]

    for cat_key, cat_name in categories:
        recs = plan.get_by_category(cat_key)
        if not recs:
            continue

        with st.expander(f"**{t(cat_name)}** ({len(recs)} {t('methods')})", expanded=(cat_key in ("preprocessing", "clustering"))):
            for rec in recs:
                priority = t("Primary") if rec.priority == 1 else t("Alternative")
                confidence = rec.confidence.value if hasattr(rec.confidence, "value") else str(rec.confidence)

                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{rec.method}**")
                    st.write(rec.reason)
                with col2:
                    st.caption(f"{t('Priority')}: {priority}")
                    st.caption(f"{t('Confidence')}: {confidence}")

                if rec.evidence:
                    for ev in rec.evidence:
                        st.caption(f"  · {ev}")

                if rec.params:
                        from app.components.params import render_params

                        render_params(rec.params)

                st.divider()

    # ---- Validation report ----
    if results.validation:
        with st.expander(t("Validation report")):
            v = results.validation
            st.write(f"{t('Completeness')}: {v.completeness_pct:.1f}%")
            st.write(f"{t('Issues found')}: {len(v.issues)}")
            for issue in v.issues:
                severity = issue.severity.value.upper()
                if severity == "ERROR":
                    st.error(f"**{issue.column}**: {issue.message}")
                elif severity == "WARNING":
                    st.warning(f"**{issue.column}**: {issue.message}")
                else:
                    st.info(f"**{issue.column}**: {issue.message}")
