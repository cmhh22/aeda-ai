"""
Page: Audit

A traceable record of what the pipeline actually did and why. The page is
designed for the scientific domain expert rather than for an ML engineer:
every automatic decision is shown in plain language, with the evidence that
drove it. Heavy ML jargon is in the "Technical" tab.

Tab structure:
1. Overview — run summary, validation, failures at a glance.
2. Decisions — auto-selector choices (what method, why, evidence).
3. Interpretation — environmental layer (EF, TEL/PEL, Birch, baseline).
4. Technical — preprocessing trace, ML metrics, under-the-hood details.
"""

import streamlit as st


def render():
    from app.components.page_header import page_header

    page_header(
        title="Audit",
        description=(
            "Trace of every decision the pipeline made on this dataset. "
            "Use this page to verify the methodology and defend each choice."
        ),
        icon="🔍",
    )

    results = st.session_state.get("results")
    if results is None:
        st.info("Run an analysis first from the Upload page.")
        return

    # ---- Tab structure ----
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Decisions", "Interpretation", "Technical"])

    with tab1:
        _render_overview(results)

    with tab2:
        _render_decisions(results)

    with tab3:
        _render_interpretation(results)

    with tab4:
        _render_technical(results)


# ============================================================
# TAB 2: DECISIONS
# ============================================================
def _render_decisions(results):
    """Auto-selector decisions with reasons and evidence."""
    plan = results.plan
    if plan is None:
        st.info("No analysis plan available.")
        return

    st.caption(
        "What the system chose to do, and why. Each entry shows the chosen "
        "method, the rationale, and the evidence from your dataset that "
        "supported the choice."
    )

    if plan.warnings:
        with st.expander(f"Plan-level warnings ({len(plan.warnings)})", expanded=True):
            for w in plan.warnings:
                st.warning(w)

    # Friendly category names — geochemistry-first, no ML jargon in headings
    category_labels = {
        "preprocessing": "Data preparation",
        "dimensionality": "Variable summarization",
        "clustering": "Sample grouping",
        "anomaly": "Anomaly detection",
        "correlation": "Variable relationships",
        "feature_analysis": "Most informative variables",
    }

    for cat_key, friendly_name in category_labels.items():
        recs = plan.get_by_category(cat_key)
        if not recs:
            continue
        primary = next((r for r in recs if r.priority == 1), recs[0])
        n_alternatives = len(recs) - 1

        confidence = (
            primary.confidence.value
            if hasattr(primary.confidence, "value")
            else str(primary.confidence)
        )

        with st.expander(
            f"**{friendly_name}** — chose **{primary.method}** "
            f"({confidence} confidence)",
            expanded=False,
        ):
            st.markdown(f"**Why:** {primary.reason or '(no reason recorded)'}")

            if primary.evidence:
                st.markdown("**Evidence from your data:**")
                for ev in primary.evidence:
                    st.write(f"· {ev}")

            if primary.params:
                st.caption("Parameters chosen by the auto-selector:")
                st.json(primary.params, expanded=False)

            if n_alternatives:
                st.caption(f"{n_alternatives} alternative method(s) were considered:")
                for alt in recs:
                    if alt is primary:
                        continue
                    st.write(f"· **{alt.method}** — {alt.reason}")


# ============================================================
# TAB 3: INTERPRETATION
# ============================================================
def _render_interpretation(results):
    """Environmental layer: EF, TEL/PEL, Birch, baseline strategy."""
    interp = results.interpretation
    if interp is None:
        st.info(
            "The interpretation layer (EF, TEL/PEL, Birch) was not executed — "
            "either no heavy metals were detected, the reference element was "
            "missing, or no depth column was available for the baseline."
        )
        return

    # ---- Configuration used ----
    diag = interp.diagnostics or {}
    ref = diag.get("reference_element", "—")
    strategy = diag.get("baseline_strategy", "—")
    n_metals = len(interp.metals_analyzed)
    n_samples = diag.get("n_samples", "—")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Reference element", ref)
    col2.metric("Baseline strategy", strategy or "not used")
    col3.metric("Metals analyzed", n_metals)
    col4.metric("Samples", n_samples)

    st.divider()

    if interp.ef_result is not None:
        n_baselines = len(interp.ef_result.baseline_concentrations)
        global_baseline = "__global__" in interp.ef_result.baseline_concentrations
        if global_baseline:
            st.write(
                "✓ EF was computed against a **single global baseline** "
                "(deepest sample in the dataset)."
            )
        else:
            st.write(
                f"✓ EF was computed against **per-site baselines** "
                f"({n_baselines} sites with their own deepest sample)."
            )

    st.markdown(f"**Metals analyzed:** {', '.join(interp.metals_analyzed)}")

    # ---- TEL/PEL classification breakdown ----
    st.markdown("**Toxicological classification (NOAA TEL/PEL):**")
    st.caption(
        "Each cell shows the number of samples in that toxicological category, "
        "per metal. Buchman (2008) and Long & MacDonald (1998)."
    )

    tel_pel_summary = _build_classification_summary(
        interp.tel_pel_classifications, interp.metals_analyzed
    )
    if tel_pel_summary is not None:
        st.dataframe(tel_pel_summary, use_container_width=True)

    st.divider()

    # ---- EF / Birch classification breakdown ----
    if interp.ef_classifications is not None:
        st.markdown("**Enrichment classification (Birch 2003):**")
        st.caption(
            "Number of samples in each enrichment band, per metal. "
            "EF computed relative to the deepest core section."
        )
        ef_summary = _build_classification_summary(
            interp.ef_classifications, interp.metals_analyzed
        )
        if ef_summary is not None:
            st.dataframe(ef_summary, use_container_width=True)

        # EF descriptive statistics
        with st.expander("Enrichment factor (EF) descriptive statistics per metal"):
            ef_stats = interp.ef_result.ef_values.describe().T
            st.dataframe(ef_stats, use_container_width=True)


def _build_classification_summary(classification_df, metals):
    """Build a wide table of class counts per metal."""
    import pandas as pd

    if classification_df is None or classification_df.empty:
        return None

    rows = []
    all_classes = set()
    counts_by_metal = {}
    for metal in metals:
        if metal not in classification_df.columns:
            continue
        counts = classification_df[metal].value_counts(dropna=False).to_dict()
        # Convert NaN keys to a string label
        counts = {("missing" if isinstance(k, float) else k): v for k, v in counts.items()}
        counts_by_metal[metal] = counts
        all_classes.update(counts.keys())

    if not counts_by_metal:
        return None

    ordered_classes = sorted(all_classes, key=str)
    for metal, counts in counts_by_metal.items():
        row = {"metal": metal}
        for cls in ordered_classes:
            row[cls] = int(counts.get(cls, 0))
        rows.append(row)

    return pd.DataFrame(rows).set_index("metal")


# ============================================================
# TAB 4: TECHNICAL
# ============================================================
def _render_technical(results):
    """Under-the-hood: preprocessing trace and ML quality metrics."""
    # ---- Preprocessing trace ----
    st.subheader("Preprocessing trace")
    st.caption(
        "Every transformation applied to the raw data, in order. "
        "This is the audit trail for reproducibility."
    )

    log = results.preprocessing_log
    if log is None or not log.steps:
        st.info("No preprocessing steps were recorded.")
    else:
        for i, step in enumerate(log.steps, 1):
            step_name = step.get("step", "?")
            details = {k: v for k, v in step.items() if k != "step"}
            with st.expander(f"Step {i}: **{step_name}**"):
                if details:
                    st.json(details, expanded=False)
                else:
                    st.caption("(no parameters recorded)")

    st.divider()

    # ---- ML Quality metrics ----
    st.subheader("ML quality metrics")
    st.caption(
        "These metrics evaluate how well the chosen models fit the data. "
        "They are useful for the analyst, not strictly necessary for "
        "scientific interpretation."
    )

    # PCA
    if results.dim_reduction is not None:
        st.markdown("**Dimensionality reduction**")
        n_comp = results.dim_reduction.n_components_selected
        total_var = results.dim_reduction.diagnostics.get(
            "total_variance_explained"
        )
        cols = st.columns(3)
        cols[0].metric("Method", results.dim_reduction.method)
        cols[1].metric("Components retained", n_comp)
        if total_var is not None:
            cols[2].metric("Cumulative variance", f"{total_var:.1%}")

    # Clustering
    if results.clustering is not None:
        st.markdown("**Clustering**")
        m = results.clustering.metrics or {}
        cols = st.columns(4)
        cols[0].metric("Method", results.clustering.method)
        cols[1].metric("Clusters", results.clustering.n_clusters)
        sil = m.get("silhouette")
        cols[2].metric(
            "Silhouette", f"{sil:.3f}" if sil is not None else "—",
            help="Range -1 to 1. Higher is better. Above 0.5 is good.",
        )
        db = m.get("davies_bouldin")
        cols[3].metric(
            "Davies-Bouldin", f"{db:.3f}" if db is not None else "—",
            help="Lower is better. Measures intra/inter-cluster ratio.",
        )

        diag = results.clustering.diagnostics or {}
        if diag.get("auto_selected"):
            compared = diag.get("compared_methods", [])
            if compared:
                st.caption("In auto mode the system compared:")
                for c in compared:
                    sil_c = c.get("silhouette")
                    sil_str = f"{sil_c:.3f}" if sil_c is not None else "—"
                    st.write(f"· {c.get('method')}: silhouette = {sil_str}")

    # Anomalies
    if results.anomalies is not None:
        st.markdown("**Anomaly detection**")
        cols = st.columns(2)
        cols[0].metric("Method", results.anomalies.method)
        cols[1].metric("Anomalies flagged", results.anomalies.n_anomalies)
