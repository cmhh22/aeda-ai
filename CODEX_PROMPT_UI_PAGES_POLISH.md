# CODEX_PROMPT_UI_PAGES_POLISH

**Tipo:** UI/UX polish — content reorganization per page
**Archivos:** 6 modificados (todas las páginas de `app/pages/`)
**Tiempo estimado:** ~45 min
**Tests esperados después:** 38 (sin cambios — esto es UI, no toca el engine)
**Pre-requisito:** El prompt anterior `UI_FOUNDATION` debe haber sido aplicado.

---

## 1. Contexto

Tras el `UI_FOUNDATION`, la app ya tiene su identidad visual (paleta tierra,
sidebar branded, page_header consistente, error handling limpio, fix del bug
de tabs). Este prompt **reorganiza el contenido de cada página** para mejorar
la legibilidad y la jerarquía de información.

Cambios clave por página:

| Página | Cambio principal |
|---|---|
| **Upload** | Workflow numerado 1→5 con `subheader` por paso, preview en expander. |
| **Plan** | Executive summary narrativo arriba que sintetiza el plan en un párrafo. |
| **Results** | Top bar con 4 KPIs (Samples, Variables, Clusters, Anomalies). |
| **Depth Profiles** | Caption pedagógico (deeper = older) + presets reordenados. |
| **Audit** | 7 secciones planas → **4 tabs internos** (Overview / Decisions / Interpretation / Technical). |
| **Advanced** | `st.data_editor` para custom baseline + diff visualizado como tabla. |

Validado end-to-end con el dataset ISOVIDA real (273 muestras × 36 variables).
El executive summary del Plan, por ejemplo, renderiza algo como:

> "The dataset has **273 samples** and **36 variables**. The system recommends
> **CLR Transform (by subgroup)** for preprocessing, **PCA** for dimensionality
> reduction, **K-Means (K=7, geographic validation)** for clustering, **Isolation
> Forest** for anomaly detection. Notable signals: 6 heavy metals detected;
> compositional structure (CLR transform recommended); mixed units across
> variables."

---

## 2. `app/pages/upload.py` — REEMPLAZAR ÍNTEGRAMENTE

```python
"""
Page 1: Upload & Configure

The user uploads an Excel/CSV file, selects which columns to exclude from
analysis (IDs, codes, coordinates), configures basic options, and runs
the pipeline. Results are stored in session_state for the other pages.

The page is structured as a numbered 1→5 workflow so the user always knows
where they are in the process.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import tempfile
import time


def render():
    from app.components.page_header import page_header

    page_header(
        title="Upload & Configure",
        description="Upload your environmental dataset and run the analysis with one click.",
        icon="📤",
    )

    # ---- Step 1: File upload ----
    st.subheader("1. File")
    uploaded_file = st.file_uploader(
        "Select an Excel or CSV file",
        type=["xlsx", "xls", "csv"],
        help="The file should contain environmental measurements with samples as rows and variables as columns.",
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        st.info("Upload a file to begin.")
        _show_example()
        return

    # Save to temp file for the pipeline
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # ---- Step 2: Sheet selection (Excel only) ----
    sheet_name = None
    if suffix in (".xlsx", ".xls"):
        xls = pd.ExcelFile(tmp_path)
        if len(xls.sheet_names) > 1:
            st.subheader("2. Sheet")
            sheet_name = st.selectbox(
                "Select sheet",
                options=xls.sheet_names,
                help="Choose the sheet containing your measurement data.",
                label_visibility="collapsed",
            )
        else:
            sheet_name = xls.sheet_names[0]

    # ---- Preview data ----
    try:
        if suffix == ".csv":
            preview_df = pd.read_csv(tmp_path, nrows=10)
        else:
            preview_df = pd.read_excel(tmp_path, sheet_name=sheet_name, nrows=10)
    except Exception as e:
        from app.components.errors import show_error
        show_error(
            "Could not read the uploaded file. Make sure it is a valid Excel/CSV "
            "and that the selected sheet contains tabular data.",
            exc=e,
        )
        return

    with st.expander(f"Data preview ({preview_df.shape[1]} columns, first 10 rows)", expanded=False):
        st.dataframe(preview_df, use_container_width=True, height=300)

    # ---- Step 3: Column exclusion ----
    st.subheader("3. Columns to analyze")
    st.caption(
        "By default, non-numeric columns (identifiers, codes, dates, sites) "
        "are excluded. Adjust below if needed."
    )

    # Auto-detect likely non-measurement columns
    all_cols = preview_df.columns.tolist()
    non_numeric = preview_df.select_dtypes(exclude="number").columns.tolist()
    suggested_exclude = non_numeric

    exclude_cols = st.multiselect(
        "Exclude these columns from the ML analysis",
        options=all_cols,
        default=suggested_exclude,
        help="These columns will be ignored during the ML analysis. Coordinates and depth are excluded from ML but used for metadata.",
        label_visibility="collapsed",
    )

    # ---- Step 4: Analysis options ----
    st.subheader("4. Analysis options")
    st.caption(
        "Sensible defaults work for most environmental datasets — for fine-grained "
        "control, use the Advanced Configuration page after the first run."
    )
    col1, col2, col3 = st.columns(3)

    with col1:
        impute = st.selectbox(
            "Missing values strategy",
            options=["median", "mean", "knn", "drop_rows"],
            index=0,
            help="How to handle remaining missing values after filtering.",
        )

    with col2:
        dim_method = st.selectbox(
            "Dimensionality reduction",
            options=["pca", "auto"],
            index=0,
            help="PCA is recommended for most environmental datasets.",
        )

    with col3:
        cluster_method = st.selectbox(
            "Clustering method",
            options=["auto", "kmeans", "dbscan", "hierarchical"],
            index=0,
            help="'auto' tries K-Means and DBSCAN, picks the best.",
        )

    # ---- Step 5: Run pipeline ----
    st.subheader("5. Run")
    if st.button("Run analysis", type="primary", use_container_width=True):
        _run_pipeline(tmp_path, sheet_name, exclude_cols, impute, dim_method, cluster_method, uploaded_file.name)


def _run_pipeline(filepath, sheet_name, exclude_cols, impute, dim_method, cluster_method, filename):
    """Execute the AEDA pipeline with a progress bar."""
    from aeda.pipeline.runner import AEDAPipeline

    progress = st.progress(0, text="Loading data...")

    try:
        # Settings used in this run — persisted so the Advanced Configuration
        # page can pre-fill its controls with the values that were actually used.
        settings = {
            "scale_method": "auto",
            "impute_strategy": impute,
            "dim_method": dim_method,
            "clustering_method": cluster_method,
            "anomaly_method": "auto",
            "correlation_method": "compare",
            "apply_clr": False,
            "contamination": 0.05,
            "run_interpretation": True,
            "reference_element": "Al",
            "baseline_strategy": "deepest",
            "custom_baseline": None,
            "dim_kwargs": {},
            "clustering_kwargs": {},
            "anomaly_kwargs": {},
        }

        pipeline = AEDAPipeline(
            impute_strategy=impute,
            dim_method=dim_method,
            clustering_method=cluster_method,
        )

        progress.progress(10, text="Loading and validating data...")
        time.sleep(0.3)

        progress.progress(20, text="Running auto-selector...")
        time.sleep(0.2)

        progress.progress(30, text="Preprocessing...")
        results = pipeline.run(
            filepath,
            exclude_cols=exclude_cols,
            sheet_name=sheet_name,
        )

        progress.progress(80, text="Generating results...")
        time.sleep(0.3)

        # Store in session state
        st.session_state.results = results
        st.session_state.raw_df = results.raw_data
        st.session_state.filename = filename
        st.session_state.run_context = {
            "tmp_path": filepath,
            "sheet_name": sheet_name,
            "exclude_cols": exclude_cols,
            "settings": settings,
        }

        progress.progress(100, text="Done!")
        time.sleep(0.5)
        progress.empty()

        st.success("Analysis complete! Navigate to the other pages to see results.")

        # Quick summary
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Samples", results.raw_data.shape[0])
        col2.metric("Variables analyzed", results.processed_data.shape[1] if results.processed_data is not None else "—")
        if results.dim_reduction:
            col3.metric("PCA components", results.dim_reduction.n_components_selected)
        if results.clustering:
            col4.metric("Clusters", results.clustering.n_clusters)

    except Exception as e:
        progress.empty()
        from app.components.errors import show_error
        show_error(
            "The pipeline could not complete. The dataset may have an unexpected "
            "structure or required columns may be missing.",
            exc=e,
        )


def _show_example():
    """Show usage instructions when no file is uploaded."""
    st.divider()
    st.subheader("How it works")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**1. Upload**")
        st.write("Upload an Excel or CSV with your environmental measurements.")
    with col2:
        st.markdown("**2. Configure**")
        st.write("Select which columns to exclude and choose analysis options.")
    with col3:
        st.markdown("**3. Explore**")
        st.write("Browse interactive plots: PCA biplot, clusters, correlations, depth profiles.")

    st.divider()
    st.caption("Supported formats: .xlsx, .xls, .csv — Datasets tested with FRX geochemistry, granulometry, and sediment data.")
```

---

## 3. `app/pages/plan.py` — REEMPLAZAR ÍNTEGRAMENTE

```python
"""
Page 2: Analysis Plan

Shows the auto-selector's dataset profile, recommendations, warnings,
and analysis scales. This is the "brain" of AEDA-AI made visible —
the user can see exactly WHY the system chose each method.

The page leads with an executive summary that synthesizes the most
important decisions in a single paragraph, then drills down into the
profile, geochemistry, scales and recommendations.
"""

import streamlit as st


def render():
    from app.components.page_header import page_header

    page_header(
        title="Analysis Plan",
        description="What the system decided to do with your dataset, and why.",
        icon="🧭",
    )

    results = st.session_state.get("results")
    if results is None or results.plan is None:
        st.info("Run an analysis first from the Upload page.")
        return

    plan = results.plan
    profile = plan.profile

    # ---- Executive summary ----
    _render_executive_summary(plan, profile)

    # ---- Dataset profile (grouped by theme) ----
    st.subheader("Dataset profile")

    st.caption("Size and shape")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Samples", profile.n_samples)
    col2.metric("Variables", profile.n_features)
    col3.metric("Effective dimensions", profile.effective_dimensionality)
    col4.metric("Samples/features ratio", f"{profile.ratio_samples_features:.1f}")

    st.caption("Distribution and quality")
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
    st.caption(
        "Each category lists every method considered, with the priority and "
        "confidence assigned by the auto-selector."
    )

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


def _render_executive_summary(plan, profile):
    """Synthesize the plan's most important decisions into a single paragraph.

    The summary is built dynamically from the plan and the profile, so it
    always reflects the actual run — not a hard-coded template.
    """
    bits: list[str] = []

    # Dataset shape
    bits.append(
        f"The dataset has **{profile.n_samples} samples** and "
        f"**{profile.n_features} variables**."
    )

    # Primary recommendations per category — we pull the first priority entry
    primary = {}
    for cat in ("preprocessing", "dimensionality", "clustering", "anomaly"):
        recs = plan.get_by_category(cat)
        if recs:
            primary[cat] = next((r for r in recs if r.priority == 1), recs[0])

    chosen = []
    if primary.get("preprocessing"):
        chosen.append(f"**{primary['preprocessing'].method}** for preprocessing")
    if primary.get("dimensionality"):
        chosen.append(f"**{primary['dimensionality'].method}** for dimensionality reduction")
    if primary.get("clustering"):
        chosen.append(f"**{primary['clustering'].method}** for clustering")
    if primary.get("anomaly"):
        chosen.append(f"**{primary['anomaly'].method}** for anomaly detection")

    if chosen:
        bits.append("The system recommends " + ", ".join(chosen) + ".")

    # Geochemistry signal
    geo_bits = []
    if profile.has_heavy_metals:
        n_hm = len(profile.heavy_metal_cols or [])
        geo_bits.append(f"{n_hm} heavy metals detected")
    if profile.is_compositional:
        geo_bits.append("compositional structure (CLR transform recommended)")
    if profile.has_depth_gradient:
        geo_bits.append("a depth gradient suggesting historical contamination")
    if profile.mixed_units_detected:
        geo_bits.append("mixed units across variables")
    if geo_bits:
        bits.append("Notable signals: " + "; ".join(geo_bits) + ".")

    # Render as info card
    summary_text = " ".join(bits)
    st.info(summary_text, icon="📋")
    st.write("")  # small spacer
```

---

## 4. `app/pages/audit.py` — REEMPLAZAR ÍNTEGRAMENTE

Cambio estructural mayor: las 7 secciones planas se convierten en 4 tabs.

```python
"""
Page: Audit

A traceable record of what the pipeline actually did and why. The page is
designed for the scientific tutor (geochemistry domain expert) rather than
for an ML engineer: every automatic decision is shown in plain language,
with the evidence that drove it. Heavy ML jargon is kept in a single
expandable section at the bottom.

The page is organized into 4 tabs to keep it scannable:

- Overview: Run summary + input validation.
- Decisions: Auto-selector recommendations + preprocessing trace.
- Interpretation: Environmental geochemistry — reference element, baseline
  strategy, metals analyzed, TEL/PEL and Birch classification breakdowns.
  This is the tab the scientific tutor will spend the most time on.
- Technical: Failed steps + ML quality metrics (silhouette, DB index, etc.).
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

    tab_overview, tab_decisions, tab_interp, tab_tech = st.tabs([
        "Overview",
        "Decisions",
        "Interpretation",
        "Technical",
    ])

    with tab_overview:
        _render_run_summary(results)
        st.divider()
        _render_validation(results)

    with tab_decisions:
        _render_brain_decisions(results)
        st.divider()
        _render_preprocessing_trace(results)

    with tab_interp:
        _render_interpretation_audit(results)

    with tab_tech:
        _render_failures(results)
        _render_ml_metrics(results)


# ============================================================
# 1. RUN SUMMARY
# ============================================================
def _render_run_summary(results):
    st.subheader("Run summary")

    info = results.dataset_info
    raw_df = results.raw_data
    filename = st.session_state.get("filename", "—")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("File", filename)
    col2.metric("Samples", raw_df.shape[0] if raw_df is not None else 0)
    col3.metric("Variables", raw_df.shape[1] if raw_df is not None else 0)

    n_measured = len(info.measurement_cols) if info else 0
    col4.metric("Measurement variables", n_measured)

    if info:
        meta_bits = []
        if info.site_col:
            n_sites = (
                raw_df[info.site_col].nunique() if info.site_col in raw_df.columns else 0
            )
            meta_bits.append(f"site column **{info.site_col}** ({n_sites} sites)")
        if info.depth_col:
            meta_bits.append(f"depth column **{info.depth_col}**")
        if info.coordinate_cols:
            meta_bits.append(
                f"coordinates **{', '.join(info.coordinate_cols)}**"
            )
        if meta_bits:
            st.write("Metadata detected: " + "; ".join(meta_bits) + ".")
        else:
            st.write("No site, depth, or coordinate columns were detected.")


# ============================================================
# 2. INPUT VALIDATION
# ============================================================
def _render_validation(results):
    st.subheader("Input validation")

    v = results.validation
    if v is None:
        st.info("Validation report not available.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Completeness", f"{v.completeness_pct:.1f}%")
    col2.metric("Issues found", len(v.issues))

    n_errors = sum(1 for i in v.issues if i.severity.value == "error")
    n_warnings = sum(1 for i in v.issues if i.severity.value == "warning")
    col3.metric("Errors / warnings", f"{n_errors} / {n_warnings}")

    if not v.issues:
        st.success("No data quality issues were detected.")
        return

    with st.expander(f"Issue details ({len(v.issues)})", expanded=n_errors > 0):
        for issue in v.issues:
            severity = issue.severity.value
            label = f"**{issue.column}** — {issue.message}"
            if severity == "error":
                st.error(label)
            elif severity == "warning":
                st.warning(label)
            else:
                st.info(label)


# ============================================================
# 3. AUTO-SELECTOR DECISIONS
# ============================================================
def _render_brain_decisions(results):
    st.subheader("Automatic decisions")
    st.caption(
        "What the system chose to do, and why. Each entry shows the chosen "
        "method, the rationale, and the evidence from your dataset that "
        "supported the choice."
    )

    plan = results.plan
    if plan is None:
        st.info("No analysis plan available.")
        return

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
# 4. PREPROCESSING TRACE
# ============================================================
def _render_preprocessing_trace(results):
    st.subheader("Preprocessing trace")
    st.caption(
        "Every transformation applied to the raw data, in order. "
        "This is the audit trail for reproducibility."
    )

    log = results.preprocessing_log
    if log is None or not log.steps:
        st.info("No preprocessing steps were recorded.")
        return

    for i, step in enumerate(log.steps, 1):
        step_name = step.get("step", "?")
        details = {k: v for k, v in step.items() if k != "step"}
        with st.expander(f"Step {i}: **{step_name}**"):
            if details:
                st.json(details, expanded=False)
            else:
                st.caption("(no parameters recorded)")


# ============================================================
# 5. ENVIRONMENTAL INTERPRETATION
# ============================================================
def _render_interpretation_audit(results):
    st.subheader("Environmental interpretation")

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

    if interp.ef_result is not None:
        n_baselines = len(interp.ef_result.baseline_concentrations)
        global_baseline = "__global__" in interp.ef_result.baseline_concentrations
        if global_baseline:
            st.write(
                "EF was computed against a **single global baseline** "
                "(deepest sample in the dataset)."
            )
        else:
            st.write(
                f"EF was computed against **per-site baselines** "
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
# 6. FAILED STEPS
# ============================================================
def _render_failures(results):
    """Surface any pipeline step that failed silently (caught in try/except)."""
    failures = []
    if results.dim_reduction is None:
        failures.append(("Dimensionality reduction", "did not produce a result"))
    if results.clustering is None:
        failures.append(("Clustering", "did not produce a result"))
    if results.anomalies is None:
        failures.append(("Anomaly detection", "did not produce a result"))
    if results.correlations is None:
        failures.append(("Correlation analysis", "did not produce a result"))
    if results.feature_importance is None and results.clustering is not None:
        failures.append(
            ("Feature importance", "did not run (clusters are available)")
        )

    st.subheader("Pipeline step status")

    if not failures:
        st.success("All pipeline steps completed successfully.")
        return

    st.caption(
        "These steps were skipped or failed silently during the run. "
        "Check the application logs for the underlying error."
    )
    for step_name, msg in failures:
        st.warning(f"**{step_name}**: {msg}")


# ============================================================
# 7. ML QUALITY METRICS
# ============================================================
def _render_ml_metrics(results):
    """Section with the ML-specific quality metrics."""
    st.divider()
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
```

---

## 5. `app/pages/results.py` — INSERTAR top bar de KPIs

Buscar el bloque que comienza con la línea `raw_df = results.raw_data` (cerca
del inicio de `render()`, justo después del `page_header`), y reemplazarlo por:

**BUSCAR:**

```python
    raw_df = results.raw_data

    # Detect available categorical columns for coloring
    categorical_cols = raw_df.select_dtypes(exclude="number").columns.tolist()
```

**REEMPLAZAR POR:**

```python
    raw_df = results.raw_data

    # ---- Top bar with key KPIs ----
    n_samples = raw_df.shape[0] if raw_df is not None else 0
    n_vars_processed = (
        results.processed_data.shape[1] if results.processed_data is not None else 0
    )
    n_clusters = results.clustering.n_clusters if results.clustering else None
    n_anomalies = results.anomalies.n_anomalies if results.anomalies else None

    cols = st.columns(4)
    cols[0].metric("Samples", n_samples)
    cols[1].metric("Variables analyzed", n_vars_processed)
    cols[2].metric("Clusters", n_clusters if n_clusters is not None else "—")
    cols[3].metric("Anomalies", n_anomalies if n_anomalies is not None else "—")
    st.write("")  # spacer before tabs

    # Detect available categorical columns for coloring
    categorical_cols = raw_df.select_dtypes(exclude="number").columns.tolist()
```

---

## 6. `app/pages/depth.py` — 2 cambios pequeños

### 6.1 Caption pedagógico al inicio

**BUSCAR:**

```python
    raw_df = results.raw_data
    info = results.dataset_info

    # Check if depth column exists
    if info is None or info.depth_col is None:
        st.warning("No depth column detected in this dataset. Depth profiles require a column like 'Profundidad' or 'Depth'.")
        return

    depth_col = info.depth_col
    site_col = info.site_col
    numeric_cols = sorted(raw_df.select_dtypes(include="number").columns.tolist())
```

**REEMPLAZAR POR:**

```python
    raw_df = results.raw_data
    info = results.dataset_info

    # Check if depth column exists
    if info is None or info.depth_col is None:
        st.warning("No depth column detected in this dataset. Depth profiles require a column like 'Profundidad' or 'Depth'.")
        return

    st.info(
        "In sediment cores, **deeper samples represent older deposits** — "
        "each profile reads as a temporal series, with the present at the top "
        "and the past at the bottom.",
        icon="🕰️",
    )

    depth_col = info.depth_col
    site_col = info.site_col
    numeric_cols = sorted(raw_df.select_dtypes(include="number").columns.tolist())
```

### 6.2 Reordenar presets en el grid

**BUSCAR** (dentro de `_render_grid`):

```python
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
```

**REEMPLAZAR POR:**

```python
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

    # Reorder so the most relevant preset is first
    preset_order = ["Heavy metals", "Major elements", "Sediment indicators", "Custom selection"]
    ordered_presets = {k: presets[k] for k in preset_order if k in presets}

    col1, col2 = st.columns([1, 2])
    with col1:
        preset = st.selectbox("Preset", options=list(ordered_presets.keys()))

    default_vars = (
        ordered_presets[preset] if preset != "Custom selection"
        else variable_options[:6]
    )
```

---

## 7. `app/pages/advanced.py` — 3 cambios

### 7.1 `_render_custom_baseline_editor` — reemplazar la función entera

**BUSCAR** la función `_render_custom_baseline_editor` completa (probablemente
~25 líneas, con un único `st.text_area` y un `json.loads`).

**REEMPLAZAR POR:**

```python
def _render_custom_baseline_editor(current, raw_df=None):
    """Allow the user to define custom baseline values per element.

    For the simple case (a single global baseline), an interactive table is
    used. For the per-site case (a dict of site → element → value), a JSON
    text area is offered as an escape hatch.
    """
    import json
    import pandas as pd

    st.markdown("**Custom baseline**")
    st.caption(
        "Define the baseline concentration for each element. The reference "
        "element (used for normalization) and every analyzed metal must be "
        "present. Use the JSON tab for the advanced case of per-site baselines."
    )

    # Detect whether the existing baseline is per-site or flat
    is_per_site = (
        isinstance(current, dict)
        and current
        and isinstance(next(iter(current.values()), None), dict)
    )

    tab_table, tab_json = st.tabs(["Table (one baseline)", "JSON (per-site)"])

    with tab_table:
        # Build initial dataframe from `current` (flat) or from numeric columns
        if isinstance(current, dict) and not is_per_site:
            initial = [
                {"element": k, "concentration": float(v)} for k, v in current.items()
            ]
        else:
            # Suggest a starting list of elements based on the dataset
            suggested = []
            if raw_df is not None:
                for c in ("Al", "Fe", "Pb", "Zn", "Cu", "Cr", "Ni", "As", "Cd", "Hg"):
                    if c in raw_df.columns:
                        suggested.append(c)
            if not suggested:
                suggested = ["Al", "Pb", "Zn"]
            initial = [{"element": e, "concentration": 0.0} for e in suggested]

        edited_df = st.data_editor(
            pd.DataFrame(initial),
            num_rows="dynamic",
            use_container_width=True,
            key="custom_baseline_editor",
            column_config={
                "element": st.column_config.TextColumn(
                    "Element",
                    help="Symbol of the element (e.g. Al, Pb, Zn).",
                    required=True,
                ),
                "concentration": st.column_config.NumberColumn(
                    "Concentration",
                    help="Baseline concentration in the same units as the dataset.",
                    min_value=0.0,
                    format="%.4f",
                    required=True,
                ),
            },
        )

        # Build flat dict from the edited table
        try:
            cleaned = edited_df.dropna(subset=["element"])
            cleaned = cleaned[cleaned["element"].astype(str).str.strip() != ""]
            flat_baseline = {
                str(row["element"]).strip(): float(row["concentration"])
                for _, row in cleaned.iterrows()
            }
            if flat_baseline:
                st.caption(f"Baseline defined for {len(flat_baseline)} element(s).")
                return flat_baseline
            return None
        except (ValueError, TypeError) as e:
            st.error(f"Invalid concentration value: {e}")
            return None

    with tab_json:
        st.caption(
            "For per-site baselines, provide a JSON object mapping each site "
            'to its own element → concentration dict. Example: '
            '`{"site1": {"Al": 8.2, "Pb": 14.0}, "site2": {...}}`'
        )
        default_text = (
            json.dumps(current, indent=2)
            if is_per_site
            else '{\n  "site1": {\n    "Al": 0.0,\n    "Pb": 0.0\n  }\n}'
        )
        raw_text = st.text_area(
            "Per-site baseline JSON",
            value=default_text,
            height=200,
            key="custom_baseline_json",
        )
        try:
            parsed = json.loads(raw_text) if raw_text.strip() else None
            if parsed and is_per_site:
                st.success(f"Parsed baselines for {len(parsed)} sites.")
            return parsed
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
            return None
```

### 7.2 Pasar `raw_df` al editor

**BUSCAR:**

```python
    if out["baseline_strategy"] == "user":
        out["custom_baseline"] = _render_custom_baseline_editor(
            settings.get("custom_baseline")
        )
    else:
        out["custom_baseline"] = None

    return out
```

**REEMPLAZAR POR:**

```python
    if out["baseline_strategy"] == "user":
        out["custom_baseline"] = _render_custom_baseline_editor(
            settings.get("custom_baseline"), raw_df=raw_df,
        )
    else:
        out["custom_baseline"] = None

    return out
```

### 7.3 Diff visualizado como tabla + helper `_format_value`

**BUSCAR:**

```python
    # ---- Diff vs. last run ----
    diff = _settings_diff(settings, new_settings)
    if diff:
        with st.expander(f"Changes vs. last run ({len(diff)} parameter(s))", expanded=True):
            for key, (old, new) in diff.items():
                st.write(f"· **{key}**: `{old}` → `{new}`")
    else:
        st.caption("No changes vs. the last run.")
```

**REEMPLAZAR POR:**

```python
    # ---- Diff vs. last run ----
    diff = _settings_diff(settings, new_settings)
    if diff:
        import pandas as pd

        st.subheader(f"Changes vs. last run ({len(diff)} parameter(s))")
        diff_rows = [
            {"Parameter": key, "Previous": _format_value(old), "New": _format_value(new)}
            for key, (old, new) in diff.items()
        ]
        st.dataframe(
            pd.DataFrame(diff_rows),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("No changes vs. the last run.")
```

Y agregar el helper `_format_value` justo después de `_settings_diff` (al
final del archivo, en la sección Helpers).

**BUSCAR:**

```python
def _settings_diff(old: dict, new: dict) -> dict:
    """Return only the keys whose value changed, as {key: (old, new)}."""
    diff = {}
    keys = set(old.keys()) | set(new.keys())
    for k in sorted(keys):
        if old.get(k) != new.get(k):
            diff[k] = (old.get(k), new.get(k))
    return diff
```

**REEMPLAZAR POR:**

```python
def _settings_diff(old: dict, new: dict) -> dict:
    """Return only the keys whose value changed, as {key: (old, new)}."""
    diff = {}
    keys = set(old.keys()) | set(new.keys())
    for k in sorted(keys):
        if old.get(k) != new.get(k):
            diff[k] = (old.get(k), new.get(k))
    return diff


def _format_value(value) -> str:
    """Render a setting value as a short, readable string for the diff table."""
    if value is None or value == "":
        return "—"
    if isinstance(value, dict):
        if not value:
            return "—"
        items = ", ".join(f"{k}={v}" for k, v in value.items())
        return items if len(items) <= 60 else items[:57] + "..."
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)
```

---

## 8. Validación

```bash
# 1. Tests siguen verdes (esto es UI puro)
pytest tests/ -q
```
**Esperado:** `38 passed`

```bash
# 2. Sintaxis y imports
python -c "
import sys
sys.path.insert(0, '.')
import importlib
for m in ['app.theme', 'app.components.page_header', 'app.components.errors',
         'app.main', 'app.pages.upload', 'app.pages.plan', 'app.pages.results',
         'app.pages.depth', 'app.pages.audit', 'app.pages.advanced']:
    importlib.import_module(m)
    print(f'OK  {m}')
"
```
**Esperado:** todas las líneas con `OK`.

```bash
# 3. Smoke visual
streamlit run app/main.py
```

**Verificación visual:**

- ✅ **Upload:** los 5 pasos se ven numerados con headings claros, el preview está plegado en un expander.
- ✅ **Plan:** arriba aparece el info card azul con el executive summary narrativo. Las métricas están agrupadas bajo "Size and shape" y "Distribution and quality".
- ✅ **Results:** top bar con 4 KPIs (Samples / Variables analyzed / Clusters / Anomalies) antes de los tabs.
- ✅ **Depth Profiles:** info card con icono 🕰️ explicando "deeper = older". Si hay metales pesados detectados, "Heavy metals" es el preset por defecto.
- ✅ **Audit:** **4 tabs en lugar de 7 secciones**. La tab "Interpretation" es donde el tutor va a estar.
- ✅ **Advanced (con baseline_strategy = "user"):** se muestra una tabla editable de Element/Concentration en lugar de un text area de JSON. El diff vs último run aparece como una tabla con columnas Parameter / Previous / New.

---

## 9. Si algo falla

- Si los 38 tests dejan de pasar → algo del engine fue tocado por error. No
  debería pasar; este prompt no toca nada bajo `aeda/`.
- Si la página Plan falla con `AttributeError: 'DataProfile' object has no
  attribute 'is_compositional'` → verificar que el atributo existe en
  `aeda/engine/auto_selector.py` (debería estar; está en línea 144 del archivo
  como `is_compositional: bool = False`).
- Si `st.data_editor` no aparece → la versión de Streamlit es muy vieja.
  Streamlit ≥ 1.23 lo soporta. El proyecto declara una versión moderna.
- No tocar `aeda/`, `tests/`, ni `pyproject.toml`. Solo `app/pages/*.py`.

---

## 10. Mensaje de commit sugerido

```
feat(ui): polish each page's content layout

Reorganizes the content of every Streamlit page for better readability and
information hierarchy, building on top of the visual foundation introduced
in the previous commit:

- Upload: numbered 1→5 workflow, preview folded into expander.
- Plan: dynamic executive summary at the top synthesizing the run in one
  paragraph; profile metrics grouped by theme (size / quality).
- Results: top bar with the 4 main KPIs above the tabs.
- Depth Profiles: pedagogical caption ("deeper = older"); presets reordered
  so the most relevant geochemistry preset is the default.
- Audit: 7 stacked sections converted into 4 internal tabs (Overview,
  Decisions, Interpretation, Technical) so the scientific tutor can jump
  directly to the geochemistry view.
- Advanced: custom baseline now uses an interactive st.data_editor table
  instead of a raw JSON textarea (per-site case still available via JSON
  tab); diff vs. last run rendered as a clean table.

No engine changes; 38 tests still pass.
```
