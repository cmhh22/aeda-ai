# CODEX_PROMPT_DEPTH_AND_AUDIT_POLISH

**Tipo:** UI/UX polish + bug fixes (consolidado tras QA de Depth Profiles y Audit)
**Archivos:** 3 modificados
**Tiempo estimado:** ~25 min
**Tests esperados después:** 38 (sin cambios)

---

## 1. Contexto

Durante la verificación visual sobre el dataset ISOVIDA aparecieron issues
en dos páginas:

### Depth Profiles

1. La lista desplegable de "Variable" / "Variables to plot" incluye basura:
   - Coordenadas (`Latitud`, `Longitud`) — ya tratadas para PCA, falta replicar acá.
   - Identificadores (`No`).
   - Columnas de **incertidumbre analítica** (`U_< 2 µm`, `U_2 < G < 63 µm`,
     `U_> 63 µm`, `U_PPI550`). Son el error de medición, no la concentración.
     Para el científico es confuso ver `U_Pb` o `U_< 2 µm` junto a `Pb`.

2. En el modo "Multi-variable grid", el nombre de cada variable aparece
   **dos veces** en cada subplot: una como título arriba (`As`, `Cr`, `Cu`)
   y otra como etiqueta del eje X abajo. Bug de layout: están seteando
   `subplot_titles` Y `update_xaxes(title_text=...)` al mismo tiempo.

3. El selector "Separate by core (Core)" muestra cores como `Delfinario (A)`,
   `Delfinario (B)` pero nunca se explica qué son `(A)` y `(B)`. Un usuario
   no sabe si son réplicas, cores distintos, profundidades distintas, etc.

### Audit

El tab **Overview** quedó **vacío** después del polish anterior — solo muestra
un `st.info("Coming soon: comprehensive audit trail summary")`. El problema es
que en el polish previo, Codex creó la estructura de los 4 tabs y las
funciones `_render_decisions`, `_render_interpretation`, `_render_technical`,
pero **omitió** tres funciones que también debían existir:

- `_render_run_summary(results)` — resumen de la corrida (archivo, samples, metadata)
- `_render_validation(results)` — completeness e issues del dataset
- `_render_failures(results)` — qué pasos del pipeline fallaron

El tab Overview debe mostrar las dos primeras, y el tab Technical debe
mostrar la tercera (además de lo que ya muestra).

Como la reescritura del archivo es lo más seguro (evita el riesgo de que
Codex inserte las funciones en el lugar equivocado), este prompt entrega
`app/views/audit.py` **íntegro**.

---

## 2. Cambio 1 — `app/views/depth.py`: filtrar lista de variables y caption de cores

**Archivo:** `app/views/depth.py`

### 2.1 BUSCAR

```python
    depth_col = info.depth_col
    site_col = info.site_col
    numeric_cols = sorted(raw_df.select_dtypes(include="number").columns.tolist())

    # Remove depth itself and other metadata from variable options
    variable_options = [c for c in numeric_cols if c != depth_col]
```

### 2.2 REEMPLAZAR POR

```python
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
```

### 2.3 BUSCAR

```python
    with col2:
        # Optional core column detection
        core_col = None
        possible_core_cols = [c for c in df.columns if c.lower() in ("core", "perfil", "profile")]
        if possible_core_cols:
            use_core = st.checkbox(f"Separate by core ({possible_core_cols[0]})", value=True)
            if use_core:
                core_col = possible_core_cols[0]
```

### 2.4 REEMPLAZAR POR

```python
    with col2:
        # Optional core column detection
        core_col = None
        possible_core_cols = [c for c in df.columns if c.lower() in ("core", "perfil", "profile")]
        if possible_core_cols:
            use_core = st.checkbox(
                f"Separate by core ({possible_core_cols[0]})",
                value=True,
                help=(
                    "When a site has multiple sediment cores (e.g. Core A, "
                    "Core B), this draws each core as a separate line. "
                    "Useful to check reproducibility between cores at the "
                    "same site."
                ),
            )
            if use_core:
                core_col = possible_core_cols[0]
```

---

## 3. Cambio 2 — `aeda/viz/profiles.py`: quitar etiquetas duplicadas del grid

**Archivo:** `aeda/viz/profiles.py`

**Problema:** En la función que genera el grid multi-variable
(`depth_profile_grid`), cada subplot recibe **dos veces** el nombre de la
variable: una como `subplot_titles` (arriba) y otra como `title_text` del
eje X (abajo). Hay que eliminar la del eje X y dejar solo la de arriba.

### 3.1 BUSCAR

```python
        fig.update_xaxes(title_text=variable, row=row, col=col, gridcolor="#F0F0F0")
        fig.update_yaxes(
            title_text=f"{depth_col} (cm)" if col == 1 else None,
            autorange="reversed",
            gridcolor="#F0F0F0",
            row=row, col=col,
        )
```

### 3.2 REEMPLAZAR POR

```python
        # The variable name is already shown as the subplot title (top of each
        # cell). Showing it again on the X-axis title would duplicate the
        # label visually. We keep the X grid styled but omit the title.
        fig.update_xaxes(row=row, col=col, gridcolor="#F0F0F0")
        fig.update_yaxes(
            title_text=f"{depth_col} (cm)" if col == 1 else None,
            autorange="reversed",
            gridcolor="#F0F0F0",
            row=row, col=col,
        )
```

---

## 4. Cambio 3 — `app/views/audit.py`: reescritura completa

**Archivo:** `app/views/audit.py`

**Acción:** reemplazar el **contenido íntegro** del archivo por el siguiente
código. Esto agrega las tres funciones que faltaban (`_render_run_summary`,
`_render_validation`, `_render_failures`) y restaura el contenido real del
tab Overview, además de añadir el bloque "Pipeline step status" al tab
Technical.

```python
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
    tab_overview, tab_decisions, tab_interpretation, tab_technical = st.tabs([
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
        _render_decisions(results)

    with tab_interpretation:
        _render_interpretation(results)

    with tab_technical:
        _render_technical(results)


# ============================================================
# TAB 1: OVERVIEW — Run summary
# ============================================================
def _render_run_summary(results):
    """High-level summary of the current run."""
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
# TAB 1: OVERVIEW — Input validation
# ============================================================
def _render_validation(results):
    """Validation report: completeness and data quality issues."""
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
    """Under-the-hood: preprocessing trace, pipeline status, ML metrics."""
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

    # ---- Pipeline step status ----
    _render_failures(results)

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


# ============================================================
# Pipeline step status (used inside the Technical tab)
# ============================================================
def _render_failures(results):
    """Report any pipeline step that did not produce a result."""
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
```

---

## 5. Validación

```bash
# 1. Tests siguen verdes
pytest tests/ -q
```
**Esperado:** `38 passed`.

```bash
# 2. Imports limpios
python -c "
import sys; sys.path.insert(0, '.')
import importlib
for m in ['app.views.audit', 'app.views.depth']:
    importlib.import_module(m)
    print(f'OK  {m}')
"
```
**Esperado:** las 2 líneas con `OK`.

```bash
# 3. Smoke visual
streamlit run app/main.py
```

**Verificación visual (con ISOVIDA cargado):**

- ✅ **Depth Profiles → Variable dropdown:** ya NO aparecen `No`, `Latitud`,
  `Longitud`, `U_< 2 µm`, `U_2 < G < 63 µm`, `U_> 63 µm`, `U_PPI550` en la lista.
- ✅ **Depth Profiles → checkbox "Separate by core":** al hacer hover sobre
  el "(?)" aparece el tooltip explicando qué son `(A)`, `(B)`.
- ✅ **Depth Profiles → Multi-variable grid:** en cada subplot el nombre
  de la variable aparece **una sola vez** (arriba como título), no duplicado.
- ✅ **Audit → tab Overview:** ya NO está vacío. Muestra "Run summary"
  (file, samples, variables, measurement variables + metadata detectada)
  y "Input validation" (completeness, issues).
- ✅ **Audit → tab Technical:** sigue mostrando Preprocessing trace y ML
  metrics, y AHORA también muestra "Pipeline step status" entre los dos
  (verde con "All pipeline steps completed successfully" o amarillo
  enumerando los pasos que fallaron).

---

## 6. Si algo falla

- Si **Audit Overview** sigue mostrando "Coming soon" después del cambio →
  el reemplazo del archivo no se guardó o se sobrescribió por error. Re-aplicar
  el contenido completo del bloque del paso 4.
- Si aparece `AttributeError` sobre algún campo de `results` (p.ej.
  `dataset_info`, `validation`, `preprocessing_log`) → el `AEDAResults`
  no expone alguno de esos campos. Reportar el traceback completo para
  evaluar caso por caso.
- Si en el grid de Depth alguna variable se ve sin etiqueta arriba →
  verificar que `subplot_titles=variables` sigue en `make_subplots(...)`.
  El cambio solo quita `update_xaxes(title_text=...)`, no toca el
  `subplot_titles`.
- No tocar `aeda/engine/`, `aeda/pipeline/`, `tests/`. Solo los 3 archivos
  mencionados.

---

## 7. Mensaje de commit sugerido

```
fix(ui): polish Depth Profiles + restore Audit Overview content

Depth Profiles:
- Variable selector now hides metadata (Latitud, Longitud, No) and analytical
  uncertainty columns (U_*). Only real measurements are listed.
- Multi-variable grid: removed the duplicated variable name on the X-axis
  title; the subplot title (top) is the single source of truth.
- "Separate by core" checkbox now has a tooltip explaining what (A)/(B)
  represent for the scientific user.

Audit:
- Restored the Overview tab content: previously displayed a placeholder
  "Coming soon" message. Now shows the Run Summary (file, samples,
  variables, metadata) and the Input Validation report (completeness,
  issues breakdown).
- Added "Pipeline step status" to the Technical tab so users can see at a
  glance whether all pipeline steps completed or some failed silently.
- All three previously-missing private functions (_render_run_summary,
  _render_validation, _render_failures) are now defined in app/views/audit.py.

No engine changes; 38 tests pass.
```
