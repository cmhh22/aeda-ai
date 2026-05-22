# CODEX_PROMPT_QA_PASS2_BUGS_AND_CAPTIONS

**Tipo:** QA pass 2 — bugs encontrados al validar el primer pass + mejoras de interpretabilidad
**Archivos:** 3 modificados
**Tiempo estimado:** 15 min
**Tests esperados después:** 71 passed

---

## 1. Contexto

Al hacer el smoke test del primer pass de fixes, aparecieron 2 bugs nuevos:

1. **Plan page:** texto duplicado en el intro ("Your dataset contains... variables. Your dataset contains... columns").
2. **Results page:** cuando el usuario re-corre con UMAP en Advanced, la tab sigue diciendo "PCA" y muestra "PCA was not executed or failed".

Además se decidió mejorar la interpretabilidad agregando **captions cortos sobre cada gráfica de Results** y **simplificando el grid multi-variable de Depth Profiles** (default a 4 metales prioritarios en lugar de 6).

---

## 2. FIX #1 — Texto duplicado en Plan intro

**Archivo:** `app/views/plan.py`

**Causa:** Codex en el commit anterior agregó una segunda frase sin borrar la original.

**BUSCAR** (puede tener variaciones — buscá la frase duplicada cerca de la línea 33):

```python
    st.markdown(f"""
    Your dataset contains **{profile.n_samples} samples** and **{profile.n_features} variables**. Your dataset contains **{profile.n_samples} samples** and **{profile.n_features} columns**.
    After preprocessing and dimensionality reduction, the system will work with approximately
    **{profile.effective_dimensionality} dimensions**. The following analysis was tailored to your data's
    statistical properties and suspected geochemical composition.
    """)
```

**REEMPLAZAR POR:**

```python
    st.markdown(f"""
    Your dataset contains **{profile.n_samples} samples** and **{profile.n_features} variables**.
    After preprocessing and dimensionality reduction, the system will work with approximately
    **{profile.effective_dimensionality} dimensions**. The following analysis was tailored to your data's
    statistical properties and suspected geochemical composition.
    """)
```

---

## 3. FIX #2 — Results: tab Dimensionality adaptativa a PCA/UMAP/t-SNE

**Archivo:** `app/views/results.py`

**Causa:** La primera tab está hardcoded a "PCA". Si el usuario cambia a UMAP/t-SNE en Advanced y hace Re-run, la tab sigue diciendo "PCA" y muestra el warning "PCA was not executed". Pero `results.dim_reduction` ya tiene el resultado de UMAP — solo falta renderizarlo.

**BUSCAR** (el bloque completo de la tab PCA — desde la declaración de las tabs hasta justo antes del comentario `# TAB 2: CORRELATIONS`):

```python
    tab_pca, tab_corr, tab_cluster, tab_anomaly, tab_surface = st.tabs([
        "PCA", "Correlations", "Clustering", "Anomalies", "Surface (spatial)"
    ])

    # ============================================================
    # TAB 1: PCA
    # ============================================================
    with tab_pca:
        if results.dim_reduction is None or results.dim_reduction.method != "PCA":
            st.warning("PCA was not executed or failed. Check the Analysis Plan for details.")
        else:
            from aeda.viz.dimensionality import pca_biplot, pca_scree_plot

            st.subheader("PCA biplot")

            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                # Build the color-by options: categorical columns first
                # (best for site/group comparisons), then numeric metadata
                # columns from the whitelist (best for gradients).
                numeric_options = [
                    c for c in raw_df.columns
                    if c in NUMERIC_COLOR_WHITELIST
                    and c in raw_df.select_dtypes(include="number").columns
                ]
                color_options = ["None"] + categorical_cols + numeric_options
                color_by = st.selectbox(
                    "Color by", options=color_options,
                    index=1 if categorical_cols else 0,
                    key="pca_color",
                    help=(
                        "Categorical columns (Site_Name, Core) show group "
                        "separation. Numeric columns (Profundidad) show a "
                        "continuous gradient — useful to see how chemistry "
                        "changes with depth."
                    ),
                )
            with col2:
                n_loadings = st.slider("Loading arrows", min_value=5, max_value=25, value=12, key="pca_loadings")
            with col3:
                n_comp = results.dim_reduction.n_components_selected
                pc_help = (
                    "Principal component to show on this axis. Components are "
                    "ordered by how much variability they capture: PC1 is the "
                    "most informative, then PC2, etc. Changing the axis shows "
                    "the dataset from a different angle."
                )
                pc_x = st.selectbox("X axis", options=list(range(1, n_comp + 1)), index=0, key="pca_x", help=pc_help)
                pc_y = st.selectbox("Y axis", options=list(range(1, n_comp + 1)), index=1, key="pca_y", help=pc_help)

            fig = pca_biplot(
                results.dim_reduction,
                df=raw_df,
                color_by=color_by if color_by != "None" else None,
                top_n_loadings=n_loadings,
                pc_x=pc_x,
                pc_y=pc_y,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Scree plot
            with st.expander("Scree plot (variance explained)"):
                fig_scree = pca_scree_plot(results.dim_reduction)
                st.plotly_chart(fig_scree, use_container_width=True)

            # Loadings table
            with st.expander("Loadings table"):
                loadings = results.dim_reduction.loadings
                if loadings is not None:
                    st.dataframe(
                        loadings.style.background_gradient(cmap="RdBu_r", axis=None, vmin=-1, vmax=1),
                        use_container_width=True,
                        height=400,
                    )
```

**REEMPLAZAR POR:**

```python
    # The first tab adapts to the dimensionality-reduction method used in
    # the current run. We give it a method-agnostic label so it does not
    # advertise PCA when the user re-ran the pipeline with UMAP or t-SNE.
    dim_method = (
        results.dim_reduction.method if results.dim_reduction is not None else "—"
    )
    tab_label_dim = f"Dimensionality ({dim_method})"
    tab_pca, tab_corr, tab_cluster, tab_anomaly, tab_surface = st.tabs([
        tab_label_dim, "Correlations", "Clustering", "Anomalies", "Surface (spatial)"
    ])

    # ============================================================
    # TAB 1: DIMENSIONALITY REDUCTION (PCA biplot / UMAP / t-SNE scatter)
    # ============================================================
    with tab_pca:
        if results.dim_reduction is None:
            st.warning(
                "Dimensionality reduction was not executed or failed. "
                "Check the Analysis Plan for details."
            )
        elif results.dim_reduction.method == "PCA":
            from aeda.viz.dimensionality import pca_biplot, pca_scree_plot

            st.subheader("PCA biplot")
            st.caption(
                "Each point is one sample. Samples close together have a similar "
                "chemical fingerprint. **Arrows** show how each variable pulls the "
                "samples — arrows pointing the same way mean those variables move "
                "together. Long arrows = the variable contributes a lot to the "
                "separation; short arrows = it contributes little."
            )

            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                numeric_options = [
                    c for c in raw_df.columns
                    if c in NUMERIC_COLOR_WHITELIST
                    and c in raw_df.select_dtypes(include="number").columns
                ]
                color_options = ["None"] + categorical_cols + numeric_options
                color_by = st.selectbox(
                    "Color by", options=color_options,
                    index=1 if categorical_cols else 0,
                    key="pca_color",
                    help=(
                        "Categorical columns (Site_Name, Core) show group "
                        "separation. Numeric columns (Profundidad) show a "
                        "continuous gradient — useful to see how chemistry "
                        "changes with depth."
                    ),
                )
            with col2:
                n_loadings = st.slider("Loading arrows", min_value=5, max_value=25, value=12, key="pca_loadings")
            with col3:
                n_comp = results.dim_reduction.n_components_selected
                pc_help = (
                    "Principal component to show on this axis. Components are "
                    "ordered by how much variability they capture: PC1 is the "
                    "most informative, then PC2, etc. Changing the axis shows "
                    "the dataset from a different angle."
                )
                pc_x = st.selectbox("X axis", options=list(range(1, n_comp + 1)), index=0, key="pca_x", help=pc_help)
                pc_y = st.selectbox("Y axis", options=list(range(1, n_comp + 1)), index=1, key="pca_y", help=pc_help)

            fig = pca_biplot(
                results.dim_reduction,
                df=raw_df,
                color_by=color_by if color_by != "None" else None,
                top_n_loadings=n_loadings,
                pc_x=pc_x,
                pc_y=pc_y,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Scree plot
            with st.expander("Scree plot (variance explained)"):
                st.caption(
                    "Each bar shows how much of the total variability is captured by that "
                    "principal component. The dashed lines mark the 80% / 90% / 95% "
                    "thresholds — the elbow of the curve is a useful place to stop adding "
                    "components without losing meaningful structure."
                )
                fig_scree = pca_scree_plot(results.dim_reduction)
                st.plotly_chart(fig_scree, use_container_width=True)

            # Loadings table
            with st.expander("Loadings table"):
                st.caption(
                    "How strongly each variable contributes to each principal component. "
                    "**Red** = positive contribution, **blue** = negative. Variables with "
                    "large absolute values on the same component move together; "
                    "opposite signs mean they move in opposite directions."
                )
                loadings = results.dim_reduction.loadings
                if loadings is not None:
                    st.dataframe(
                        loadings.style.background_gradient(cmap="RdBu_r", axis=None, vmin=-1, vmax=1),
                        use_container_width=True,
                        height=400,
                    )
        else:
            # Non-linear embedding (UMAP / t-SNE): no loadings, no scree —
            # render a clean 2D scatter and explain the differences from PCA.
            from aeda.viz.dimensionality import embedding_scatter

            method_name = results.dim_reduction.method
            st.subheader(f"{method_name} embedding")
            st.caption(
                f"{method_name} is a non-linear projection: distances on the plot reflect "
                "local similarity but the axes do not have a direct chemical meaning "
                "(unlike PCA, where PC1 and PC2 capture explicit variance). "
                "Useful for visualizing groups, but not for biplot-style interpretation."
            )

            color_options = ["None"] + categorical_cols
            color_by = st.selectbox(
                "Color by", options=color_options,
                index=1 if categorical_cols else 0,
                key="emb_color",
            )

            fig = embedding_scatter(
                results.dim_reduction,
                df=raw_df,
                color_by=color_by if color_by != "None" else None,
                title=f"{method_name} 2D embedding",
            )
            st.plotly_chart(fig, use_container_width=True)
```

---

## 4. FIX #3 — Captions interpretativos en Correlations tab

**Archivo:** `app/views/results.py`

**Causa:** Las tablas y heatmaps son ricos en información pero un científico no-ML no sabe qué buscar. Los captions le dicen qué leer en cada uno.

### 4.1 Caption en correlation_heatmap

**BUSCAR:**

```python
            from aeda.viz.correlations import correlation_heatmap, cross_correlation_heatmap

            st.subheader("Correlation matrix")

            col1, col2 = st.columns([1, 1])
```

**REEMPLAZAR POR:**

```python
            from aeda.viz.correlations import correlation_heatmap, cross_correlation_heatmap

            st.subheader("Correlation matrix")
            st.caption(
                "**Red** = the two variables tend to move together (rise and fall in sync). "
                "**Blue** = they move in opposite directions. **White** = no relationship. "
                "With **cluster-reorder axes** ON, similar variables are grouped along the "
                "diagonal, making blocks of co-varying elements (e.g. lithogenic vs. "
                "anthropogenic) easier to spot visually."
            )

            col1, col2 = st.columns([1, 1])
```

### 4.2 Captions en pairs, nonlinear y cross-correlation

**BUSCAR:**

```python
            # Significant pairs
            if hasattr(corr_result, "significant_pairs") and corr_result.significant_pairs:
                with st.expander(f"Significant correlations ({corr_result.n_strong} strong, {corr_result.n_moderate} moderate)"):
                    import pandas as pd
                    pairs_df = pd.DataFrame(corr_result.significant_pairs[:30])
                    st.dataframe(pairs_df, use_container_width=True)

            # Nonlinear candidates
            if isinstance(results.correlations, dict):
                nonlinear = results.correlations.get("nonlinear_candidates", [])
                if nonlinear:
                    with st.expander(f"Nonlinear relationship candidates ({len(nonlinear)})"):
                        import pandas as pd
                        nl_df = pd.DataFrame(nonlinear[:20])
                        st.dataframe(nl_df, use_container_width=True)
                        st.caption("These variable pairs have much higher Spearman than Pearson correlation, suggesting a nonlinear relationship.")

            # Cross-correlation
            plan = results.plan
            if plan and plan.profile.has_heavy_metals and plan.profile.has_granulometry:
                with st.expander("Heavy metals vs. grain size"):
                    fig_cross = cross_correlation_heatmap(
```

**REEMPLAZAR POR:**

```python
            # Significant pairs
            if hasattr(corr_result, "significant_pairs") and corr_result.significant_pairs:
                with st.expander(f"Significant correlations ({corr_result.n_strong} strong, {corr_result.n_moderate} moderate)"):
                    st.caption(
                        "Variable pairs sorted by the strength of their association. "
                        "**Strong** (|r| ≥ 0.7) means at least ~49% of their variability is "
                        "shared — common in geochemically associated elements (Pb–Zn from "
                        "industrial sources, Fe–Ti from lithogenic input). "
                        "**Moderate** (0.5 ≤ |r| < 0.7) suggests weaker but consistent links."
                    )
                    import pandas as pd
                    pairs_df = pd.DataFrame(corr_result.significant_pairs[:30])
                    st.dataframe(pairs_df, use_container_width=True)

            # Nonlinear candidates
            if isinstance(results.correlations, dict):
                nonlinear = results.correlations.get("nonlinear_candidates", [])
                if nonlinear:
                    with st.expander(f"Nonlinear relationship candidates ({len(nonlinear)})"):
                        st.caption(
                            "Pairs where Spearman (rank-based) is much higher than Pearson "
                            "(straight-line) correlation. This is a signal that the "
                            "relationship is monotonic but not linear — for example, a "
                            "saturation curve, or a threshold effect. Worth exploring with "
                            "a scatter plot before drawing conclusions."
                        )
                        import pandas as pd
                        nl_df = pd.DataFrame(nonlinear[:20])
                        st.dataframe(nl_df, use_container_width=True)

            # Cross-correlation
            plan = results.plan
            if plan and plan.profile.has_heavy_metals and plan.profile.has_granulometry:
                with st.expander("Heavy metals vs. grain size"):
                    st.caption(
                        "Spearman correlation between each heavy metal and each grain-size "
                        "fraction. A positive value with the **fine fraction (<2 µm)** is "
                        "the classical geochemical signal that clays are adsorbing the "
                        "metal. Strong positive correlation with the **coarse fraction "
                        "(>63 µm)** is unusual and may flag a specific source (e.g. "
                        "particulate contamination)."
                    )
                    fig_cross = cross_correlation_heatmap(
```

---

## 5. FIX #4 — Captions en Clustering tab

**Archivo:** `app/views/results.py`

### 5.1 Caption en cluster_scatter

**BUSCAR:**

```python
            from aeda.viz.clustering import cluster_scatter, cluster_composition

            st.subheader("Cluster analysis")

            compare_col = st.selectbox(
```

**REEMPLAZAR POR:**

```python
            from aeda.viz.clustering import cluster_scatter, cluster_composition

            st.subheader("Cluster analysis")
            st.caption(
                "**Left:** samples colored by the chemical groups the algorithm found "
                "automatically. **Right:** the same samples colored by a known label "
                "(e.g. site). If both panels look similar, chemistry and the chosen "
                "label agree — useful evidence that site-level differences are real. "
                "If they look different, chemistry is driven by something other than "
                "that label (depth, contamination, mineralogy)."
            )

            compare_col = st.selectbox(
```

### 5.2 Captions en composition y feature importance

**BUSCAR:**

```python
            # Composition chart
            if compare_col != "None":
                with st.expander("Cluster composition"):
                    fig_comp = cluster_composition(results.clustering, raw_df, compare_col)
                    st.plotly_chart(fig_comp, use_container_width=True)

            # Feature importance
            if results.feature_importance is not None:
                with st.expander("Variables that most discriminate between clusters"):
                    import pandas as pd
                    imp = results.feature_importance.importances
                    imp_df = pd.DataFrame({"Variable": imp.index, "Importance": imp.values})
                    st.bar_chart(imp_df.set_index("Variable").head(15))
```

**REEMPLAZAR POR:**

```python
            # Composition chart
            if compare_col != "None":
                with st.expander("Cluster composition"):
                    st.caption(
                        "How each cluster is composed in terms of the comparison label. "
                        "A **pure** bar (one dominant color) means that cluster maps cleanly "
                        "to one site / category. A **mixed** bar means samples from different "
                        "sites share the same chemical fingerprint — which often points to a "
                        "shared geochemical control (similar source, similar depositional "
                        "environment) overriding location."
                    )
                    fig_comp = cluster_composition(results.clustering, raw_df, compare_col)
                    st.plotly_chart(fig_comp, use_container_width=True)

            # Feature importance
            if results.feature_importance is not None:
                with st.expander("Variables that most discriminate between clusters"):
                    st.caption(
                        "Variables ranked by how useful they are to tell clusters apart. "
                        "Computed with permutation importance over a Random Forest, which "
                        "is robust to multicollinearity. The top variables are the "
                        "**chemical signature** that defines the groups — Yoelvis can focus "
                        "on these when interpreting what each cluster represents."
                    )
                    import pandas as pd
                    imp = results.feature_importance.importances
                    imp_df = pd.DataFrame({"Variable": imp.index, "Importance": imp.values})
                    st.bar_chart(imp_df.set_index("Variable").head(15))
```

---

## 6. FIX #5 — Captions en Anomalies tab

**Archivo:** `app/views/results.py`

### 6.1 Caption arriba del scatter

**BUSCAR:**

```python
        else:
            st.subheader("Anomaly detection")

            col1, col2 = st.columns(2)
```

**REEMPLAZAR POR:**

```python
        else:
            st.subheader("Anomaly detection")
            st.caption(
                "Samples flagged as **unusually different** from the rest of the dataset. "
                "These are not automatically 'contaminated' or 'wrong' — just statistical "
                "outliers in the multivariate chemical space. Each one deserves a look: "
                "they may be hotspots (real contamination), measurement errors, or "
                "samples from a chemically distinct sub-environment."
            )

            col1, col2 = st.columns(2)
```

### 6.2 Caption en tabla anomalous samples

**BUSCAR:**

```python
            # Anomaly details
            if results.anomalies.n_anomalies > 0:
                with st.expander("Anomalous samples"):
                    import pandas as pd
                    anomaly_idx = results.anomalies.anomaly_indices
                    if raw_df is not None and len(anomaly_idx) > 0:
                        anomaly_rows = raw_df.loc[anomaly_idx]
                        st.dataframe(anomaly_rows, use_container_width=True)
```

**REEMPLAZAR POR:**

```python
            # Anomaly details
            if results.anomalies.n_anomalies > 0:
                with st.expander("Anomalous samples"):
                    st.caption(
                        "All the columns from the original Excel for each flagged sample. "
                        "Use the site / depth / code columns to locate these samples in "
                        "the field and cross-check against laboratory notes."
                    )
                    import pandas as pd
                    anomaly_idx = results.anomalies.anomaly_indices
                    if raw_df is not None and len(anomaly_idx) > 0:
                        anomaly_rows = raw_df.loc[anomaly_idx]
                        st.dataframe(anomaly_rows, use_container_width=True)
```

---

## 7. FIX #6 — Simplificar grid de Depth Profiles

**Archivo:** `app/views/depth.py`

**Causa:** El grid empieza con 6 paneles de heavy metals que se ven apretados. Simplificamos a 4 metales prioritarios por default (Pb, Zn, Cu, As — los más relevantes para evaluación de contaminación) y agregamos caption arriba con cómo leer.

**BUSCAR** la función `_render_grid` completa (desde el `def`):

```python
def _render_grid(df, variable_options, depth_col, site_col, units=None):
    """Render a grid of depth profiles for multiple variables."""
    from aeda.viz.profiles import depth_profile_grid

    # Preset groups for convenience
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
        units=units,
    )
    st.plotly_chart(fig, use_container_width=True)
```

**REEMPLAZAR POR:**

```python
def _render_grid(df, variable_options, depth_col, site_col, units=None):
    """Render a grid of depth profiles for multiple variables."""
    from aeda.viz.profiles import depth_profile_grid

    st.caption(
        "Compare several variables side-by-side. Each panel is **read top-to-bottom: "
        "0 cm is the most recent sediment, deeper rows are older**. A line that rises "
        "(toward the surface) means the concentration has increased over time at that "
        "site; a line that stays flat means the chemistry is stable. "
        "**Tip:** start with 3–4 variables to keep the figure readable, then add more."
    )

    # Preset groups for convenience
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
            "Preset",
            options=list(presets.keys()),
            help=(
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
            "Variables to plot",
            options=variable_options,
            default=default_vars,
            help="2–9 variables work well; more than that and the panels get crowded.",
        )

    if len(variables) < 2:
        st.info("Select at least 2 variables.")
        return

    n_cols = st.slider(
        "Columns in grid", min_value=2, max_value=4,
        value=min(3, len(variables)),
        help="Fewer columns = wider panels = easier to compare individual sites.",
    )

    fig = depth_profile_grid(
        df,
        variables=variables,
        depth_col=depth_col,
        site_col=site_col,
        n_cols=n_cols,
        units=units,
    )
    st.plotly_chart(fig, use_container_width=True)
```

---

## 8. Validación

```bash
# 1. Sintaxis
python -c "
import ast
for f in ['app/views/results.py', 'app/views/depth.py', 'app/views/plan.py']:
    ast.parse(open(f).read())
    print(f'OK {f}')
"
```

```bash
# 2. Tests
pytest tests/ -q
```

**Esperado:** `71 passed`.

```bash
# 3. Smoke visual
streamlit run app/main.py
```

Pasos para validar manualmente:

1. **Carga ISOVIDA y corré con defaults.**
2. **Plan:** verificá que el intro dice "Your dataset contains 273 samples and 37 variables" **una sola vez** (no duplicado).
3. **Results:**
   - Tab dice "Dimensionality (PCA)" en lugar de "PCA"
   - Hay un caption explicativo bajo "PCA biplot"
   - Hay captions en Scree, Loadings, todos los expanders de Correlations, los expanders de Clustering, y la tabla de Anomalies
4. **Depth Profiles, modo grid:**
   - Hay un caption arriba que explica cómo leer
   - Default ahora son 4 metales (Pb, Zn, Cu, As) en lugar de 6
   - Los sliders y selectbox tienen `help` tooltips
5. **Ir a Advanced** → cambiar Dimensionality reduction de `pca` a `umap` → Re-run
6. **Volver a Results:**
   - La tab ahora dice **"Dimensionality (UMAP)"**
   - Muestra un scatter 2D de UMAP con color por sitio
   - Aparece un caption explicando que UMAP es no-lineal y no tiene loadings/scree
7. **Volver a Advanced** → cambiar a `pca` → Re-run → todo vuelve a la vista PCA original

---

## 9. Mensaje de commit sugerido

```
fix(ui): QA pass 2 — bugs from first pass + interpretability captions

Bugs:
- plan: remove duplicate intro sentence ("Your dataset contains..."
  appeared twice after the first QA pass touched the file).
- results: Dimensionality tab is now method-aware. When the user
  re-runs the pipeline with UMAP or t-SNE in Advanced, the tab label
  switches to "Dimensionality (UMAP)" / "(t-SNE)" and renders a 2D
  scatter via embedding_scatter() instead of incorrectly showing
  "PCA was not executed".

Interpretability:
- results: short Yoelvis-facing captions on every chart and expander
  (PCA biplot, scree, loadings; correlation heatmap, significant
  pairs, nonlinear candidates, heavy-metals × grain size; cluster
  scatter, composition, feature importance; anomaly scatter and
  details table).
- depth: grid mode now opens with a caption explaining the
  "top = recent / bottom = older" convention. The Heavy-metals
  preset defaults to 4 NOAA priority metals (Pb, Zn, Cu, As)
  instead of all 6, so the figure is readable from the start.
  Added tooltips on Preset, Variables and Columns controls.

All 71 tests pass.
```
