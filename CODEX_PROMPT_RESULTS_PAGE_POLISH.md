# CODEX_PROMPT_RESULTS_PAGE_POLISH

**Tipo:** UI/UX polish + small viz fixes (consolidado desde QA visual)
**Archivos:** 4 modificados
**Tiempo estimado:** ~25 min
**Tests esperados después:** 38 (sin cambios)

---

## 1. Contexto

Durante la verificación visual de la página **Results** sobre el dataset
ISOVIDA aparecieron varios issues. Ninguno bloquea el funcionamiento, pero
todos son visibles para un usuario científico y conviene cerrarlos antes
de pulir el resto de la app.

Issues a resolver, agrupados:

| # | Archivo | Problema |
|---|---|---|
| A1 | `app/views/upload.py` | Latitud/Longitud/No no se auto-excluyen → aparecen como flechas falsas en PCA, en la matriz de correlación y en los loadings |
| A2 | `app/views/upload.py` | Los 3 selectboxes de "Analysis options" no tienen descripción en lenguaje no técnico |
| B1 | `app/views/results.py` | Selectores X/Y axis del biplot sin tooltip que explique qué es un componente principal |
| B2 | `app/views/results.py` | "Color by" solo lista columnas categóricas → no se puede colorear por Profundidad |
| B3 | `app/views/results.py` | Las 3 métricas del tab Clustering (Clusters / Silhouette / Davies-Bouldin) no tienen tooltip explicativo |
| C1 | `aeda/viz/dimensionality.py` | Las anotaciones "80% / 90% / 95%" del scree plot se solapan con la leyenda |
| D1 | `aeda/viz/clustering.py` | En el scatter de clustering el título de leyenda "Clusters" se repite por cada cluster en lugar de aparecer una sola vez |

---

## 2. Cambio A1 — Auto-excluir coordenadas/identificadores

**Archivo:** `app/views/upload.py`

**Problema:** El cálculo actual de `suggested_exclude` solo toma columnas
no-numéricas. Pero `Latitud`, `Longitud` y `No` son numéricas (números
flotantes/enteros), así que entran al análisis ML y aparecen como variables
en el biplot de PCA, en la matriz de correlación, en la tabla de loadings
y como features en clustering — donde no aportan información química.

**BUSCAR:**

```python
    # Auto-detect likely non-measurement columns
    all_cols = preview_df.columns.tolist()
    non_numeric = preview_df.select_dtypes(exclude="number").columns.tolist()
    suggested_exclude = non_numeric
```

**REEMPLAZAR POR:**

```python
    # Auto-detect likely non-measurement columns. Two sources:
    # 1. Non-numeric columns (identifiers, dates, site names, etc.).
    # 2. Numeric columns that are NOT measurements (coordinates, row numbers,
    #    etc.). These are numeric but carry no chemical information and
    #    distort PCA, correlations and clustering if left in.
    all_cols = preview_df.columns.tolist()
    non_numeric = preview_df.select_dtypes(exclude="number").columns.tolist()

    # Common metadata column names across Spanish and English datasets
    METADATA_COLUMN_NAMES = {
        # Coordinates
        "Latitud", "Longitud", "Latitude", "Longitude", "Lat", "Lon", "Lng",
        "X_UTM", "Y_UTM", "UTM_X", "UTM_Y",
        # Row numbers / sample IDs that may be numeric
        "No", "N", "ID", "Id", "Sample_ID", "SampleID", "Sample_No", "Sample",
        "Order", "Row",
    }
    numeric_metadata = [c for c in all_cols if c in METADATA_COLUMN_NAMES]

    suggested_exclude = sorted(set(non_numeric) | set(numeric_metadata))
```

---

## 3. Cambio A2 — Descripciones bajo los 3 selectboxes de Upload

**Archivo:** `app/views/upload.py`

**Problema:** Los selectboxes muestran términos técnicos (`median`, `pca`,
`auto`) sin contexto. Un científico no-ML no sabe qué significan ni cuándo
elegir uno u otro. Hay que añadir una pequeña descripción explicativa.

**BUSCAR:**

```python
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
```

**REEMPLAZAR POR:**

```python
    col1, col2, col3 = st.columns(3)

    with col1:
        impute = st.selectbox(
            "Missing values strategy",
            options=["median", "mean", "knn", "drop_rows"],
            index=0,
            help="How to fill in or remove missing values.",
        )
        st.caption(
            "Replaces empty cells with a plausible value so ML algorithms can "
            "process the data. **Median** is robust against extreme values."
        )

    with col2:
        dim_method = st.selectbox(
            "Dimensionality reduction",
            options=["pca", "auto"],
            index=0,
            help="Method used to compress the dataset into a smaller number of components.",
        )
        st.caption(
            "Compresses many variables into a few summary axes (components) "
            "that capture the main patterns. **PCA** is the standard choice "
            "for environmental data."
        )

    with col3:
        cluster_method = st.selectbox(
            "Clustering method",
            options=["auto", "kmeans", "dbscan", "hierarchical"],
            index=0,
            help="Algorithm used to group similar samples.",
        )
        st.caption(
            "Groups samples with similar chemistry. **Auto** tries K-Means "
            "and DBSCAN and keeps the best one according to a quality score."
        )
```

---

## 4. Cambio B1 — Tooltips en X/Y axis del biplot

**Archivo:** `app/views/results.py`

**Problema:** Los selectores X axis y Y axis muestran números (1, 2, 3, 4...)
sin explicar qué representan. Un usuario no familiarizado con PCA no sabe
por qué cambiar de "2" a "3" muestra otro gráfico.

**BUSCAR:**

```python
            with col3:
                n_comp = results.dim_reduction.n_components_selected
                pc_x = st.selectbox("X axis", options=list(range(1, n_comp + 1)), index=0, key="pca_x")
                pc_y = st.selectbox("Y axis", options=list(range(1, n_comp + 1)), index=1, key="pca_y")
```

**REEMPLAZAR POR:**

```python
            with col3:
                n_comp = results.dim_reduction.n_components_selected
                pc_help = (
                    "Principal component to show on this axis. Components are "
                    "ordered by how much variability they capture: PC1 is the "
                    "most informative, then PC2, etc. Changing the axis shows "
                    "the dataset from a different angle."
                )
                pc_x = st.selectbox(
                    "X axis", options=list(range(1, n_comp + 1)),
                    index=0, key="pca_x", help=pc_help,
                )
                pc_y = st.selectbox(
                    "Y axis", options=list(range(1, n_comp + 1)),
                    index=1, key="pca_y", help=pc_help,
                )
```

---

## 5. Cambio B2 — Color by numérico (Profundidad)

**Archivo:** `app/views/results.py`

**Problema:** El dropdown "Color by" en el biplot de PCA actualmente solo
ofrece columnas categóricas (`Site_Name`, `Code`, etc.). Para sedimentos,
**colorear por profundidad** es la vista más reveladora (muestra el gradiente
temporal). Hay que permitir también columnas numéricas con escala continua.

**BUSCAR:**

```python
    raw_df = results.raw_data

    # ---- Top bar with key KPIs ----
```

(Si la línea anterior no aparece exacta, buscar la primera vez que se asigna
`raw_df = results.raw_data` después de `def render():`, justo antes del top
bar de KPIs.)

**REEMPLAZAR POR:**

```python
    raw_df = results.raw_data

    # Detect available columns to color the PCA biplot by. We allow both
    # categorical columns (sites, cores) and a small whitelist of numeric
    # metadata columns (depth, sample order) — these are ideal for showing
    # gradients (e.g. concentration vs. depth).
    NUMERIC_COLOR_WHITELIST = {
        "Profundidad", "Depth", "depth",
        "Fecha", "Date", "Year",
        "No", "N", "Order",
    }

    # ---- Top bar with key KPIs ----
```

Y dentro del **tab PCA**, BUSCAR:

```python
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                color_by = st.selectbox(
                    "Color by", options=["None"] + categorical_cols,
                    index=1 if categorical_cols else 0,
                    key="pca_color",
                )
```

REEMPLAZAR POR:

```python
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
```

---

## 6. Cambio B3 — Tooltips en métricas de Clustering

**Archivo:** `app/views/results.py`

**Problema:** Las 3 métricas del tab Clustering (`Clusters`, `Silhouette
score`, `Davies-Bouldin`) no tienen tooltip. El usuario ve números sin
saber qué significan ni si son buenos o malos.

**BUSCAR:**

```python
            # Cluster metrics
            metrics = results.clustering.metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Clusters", results.clustering.n_clusters)
            sil = metrics.get("silhouette")
            col2.metric("Silhouette score", f"{sil:.3f}" if sil else "—")
            db = metrics.get("davies_bouldin")
            col3.metric("Davies-Bouldin", f"{db:.3f}" if db else "—")
```

**REEMPLAZAR POR:**

```python
            # Cluster metrics with explanatory tooltips
            metrics = results.clustering.metrics
            col1, col2, col3 = st.columns(3)
            col1.metric(
                "Clusters", results.clustering.n_clusters,
                help="Number of groups the algorithm found in the data.",
            )
            sil = metrics.get("silhouette")
            col2.metric(
                "Silhouette score", f"{sil:.3f}" if sil else "—",
                help=(
                    "How well separated the clusters are. Range -1 to 1; "
                    "higher is better. Above 0.5 = good separation, "
                    "0.25–0.5 = moderate, below 0.25 = weak."
                ),
            )
            db = metrics.get("davies_bouldin")
            col3.metric(
                "Davies-Bouldin", f"{db:.3f}" if db else "—",
                help=(
                    "Average ratio of within-cluster to between-cluster "
                    "distances. Lower is better. Below 1.0 = good, "
                    "above 2.0 = weak."
                ),
            )
```

---

## 7. Cambio C1 — Scree plot: mover anotaciones de umbral

**Archivo:** `aeda/viz/dimensionality.py`

**Problema:** Las anotaciones "80% / 90% / 95%" del scree plot están
posicionadas a la derecha del gráfico (`x=1.02`) donde se solapan con la
leyenda de Plotly. Visualmente se ven encimadas y casi ilegibles.

**Solución:** moverlas al borde izquierdo del gráfico, donde hay espacio
libre. Y reducir el tamaño de fuente para que sean más discretas.

**BUSCAR:**

```python
    # Threshold reference lines on the cumulative axis
    for threshold in (80, 90, 95):
        fig.add_shape(
            type="line",
            x0=0, x1=1, xref="paper",
            y0=threshold, y1=threshold, yref="y2",
            line=dict(color="#AAAAAA", width=1, dash="dash"),
        )
        fig.add_annotation(
            text=f"{threshold}%",
            xref="paper", x=1.02,
            yref="y2", y=threshold,
            showarrow=False,
            font=dict(size=10, color="#888888"),
            xanchor="left",
        )
```

**REEMPLAZAR POR:**

```python
    # Threshold reference lines on the cumulative axis. The annotation labels
    # are placed at the *left* edge of the plot to avoid overlapping with the
    # Plotly legend on the right.
    for threshold in (80, 90, 95):
        fig.add_shape(
            type="line",
            x0=0, x1=1, xref="paper",
            y0=threshold, y1=threshold, yref="y2",
            line=dict(color="#AAAAAA", width=1, dash="dash"),
        )
        fig.add_annotation(
            text=f"{threshold}%",
            xref="paper", x=-0.02,
            yref="y2", y=threshold,
            showarrow=False,
            font=dict(size=9, color="#888888"),
            xanchor="right",
        )
```

---

## 8. Cambio D1 — Cluster scatter: leyenda no duplicada

**Archivo:** `aeda/viz/clustering.py`

**Problema:** Cada trace que se añade al scatter de clustering lleva el
parámetro `legendgrouptitle_text="Clusters"` (o el nombre del compare_with).
Plotly renderiza ese título **una vez por cada trace**, así que aparece
"Clusters / Cluster 0 / Clusters / Cluster 1 / Clusters / Cluster 2..."
en lugar de "Clusters / Cluster 0 / Cluster 1 / Cluster 2...".

**Solución:** asignar `legendgrouptitle_text` **solo al primer trace** de
cada grupo. Hay dos lugares donde corregir: en `cluster_scatter()` (donde
añade el ground truth) y en `_add_cluster_traces()` (donde añade los
clusters).

### 8.1 En la función `cluster_scatter` — ground truth traces

**BUSCAR:**

```python
    color_values = df.loc[scores.index, compare_with]
    color_map = get_categorical_colors(color_values.unique())
    for group, color in color_map.items():
        mask = color_values == group
        fig.add_trace(
            go.Scatter(
                x=scores.loc[mask, x_col], y=scores.loc[mask, y_col],
                mode="markers", name=str(group),
                legendgroup=str(group), legendgrouptitle_text=compare_with,
                marker=dict(size=7, color=color, line=dict(width=0.5, color="white")),
                hovertemplate=f"<b>{group}</b><br>{x_col}: %{{x:.2f}}<br>{y_col}: %{{y:.2f}}<extra></extra>",
            ),
            row=1, col=2,
        )
```

**REEMPLAZAR POR:**

```python
    color_values = df.loc[scores.index, compare_with]
    color_map = get_categorical_colors(color_values.unique())
    # The legend group title is set only on the first trace, otherwise Plotly
    # repeats it once per trace and the legend looks duplicated.
    for i, (group, color) in enumerate(color_map.items()):
        mask = color_values == group
        fig.add_trace(
            go.Scatter(
                x=scores.loc[mask, x_col], y=scores.loc[mask, y_col],
                mode="markers", name=str(group),
                legendgroup="ground_truth",
                legendgrouptitle_text=compare_with if i == 0 else None,
                marker=dict(size=7, color=color, line=dict(width=0.5, color="white")),
                hovertemplate=f"<b>{group}</b><br>{x_col}: %{{x:.2f}}<br>{y_col}: %{{y:.2f}}<extra></extra>",
            ),
            row=1, col=2,
        )
```

### 8.2 En la función `_add_cluster_traces` — cluster traces

**BUSCAR:**

```python
    for i, label in enumerate(unique_labels):
        mask = labels == label
        if label == -1:
            name = "Noise"
            color = "#AAAAAA"
            marker_opts = dict(size=6, color=color, symbol="x", line=dict(width=0.5))
        else:
            name = f"Cluster {label}"
            color = CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)]
            marker_opts = dict(size=8, color=color, line=dict(width=0.5, color="white"))

        trace = go.Scatter(
            x=scores.loc[mask, x_col], y=scores.loc[mask, y_col],
            mode="markers", name=name,
            legendgroup=name, legendgrouptitle_text="Clusters",
            marker=marker_opts,
            hovertemplate=f"<b>{name}</b><br>{x_col}: %{{x:.2f}}<br>{y_col}: %{{y:.2f}}<extra></extra>",
        )
```

**REEMPLAZAR POR:**

```python
    for i, label in enumerate(unique_labels):
        mask = labels == label
        if label == -1:
            name = "Noise"
            color = "#AAAAAA"
            marker_opts = dict(size=6, color=color, symbol="x", line=dict(width=0.5))
        else:
            name = f"Cluster {label}"
            color = CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)]
            marker_opts = dict(size=8, color=color, line=dict(width=0.5, color="white"))

        # The legend group title is set only on the first trace so Plotly
        # does not repeat "Clusters" once per cluster in the side legend.
        trace = go.Scatter(
            x=scores.loc[mask, x_col], y=scores.loc[mask, y_col],
            mode="markers", name=name,
            legendgroup="clusters",
            legendgrouptitle_text="Clusters" if i == 0 else None,
            marker=marker_opts,
            hovertemplate=f"<b>{name}</b><br>{x_col}: %{{x:.2f}}<br>{y_col}: %{{y:.2f}}<extra></extra>",
        )
```

---

## 9. Validación

```bash
# 1. Tests siguen verdes
pytest tests/ -q
```
**Esperado:** `38 passed`.

```bash
# 2. La app arranca sin errores
streamlit run app/main.py
```

**Verificación visual (subiendo ISOVIDA):**

- ✅ **Upload — Step 3:** Latitud, Longitud y No aparecen ya excluidos por defecto en el multiselect (no hace falta marcarlos a mano).
- ✅ **Upload — Step 4:** debajo de cada uno de los 3 selectboxes hay una descripción explicativa.
- ✅ **Results → PCA:** al pasar el mouse sobre "X axis" / "Y axis", aparece el tooltip explicando qué es un componente.
- ✅ **Results → PCA → Color by:** aparece `Profundidad` al final de la lista, después de las columnas categóricas. Al seleccionarla, los puntos se colorean en gradiente continuo.
- ✅ **Results → Clustering:** las 3 métricas (Clusters / Silhouette / Davies-Bouldin) muestran tooltip al hacer hover.
- ✅ **Results → PCA → Scree plot:** las etiquetas "80% / 90% / 95%" aparecen pegadas al borde izquierdo del gráfico, sin solaparse con la leyenda.
- ✅ **Results → Clustering scatter:** la leyenda lateral muestra "Clusters" como un único título seguido de Cluster 0, 1, 2... — sin la repetición.

---

## 10. Si algo falla

- Si el biplot crashea al elegir `Profundidad` como Color by → puede ser
  que `pca_biplot` en `aeda/viz/dimensionality.py` no acepte valores
  numéricos para colorear. Reportar el traceback y se resuelve con un
  prompt aparte.
- Si la leyenda del cluster scatter sigue duplicada → verificar que la
  versión de Plotly instalada soporta `legendgrouptitle_text=None`
  (debería en Plotly ≥ 5.x). Si no, alternativa: dejar `legendgrouptitle_text`
  fijo en todos pero usar `legendgroup="clusters"` igual para todos.
- No tocar `tests/`, `aeda/engine/`, ni `aeda/pipeline/`. Solo
  `app/views/upload.py`, `app/views/results.py`, `aeda/viz/dimensionality.py`,
  `aeda/viz/clustering.py`.

---

## 11. Mensaje de commit sugerido

```
fix(ui): polish Results page — auto-exclude metadata, tooltips, legend dedupe

Issues detected during QA on the ISOVIDA dataset:

- Upload: numeric metadata columns (Latitud, Longitud, No) were not
  auto-excluded from the ML analysis, polluting PCA biplot loadings,
  correlation matrix, and clustering features. Added a whitelist of
  common metadata column names that are now auto-excluded.

- Upload: added contextual captions under each of the 3 analysis option
  selectboxes so non-ML users understand what they are choosing.

- Results > PCA: added tooltips to X/Y axis selectors explaining what a
  principal component is.

- Results > PCA: Color by now also accepts numeric metadata columns
  (Profundidad), rendered as a continuous gradient. Essential to visualize
  temporal trends in sediment cores.

- Results > Clustering: added tooltips to the 3 metrics (Clusters,
  Silhouette, Davies-Bouldin) explaining what they mean and what is "good".

- Scree plot: threshold annotations (80/90/95%) moved from the right edge
  to the left edge of the plot to avoid overlap with the legend.

- Cluster scatter: legend group title "Clusters" was being repeated once
  per trace; now set only on the first trace per group.

No engine changes; 38 tests pass.
```
