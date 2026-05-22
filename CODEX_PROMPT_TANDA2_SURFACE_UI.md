# CODEX_PROMPT_TANDA2_SURFACE_UI

**Tipo:** Feature UI — pestaña "Surface (spatial)" en Results (Yoelvis Tanda 2, etapa 2)
**Archivos:** 1 modificado, 1 nuevo
**Tiempo estimado:** ~15 min
**Tests esperados después:** 71 passed (sin cambios, esto es UI puro)
**Pre-requisito:** `CODEX_PROMPT_TANDA2_SURFACE_BACKEND.md` ya aplicado y commiteado.

---

## 1. Contexto

El backend del análisis espacial de la fracción superficial ya está
funcionando (commit anterior). El pipeline expone
`results.surface_analysis` con todo lo necesario para visualizar:
medias por sitio, clusters, coordenadas.

Esta etapa agrega la **pestaña "Surface (spatial)" en la página
Results**, al lado de PCA / Correlations / Clustering / Anomalies.

Contenido de la pestaña:

| Componente | Función |
|---|---|
| Dropdown 5 / 10 / 20 cm | Permite cambiar el corte de la capa superficial sin re-correr todo el pipeline (solo el módulo de surface, que es rápido) |
| KPI bar | Profundidad, sitios con datos, muestras totales, número de grupos |
| Heatmap sitio × variable | Z-score por columna, sitios ordenados por cluster |
| Scatter geográfico | Lat/Lon de cada sitio, coloreado por cluster (si hay coordenadas) |
| Tabla de composición | Qué sitios cayeron en qué cluster |

Validado contra ISOVIDA: 7 sitios, 54 muestras a 10cm, 3 grupos. Cambiar
a 5cm baja a 32 muestras, 7 sitios. El recálculo es instantáneo.

---

## 2. Archivo nuevo: `app/views/_surface_tab.py`

Crear este archivo con el contenido íntegro:

```python
"""Visualizations for the surface-layer spatial analysis.

This module renders the contents of the "Surface (spatial)" tab in the
Results page. It is deliberately self-contained so the tab can be added
to ``app/views/results.py`` with a single import + function call.

The data source is ``results.surface_analysis`` (a
``SurfaceAnalysisResult``) populated by the pipeline. The tab exposes:

- A depth selector (5 / 10 / 20 cm) that recomputes the analysis when
  the user changes the threshold — only the surface module re-runs, the
  rest of the pipeline output is reused.
- A KPI bar (sites, samples, clusters).
- A heatmap of standardized concentrations (site × variable), with sites
  reordered so cluster members appear next to each other.
- A geographic scatter when coordinates are available.
- A small table listing which sites fell into which cluster.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from aeda.engine.spatial_surface import (
    COMMON_SURFACE_DEPTHS_CM,
    DEFAULT_SURFACE_DEPTH_CM,
    surface_spatial_analysis,
)


def render_surface_tab(results) -> None:
    """Top-level entry point: renders the Surface (spatial) tab.

    Parameters
    ----------
    results : AEDAResults
        Pipeline results with ``surface_analysis`` populated (or None if
        the pipeline could not run it for this dataset).
    """
    st.subheader("Surface-layer inter-site analysis")
    st.caption(
        "Compares sites using **only their surface sediment** (the most "
        "recent deposition). This avoids mixing different historical "
        "periods across sites, which is the standard approach for "
        "present-day spatial contamination studies."
    )

    initial = results.surface_analysis
    if initial is None:
        st.info(
            "Surface analysis was not executed. It requires the dataset to "
            "have both a **site column** and a **depth column**. "
            "Check the Analysis Plan or Audit page to see which were detected."
        )
        return

    # ---- Depth selector ----------------------------------------------
    info = results.dataset_info
    raw_df = results.raw_data
    default_depth = float(initial.max_depth)
    presets = list(COMMON_SURFACE_DEPTHS_CM)
    try:
        default_index = presets.index(default_depth)
    except ValueError:
        default_index = presets.index(DEFAULT_SURFACE_DEPTH_CM)

    col1, col2 = st.columns([1, 3])
    with col1:
        selected_depth = st.selectbox(
            "Surface depth (cm)",
            options=presets,
            index=default_index,
            help=(
                "Defines the cutoff between 'surface' (recent) and 'deep' "
                "(older) sediment. Yoelvis (LEA-CEAC) recommends 10 cm as "
                "the default; 5 cm or 20 cm are common alternatives in "
                "different authors."
            ),
            key="surface_depth_selector",
        )

    # Recompute only if the user picked a different depth than the cached
    # pipeline result. The surface module is cheap; the rest of the
    # pipeline output is preserved.
    if float(selected_depth) != float(initial.max_depth):
        try:
            current = surface_spatial_analysis(
                raw_df,
                depth_col=info.depth_col,
                site_col=info.site_col,
                measurement_cols=info.measurement_cols,
                max_depth_cm=float(selected_depth),
                coordinate_cols=info.coordinate_cols or None,
            )
        except Exception as e:
            st.error(
                f"Could not recompute the surface analysis at {selected_depth} cm: "
                f"{type(e).__name__}: {e}"
            )
            return
    else:
        current = initial

    if current.n_samples_in_surface == 0:
        st.warning(
            f"No samples fall within the top {selected_depth} cm. "
            "Try a deeper threshold."
        )
        return

    # ---- KPI bar -----------------------------------------------------
    n_clusters = (
        current.site_clustering.get("n_clusters", 0)
        if current.site_clustering and current.site_clustering.get("labels") is not None
        else None
    )

    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Depth threshold", f"{selected_depth:g} cm")
    kpi_cols[1].metric("Sites with surface data", current.n_sites_with_data)
    kpi_cols[2].metric("Surface samples", current.n_samples_in_surface)
    kpi_cols[3].metric(
        "Site groups",
        n_clusters if n_clusters else "—",
        help=(
            "Number of clusters of sites with similar surface chemistry. "
            "Computed by hierarchical Ward clustering."
        ),
    )

    # ---- Heatmap: site × variable -----------------------------------
    _render_heatmap(current)

    # ---- Geographic map (if coordinates available) ------------------
    if current.site_coordinates is not None and not current.site_coordinates.empty:
        _render_geographic_scatter(current)

    # ---- Cluster composition table ----------------------------------
    _render_cluster_table(current)


# ----------------------------------------------------------------------
# Internal renderers
# ----------------------------------------------------------------------


def _render_heatmap(current) -> None:
    """Plot a Z-score heatmap of site means.

    Each variable is independently standardized (mean=0, std=1 across
    sites) so the heatmap shows *relative* enrichment per variable: a
    cell is red when that site has high values for that variable
    compared to the other sites, blue when it has low values.

    Sites are reordered so cluster members appear next to each other.
    """
    st.markdown("**Site × variable heatmap (Z-score per variable)**")
    st.caption(
        "Each variable is standardized across sites. **Red** = this site is high "
        "in that variable relative to other sites; **blue** = low. Sites are "
        "ordered by cluster, so members of the same group appear together."
    )

    site_means = current.site_means
    if site_means.empty:
        st.info("No site means to display.")
        return

    # Filter to columns with any variance — flat columns are uninformative
    stds = site_means.std(axis=0, numeric_only=True)
    informative_cols = stds[stds > 0].index.tolist()
    if not informative_cols:
        st.info("No variables show variance across sites.")
        return
    data = site_means[informative_cols]

    # Z-score per column
    z = (data - data.mean(axis=0)) / data.std(axis=0)

    # Reorder rows by cluster label (members of the same cluster together)
    if (
        current.site_clustering
        and current.site_clustering.get("labels") is not None
    ):
        labels = current.site_clustering["labels"]
        order = labels.sort_values().index.tolist()
        z = z.reindex(order)

    fig = px.imshow(
        z,
        labels=dict(x="Variable", y="Site", color="Z-score"),
        color_continuous_scale="RdBu_r",
        zmin=-2.5,
        zmax=2.5,
        aspect="auto",
    )
    fig.update_layout(
        height=max(280, 60 + 28 * len(z)),
        margin=dict(l=10, r=10, t=20, b=10),
        coloraxis_colorbar=dict(title="Z"),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_geographic_scatter(current) -> None:
    """Scatter sites on a lat/lon plane, colored by cluster.

    Uses Plotly's basic scatter (not Mapbox) because we don't want to
    depend on a tile server — the spatial pattern between sites is the
    point, not a real map.
    """
    coords = current.site_coordinates
    if coords is None or coords.empty:
        return

    # Try common name patterns to identify lat/lon columns
    lat_col = next(
        (c for c in coords.columns if c.lower() in ("lat", "latitud", "latitude")),
        coords.columns[0],
    )
    lon_col = next(
        (c for c in coords.columns if c.lower() in ("lon", "lng", "longitud", "longitude")),
        coords.columns[1] if len(coords.columns) > 1 else coords.columns[0],
    )

    st.markdown("**Geographic distribution of sites**")
    st.caption(
        "Each point is a site, positioned by its average coordinates. "
        "Color encodes the cluster from the surface-chemistry analysis: "
        "geographically close sites with the same color have similar "
        "surface chemistry."
    )

    df = coords.copy()
    df = df.rename_axis("site").reset_index()

    # Attach cluster label if available
    cluster_col = None
    if (
        current.site_clustering
        and current.site_clustering.get("labels") is not None
    ):
        labels = current.site_clustering["labels"]
        df["cluster"] = df["site"].map(labels).astype("Int64").astype(str)
        cluster_col = "cluster"

    fig = px.scatter(
        df,
        x=lon_col,
        y=lat_col,
        text="site",
        color=cluster_col,
        labels={lon_col: "Longitude", lat_col: "Latitude"},
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    fig.update_traces(
        textposition="top center",
        marker=dict(size=14, line=dict(width=1, color="white")),
    )
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=20, b=10),
        showlegend=cluster_col is not None,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_cluster_table(current) -> None:
    """Compact table: cluster → list of sites."""
    if (
        current.site_clustering is None
        or current.site_clustering.get("labels") is None
    ):
        st.caption(
            f"Clustering was skipped (only {current.n_sites_with_data} site(s)). "
            "It needs at least 3 sites with surface samples."
        )
        return

    labels = current.site_clustering["labels"]
    rows = []
    for cluster_id, sites in labels.groupby(labels):
        rows.append({
            "Cluster": int(cluster_id),
            "Sites": ", ".join(sites.index.tolist()),
            "Count": len(sites),
        })
    df = pd.DataFrame(rows).set_index("Cluster")

    st.markdown("**Cluster composition**")
    st.caption(
        "Sites in the same cluster have similar surface chemistry. The "
        "groupings are computed only over the surface layer, so they "
        "reflect *current* spatial patterns rather than the full core history."
    )
    st.dataframe(df, use_container_width=True)
```

---

## 3. Cambios en `app/views/results.py`

Dos cambios localizados.

### 3.1 Agregar el tab a la lista de tabs

**BUSCAR:**

```python
    # ---- TAB LAYOUT ----
    tab_pca, tab_corr, tab_cluster, tab_anomaly = st.tabs([
        "PCA", "Correlations", "Clustering", "Anomalies"
    ])
```

**REEMPLAZAR POR:**

```python
    # ---- TAB LAYOUT ----
    tab_pca, tab_corr, tab_cluster, tab_anomaly, tab_surface = st.tabs([
        "PCA", "Correlations", "Clustering", "Anomalies", "Surface (spatial)"
    ])
```

### 3.2 Renderizar el contenido del nuevo tab al final del archivo

**BUSCAR** las últimas líneas del archivo (el bloque que cierra el tab de Anomalies):

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

**REEMPLAZAR POR** (agrega el bloque del nuevo tab justo después):

```python
            # Anomaly details
            if results.anomalies.n_anomalies > 0:
                with st.expander("Anomalous samples"):
                    import pandas as pd
                    anomaly_idx = results.anomalies.anomaly_indices
                    if raw_df is not None and len(anomaly_idx) > 0:
                        anomaly_rows = raw_df.loc[anomaly_idx]
                        st.dataframe(anomaly_rows, use_container_width=True)

    # ============================================================
    # TAB 5: SURFACE (SPATIAL)
    # ============================================================
    with tab_surface:
        from app.views._surface_tab import render_surface_tab

        render_surface_tab(results)
```

---

## 4. Validación

```bash
# 1. Sintaxis
python -c "
import ast
for f in ['app/views/results.py', 'app/views/_surface_tab.py']:
    ast.parse(open(f).read())
    print(f'OK {f}')
"
```

```bash
# 2. Tests (no deben cambiar)
pytest tests/ -q
```
**Esperado:** `71 passed`.

```bash
# 3. Smoke visual
streamlit run app/main.py
```

**Verificación visual (subir ISOVIDA, ir a Results):**

- ✅ Hay 5 pestañas: PCA, Correlations, Clustering, Anomalies, **Surface (spatial)**.
- ✅ En "Surface (spatial)":
  - Dropdown con valores 5.0, 10.0, 20.0 (preselectionado 10.0).
  - KPI bar muestra: 10 cm / 7 sitios / 54 muestras / 3 grupos.
  - Heatmap rojo-azul (Z-score) con 7 filas (sitios) y unas 30 columnas (variables). Las filas están ordenadas por cluster: Junco Sur, Laguna Guanaroca y Obourke aparecen juntos arriba; después Delfinario, Río Damují y Río Salado; y al final Arrollo Inglés solo.
  - Mapa scatter con 7 puntos etiquetados con sus nombres de sitio, coloreados por cluster.
  - Tabla "Cluster composition" con los 3 grupos y sus sitios.
- ✅ Cambiar el dropdown a **5 cm**: la página se actualiza rápido. La KPI bar dice 5 cm / 7 sitios / 32 muestras / 3 grupos. El heatmap y la tabla se redibujan.
- ✅ Cambiar el dropdown a **20 cm**: 20 cm / 7 sitios / 94 muestras / 3 grupos.

---

## 5. Si algo falla

- Si la pestaña aparece pero está vacía con "Surface analysis was not
  executed" → el dataset cargado no tiene site_col o depth_col. Revisar
  Audit Overview para ver qué se detectó. Para ISOVIDA ambos están
  presentes.
- Si crashea con `AttributeError: 'AEDAResults' object has no attribute
  'surface_analysis'` → el commit del backend (Tanda 2 etapa 1) no se
  aplicó. Re-aplicar `CODEX_PROMPT_TANDA2_SURFACE_BACKEND.md` primero.
- Si el heatmap se ve "rojo y nada más" o "blanco" → es porque alguna
  variable tiene varianza cero entre sitios (todas las filas iguales).
  El módulo ya filtra estas columnas (`stds > 0`), pero si todas las
  variables son planas la lógica devuelve `st.info("No variables show
  variance across sites.")`. Eso es esperado.
- Si el mapa no aparece → el dataset no tiene coordenadas detectadas
  (verificar `dataset_info.coordinate_cols` en Audit).
- No tocar nada bajo `aeda/`, `tests/`. Esta etapa es solo UI:
  `app/views/results.py` + nuevo `app/views/_surface_tab.py`.

---

## 6. Mensaje de commit sugerido

```
feat(ui): Surface (spatial) tab in Results page (Yoelvis Tanda 2 — UI)

Adds the user-facing layer for the surface-spatial analysis introduced
in the previous commit. New tab in the Results page:

- Depth selector (5 / 10 / 20 cm) that recomputes only the surface
  module — the rest of the pipeline output stays cached.
- KPI bar with depth, site count, sample count, cluster count.
- Z-score heatmap (site × variable), rows ordered by cluster so
  members of the same group are adjacent.
- Geographic scatter (lat/lon) with sites colored by cluster, when
  coordinates are available.
- Cluster composition table.

Implementation moved to a self-contained module
``app/views/_surface_tab.py`` to keep ``app/views/results.py`` short.
The tab is the 5th alongside PCA / Correlations / Clustering / Anomalies.

UI only; engine and tests untouched. 71 tests still pass.
```
