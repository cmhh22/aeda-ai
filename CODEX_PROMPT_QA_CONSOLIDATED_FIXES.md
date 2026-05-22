# CODEX_PROMPT_QA_CONSOLIDATED_FIXES

**Tipo:** Consolidación de fixes del QA walkthrough (Yoelvis perspective)
**Archivos:** 8 modificados, 1 nuevo
**Tiempo estimado:** 25 min
**Tests esperados después:** 71 passed (sin cambios)

---

## 1. Contexto

Después del recorrido completo por las 6 páginas con Yoelvis como persona,
acumulamos 10 hallazgos: 2 críticos, 2 funcionales-UI, 3 UX importantes,
3 cosméticos. Este `.md` los aplica todos en un solo commit consolidado.

Todos los fixes fueron implementados y validados contra ISOVIDA en
sandbox antes de empaquetar:
- 71/71 tests pasan
- `coordinate_cols` correcto (solo Latitud, Longitud)
- `Y` está en measurement_cols (no como coordenada)
- K=7 se aplica (no K=9 por auto-silhouette)
- Spatial recommendation aparece en el plan
- Label 0.6 consistente con threshold real
- Units cargadas (Pb → mg/kg)

---

## 2. FIX #1 — Crítico A1: `Y` (itrio) detectado como coordenada

**Archivo:** `aeda/io/parsers.py`

**Causa:** El patrón `"y"` en `KNOWN_COORDINATE_PATTERNS` se chequea con
substring match (`p in low`), así que la columna `Y` (elemento químico
itrio) cae como coordenada. Lo mismo `"x"`.

### 2.1 Quitar x/y de la lista de patterns geográficos

**BUSCAR:**

```python
# Common patterns in LEA environmental datasets
KNOWN_COORDINATE_PATTERNS = ["latitud", "longitud", "lat", "lon", "x", "y"]
```

**REEMPLAZAR POR:**

```python
# Common patterns in LEA environmental datasets.
# X/Y are intentionally excluded from this list because they are too short
# to match by substring (the substring "y" matches the chemical element Y
# = yttrium). UTM-style 'X' and 'Y' coordinates are handled below by a
# separate contextual rule that requires BOTH columns to be present AND
# no explicit geographic coordinates (lat/lon) to be detected first.
KNOWN_COORDINATE_PATTERNS = [
    "latitud", "longitud", "latitude", "longitude", "lat", "lon", "lng"
]
```

### 2.2 Agregar la regla contextual X/Y

En la función `_detect_special_columns`:

**BUSCAR:**

```python
    for orig, low in cols_lower.items():
        if any(p in low for p in KNOWN_COORDINATE_PATTERNS):
            result["coordinate_cols"].append(orig)
        elif any(p in low for p in KNOWN_DEPTH_PATTERNS):
            result["depth_col"] = orig
        elif any(p in low for p in KNOWN_SITE_PATTERNS):
            result["site_col"] = orig
        elif any(low.startswith(p) for p in KNOWN_UNCERTAINTY_PREFIX):
            result["uncertainty_cols"].append(orig)

    non_special = set(result["coordinate_cols"] + result["uncertainty_cols"])
```

**REEMPLAZAR POR:**

```python
    for orig, low in cols_lower.items():
        if any(p in low for p in KNOWN_COORDINATE_PATTERNS):
            result["coordinate_cols"].append(orig)
        elif any(p in low for p in KNOWN_DEPTH_PATTERNS):
            result["depth_col"] = orig
        elif any(p in low for p in KNOWN_SITE_PATTERNS):
            result["site_col"] = orig
        elif any(low.startswith(p) for p in KNOWN_UNCERTAINTY_PREFIX):
            result["uncertainty_cols"].append(orig)

    # Contextual UTM detection: only when no geographic coordinates (lat/lon)
    # were detected, AND both 'X' and 'Y' columns are present, treat them
    # as UTM coordinates. The 'both required' rule prevents the chemical
    # element Y (yttrium) from being mis-detected as a coordinate.
    if not result["coordinate_cols"]:
        lower_values = set(cols_lower.values())
        if "x" in lower_values and "y" in lower_values:
            for orig, low in cols_lower.items():
                if low in {"x", "y"}:
                    result["coordinate_cols"].append(orig)

    non_special = set(result["coordinate_cols"] + result["uncertainty_cols"])
```

---

## 3. FIX #2 — Crítico A2: K=7 vs K=9

**Archivo:** `aeda/pipeline/runner.py`

**Causa:** El brain recomienda K=7 como Primary HIGH (regla geográfica),
pero el clustering engine corre `cluster(method="auto", n_clusters=None)`
que dispara auto-K por silhouette y elige K=9. El número de clusters de
la recomendación primary no se pasa al engine. Resultado: Decisions dice
K=7, Technical dice K=9, audit incoherente.

### 3.1 Agregar resolución de clustering_kwargs

**BUSCAR:**

```python
        # Resolve contamination: if the user did not set it explicitly, use
        # the value recommended by the auto-selector (primary anomaly rec-
        # ommendation). Falls back to 0.05 if no recommendation is available.
        effective_contamination = self.contamination
        if effective_contamination is None:
            for rec in plan.recommendations:
                if rec.category == "anomaly" and rec.priority == 1:
                    candidate = rec.params.get("contamination")
                    if candidate is not None:
                        effective_contamination = float(candidate)
                    break
            if effective_contamination is None:
                effective_contamination = 0.05
```

**REEMPLAZAR POR:**

```python
        # Resolve contamination: if the user did not set it explicitly, use
        # the value recommended by the auto-selector (primary anomaly rec-
        # ommendation). Falls back to 0.05 if no recommendation is available.
        effective_contamination = self.contamination
        if effective_contamination is None:
            for rec in plan.recommendations:
                if rec.category == "anomaly" and rec.priority == 1:
                    candidate = rec.params.get("contamination")
                    if candidate is not None:
                        effective_contamination = float(candidate)
                    break
            if effective_contamination is None:
                effective_contamination = 0.05

        # Resolve clustering kwargs: if the user did not pass any expert
        # overrides, honor the brain's primary clustering recommendation
        # (e.g. K=7 for the geographic-validation rule on ISOVIDA).
        # Without this wiring, the engine's auto-K silhouette would override
        # the rule-based choice, breaking the Decisions ↔ Technical audit
        # consistency.
        effective_clustering_kwargs = dict(self.clustering_kwargs)
        if not effective_clustering_kwargs:
            for rec in plan.recommendations:
                if rec.category == "clustering" and rec.priority == 1:
                    primary_params = rec.params or {}
                    candidate_k = primary_params.get("n_clusters")
                    if candidate_k is not None:
                        effective_clustering_kwargs["n_clusters"] = int(candidate_k)
                    break
```

### 3.2 Usar effective_clustering_kwargs en cluster()

**BUSCAR:**

```python
        # 6. CLUSTERING
        try:
            results.clustering = cluster(
                processed, method=self.clustering_method, **self.clustering_kwargs
            )
        except Exception as e:
            logger.warning(f"Clustering failed: {type(e).__name__}: {e}")
            results.clustering = None
```

**REEMPLAZAR POR:**

```python
        # 6. CLUSTERING
        try:
            results.clustering = cluster(
                processed, method=self.clustering_method, **effective_clustering_kwargs
            )
        except Exception as e:
            logger.warning(f"Clustering failed: {type(e).__name__}: {e}")
            results.clustering = None
```

### 3.3 Persistir el valor efectivo en effective_settings

**BUSCAR:**

```python
            "dim_kwargs": dict(self.dim_kwargs),
            "clustering_kwargs": dict(self.clustering_kwargs),
            "anomaly_kwargs": dict(self.anomaly_kwargs),
        }
```

**REEMPLAZAR POR:**

```python
            "dim_kwargs": dict(self.dim_kwargs),
            "clustering_kwargs": dict(effective_clustering_kwargs),
            "anomaly_kwargs": dict(self.anomaly_kwargs),
        }
```

---

## 4. FIX #3 — Categoría "Spatial" no se renderiza

**Archivos:** `app/views/plan.py`, `app/views/audit.py`

**Causa:** El brain emite recomendaciones con `category="spatial"`
desde la Tanda 2, pero los renderers de Plan y Audit no la incluyen en
su mapeo de categorías. Resultado: la recomendación queda invisible para
el usuario.

### 4.1 plan.py

**BUSCAR:**

```python
    categories = [
        ("preprocessing", "Preprocessing"),
        ("dimensionality", "Dimensionality reduction"),
        ("clustering", "Clustering"),
        ("anomaly", "Anomaly detection"),
        ("correlation", "Correlations"),
        ("feature_analysis", "Feature analysis"),
    ]
```

**REEMPLAZAR POR:**

```python
    categories = [
        ("preprocessing", "Preprocessing"),
        ("dimensionality", "Dimensionality reduction"),
        ("clustering", "Clustering"),
        ("spatial", "Spatial analysis"),
        ("anomaly", "Anomaly detection"),
        ("correlation", "Correlations"),
        ("feature_analysis", "Feature analysis"),
    ]
```

### 4.2 audit.py

**BUSCAR:**

```python
    # Friendly category names — geochemistry-first, no ML jargon in headings
    category_labels = {
        "preprocessing": "Data preparation",
        "dimensionality": "Variable summarization",
        "clustering": "Sample grouping",
        "anomaly": "Anomaly detection",
        "correlation": "Variable relationships",
        "feature_analysis": "Most informative variables",
    }
```

**REEMPLAZAR POR:**

```python
    # Friendly category names — geochemistry-first, no ML jargon in headings
    category_labels = {
        "preprocessing": "Data preparation",
        "dimensionality": "Variable summarization",
        "clustering": "Sample grouping",
        "spatial": "Spatial analysis",
        "anomaly": "Anomaly detection",
        "correlation": "Variable relationships",
        "feature_analysis": "Most informative variables",
    }
```

---

## 5. FIX #4 — Label `|r|>0.7` debería ser `|r|>0.6`

**Archivo:** `app/views/plan.py`

**Causa:** Tanda 1 bajó el `CORRELATION_BLOCK_THRESHOLD` de 0.7 a 0.6
pero el label de la UI quedó hardcoded en 0.7.

**BUSCAR:**

```python
    col3.metric("Correlated pairs (|r|>0.7)", profile.high_correlation_pairs)
```

**REEMPLAZAR POR:**

```python
    col3.metric("Correlated pairs (|r|>0.6)", profile.high_correlation_pairs)
```

---

## 6. FIX #5 — Inconsistencia "Variables: 52 / 37 / 36"

**Archivos:** `app/main.py`, `app/views/advanced.py`, `app/views/audit.py`

**Causa:** El mismo label "Variables" significa cosas distintas en
páginas distintas (total del Excel vs analizadas vs solo químicas).

### 6.1 Sidebar (`app/main.py`)

**BUSCAR:**

```python
    cols = st.sidebar.columns(2)
    cols[0].metric("Samples", results.raw_data.shape[0])
    cols[1].metric("Variables", results.raw_data.shape[1])
```

**REEMPLAZAR POR:**

```python
    cols = st.sidebar.columns(2)
    cols[0].metric("Samples", results.raw_data.shape[0])
    cols[1].metric("Columns", results.raw_data.shape[1],
                   help="Total columns in the uploaded file. "
                        "The number actually analyzed (after excluding identifiers, "
                        "metadata and coordinates) is shown on each page.")
```

### 6.2 Advanced (`app/views/advanced.py`)

**BUSCAR:**

```python
    col1.metric("Dataset", st.session_state.get("filename", "—"))
    col2.metric("Samples", raw_df.shape[0] if raw_df is not None else 0)
    col3.metric("Variables", raw_df.shape[1] if raw_df is not None else 0)
```

**REEMPLAZAR POR:**

```python
    col1.metric("Dataset", st.session_state.get("filename", "—"))
    col2.metric("Samples", raw_df.shape[0] if raw_df is not None else 0)
    col3.metric("Total columns", raw_df.shape[1] if raw_df is not None else 0)
```

### 6.3 Audit Overview (`app/views/audit.py`)

**BUSCAR:**

```python
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("File", filename)
    col2.metric("Samples", raw_df.shape[0] if raw_df is not None else 0)
    col3.metric("Variables", raw_df.shape[1] if raw_df is not None else 0)

    n_measured = len(info.measurement_cols) if info else 0
    col4.metric("Measurement variables", n_measured)
```

**REEMPLAZAR POR:**

```python
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("File", filename)
    col2.metric("Samples", raw_df.shape[0] if raw_df is not None else 0)
    col3.metric("Total columns", raw_df.shape[1] if raw_df is not None else 0)

    n_measured = len(info.measurement_cols) if info else 0
    col4.metric("Chemistry variables", n_measured,
                help="Numeric measurement columns (metals, granulometry, ancillary). "
                     "Excludes identifiers, coordinates, depth and site columns.")
```

---

## 7. FIX #6 — Leyendas cortadas en plots

**Archivo:** `aeda/viz/base.py`

**Causa:** El `margin.r=40` global no deja espacio para que la leyenda
lateral muestre nombres completos. Plotly trunca a la primera letra
("D, L, O, J, A, R, R" en vez de "Delfinario, Laguna Guanaroca, ...").

**BUSCAR:**

```python
#: Standard layout applied to every figure. Values can be overridden per-plot.
DEFAULT_LAYOUT = dict(
    template="simple_white",
    font=dict(family="Arial, sans-serif", size=13, color="#2E4057"),
    title=dict(x=0.5, xanchor="center", font=dict(size=16)),
    margin=dict(l=60, r=40, t=70, b=60),
    plot_bgcolor="white",
    paper_bgcolor="white",
    hoverlabel=dict(
        bgcolor="white",
        bordercolor="#CCCCCC",
        font=dict(family="Arial, sans-serif", size=12),
    ),
    legend=dict(
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#CCCCCC",
        borderwidth=1,
    ),
)
```

**REEMPLAZAR POR:**

```python
#: Standard layout applied to every figure. Values can be overridden per-plot.
#: ``margin.r`` is generous so legends with longish category names (e.g. site
#: names like "Laguna Guanaroca", "Arrollo Inglés", "Río Damují") fit without
#: getting clipped to single letters.
DEFAULT_LAYOUT = dict(
    template="simple_white",
    font=dict(family="Arial, sans-serif", size=13, color="#2E4057"),
    title=dict(x=0.5, xanchor="center", font=dict(size=16)),
    margin=dict(l=60, r=160, t=70, b=60),
    plot_bgcolor="white",
    paper_bgcolor="white",
    hoverlabel=dict(
        bgcolor="white",
        bordercolor="#CCCCCC",
        font=dict(family="Arial, sans-serif", size=12),
    ),
    legend=dict(
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#CCCCCC",
        borderwidth=1,
        font=dict(size=11),
    ),
)
```

---

## 8. FIX #7 — JSON crudo → tabla key-value amigable

**Archivos:** Nuevo `app/components/params.py`; modificados `app/views/plan.py` y `app/views/audit.py`

**Causa:** Los parámetros del autoselector se muestran con `st.json`,
que es ruido visual para un usuario científico no-programador.

### 8.1 Archivo nuevo: `app/components/params.py`

Crear con este contenido íntegro:

```python
"""Friendly rendering of method parameters.

These helpers replace raw ``st.json`` calls in the Plan and Audit pages
with a compact key-value layout that is easier to read for non-developer
users (e.g. the scientific tutor). Nested values such as the CLR subgroup
dictionaries are flattened into multi-line strings so a scientist can
read the configuration without parsing JSON syntax.
"""

from __future__ import annotations

from typing import Any

import streamlit as st


def _format_value(value: Any) -> str:
    """Render a single parameter value as a short human-readable string."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if value is None:
        return "—"
    if isinstance(value, (list, tuple)):
        if not value:
            return "—"
        # Lists of primitives → comma-separated; lists of dicts → counts
        if all(isinstance(v, (str, int, float)) for v in value):
            return ", ".join(str(v) for v in value)
        return f"{len(value)} item(s)"
    if isinstance(value, dict):
        # Single-level dicts: render as inline key=value pairs
        if all(not isinstance(v, (dict, list, tuple)) for v in value.values()):
            return ", ".join(f"{k}={_format_value(v)}" for k, v in value.items())
        return f"{len(value)} entries (nested)"
    if isinstance(value, float):
        # Avoid noise from float printing: keep 4 significant digits
        if abs(value) < 1e-3 or abs(value) > 1e4:
            return f"{value:.3e}"
        return f"{value:g}"
    return str(value)


def _format_key(key: str) -> str:
    """Turn a snake_case parameter key into Title Case."""
    return key.replace("_", " ").capitalize()


def render_params(params: dict, key_prefix: str = "") -> None:
    """Render a parameters dict as a two-column key-value layout.

    Top-level scalar values render inline. Nested dicts and lists of dicts
    (e.g. CLR subgroups) get their own indented block so the structure
    stays readable without dumping raw JSON.
    """
    if not params:
        return

    for key, value in params.items():
        label = _format_key(key)

        if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
            # Lists of dicts (e.g. CLR subgroups): render each as a sub-block
            st.markdown(f"**{label}**")
            for i, item in enumerate(value, 1):
                name = item.get("name", f"#{i}")
                st.caption(f"  · {name}")
                for sub_k, sub_v in item.items():
                    if sub_k == "name":
                        continue
                    st.caption(f"     - {_format_key(sub_k)}: {_format_value(sub_v)}")
        elif isinstance(value, dict):
            st.markdown(f"**{label}**")
            for sub_k, sub_v in value.items():
                st.caption(f"  · {_format_key(sub_k)}: {_format_value(sub_v)}")
        else:
            col1, col2 = st.columns([1, 2])
            col1.markdown(f"**{label}**")
            col2.markdown(_format_value(value))
```

### 8.2 plan.py

**BUSCAR:**

```python
                # Parameters are technical detail — hide them behind a small
                # collapsed expander so non-developer users (e.g. the tutor)
                # are not faced with raw JSON by default.
                if rec.params:
                    with st.expander("Show parameters", expanded=False):
                        st.json(rec.params, expanded=False)
```

**REEMPLAZAR POR:**

```python
                # Parameters are technical detail — hide them behind a small
                # collapsed expander so non-developer users (e.g. the tutor)
                # are not faced with raw JSON by default.
                if rec.params:
                    with st.expander("Show parameters", expanded=False):
                        from app.components.params import render_params
                        render_params(rec.params)
```

### 8.3 audit.py — Decisions tab

**BUSCAR:**

```python
            if primary.params:
                st.caption("Parameters chosen by the auto-selector:")
                st.json(primary.params, expanded=False)
```

**REEMPLAZAR POR:**

```python
            if primary.params:
                with st.expander("Parameters chosen by the auto-selector", expanded=False):
                    from app.components.params import render_params
                    render_params(primary.params)
```

### 8.4 audit.py — Technical preprocessing trace

**BUSCAR:**

```python
        for i, step in enumerate(log.steps, 1):
            step_name = step.get("step", "?")
            details = {k: v for k, v in step.items() if k != "step"}
            with st.expander(f"Step {i}: **{step_name}**"):
                if details:
                    st.json(details, expanded=False)
                else:
                    st.caption("(no parameters recorded)")
```

**REEMPLAZAR POR:**

```python
        for i, step in enumerate(log.steps, 1):
            step_name = step.get("step", "?")
            details = {k: v for k, v in step.items() if k != "step"}
            with st.expander(f"Step {i}: **{step_name}**"):
                if details:
                    from app.components.params import render_params
                    render_params(details)
                else:
                    st.caption("(no parameters recorded)")
```

---

## 9. FIX #9 — Tabla "Variable statistics" sin título

**Archivo:** `app/views/depth.py`

**Causa:** La tabla aparece sin contexto. Falta título y caption.

**BUSCAR:**

```python
    # Quick stats
    with st.expander("Variable statistics"):
        stats = filtered_df.groupby(site_col)[variable].describe() if site_col else filtered_df[variable].describe()
        st.dataframe(stats, use_container_width=True)
```

**REEMPLAZAR POR:**

```python
    # Quick stats
    with st.expander("Variable statistics"):
        if site_col:
            st.markdown(f"**Distribution of {variable} per site**")
            st.caption(
                "Descriptive statistics computed across all depths within each site. "
                "Useful to compare typical values and variability between sampling locations."
            )
            stats = filtered_df.groupby(site_col)[variable].describe()
        else:
            st.markdown(f"**Distribution of {variable}**")
            stats = filtered_df[variable].describe()
        st.dataframe(stats, use_container_width=True)
```

---

## 10. FIX #10 — Unidades implícitas en Depth Profiles

**Archivos:** `aeda/viz/profiles.py`, `app/views/depth.py`

**Causa:** Los ejes muestran solo el nombre de la variable, sin unidad.
Yoelvis lo sabe pero está implícito. El dataset_info ya tiene las
unidades cargadas desde el Diccionario_Data del Excel (Tanda 1).

### 10.1 Pasar `unit` opcional al depth_profile single

En `aeda/viz/profiles.py`, **BUSCAR** la firma de la función `depth_profile`:

```python
def depth_profile(
    df: pd.DataFrame,
    variable: str,
    depth_col: str,
    site_col: Optional[str] = None,
    core_col: Optional[str] = None,
    title: Optional[str] = None,
    width: int = 900,
    height: int = 600,
) -> go.Figure:
```

**REEMPLAZAR POR:**

```python
def depth_profile(
    df: pd.DataFrame,
    variable: str,
    depth_col: str,
    site_col: Optional[str] = None,
    core_col: Optional[str] = None,
    title: Optional[str] = None,
    width: int = 900,
    height: int = 600,
    unit: Optional[str] = None,
) -> go.Figure:
```

Y en el cuerpo, **BUSCAR:**

```python
    fig.update_xaxes(title=variable, gridcolor="#F0F0F0")
```

**REEMPLAZAR POR:**

```python
    x_label = f"{variable} ({unit})" if unit else variable
    fig.update_xaxes(title=x_label, gridcolor="#F0F0F0")
```

### 10.2 Pasar `units` opcional al depth_profile_grid

En `aeda/viz/profiles.py`, **BUSCAR** la firma de `depth_profile_grid`:

```python
    df: pd.DataFrame,
    variables: list[str],
    depth_col: str,
    site_col: Optional[str] = None,
    n_cols: int = 3,
    title: Optional[str] = None,
    width: int = 1100,
    height: Optional[int] = None,
) -> go.Figure:
```

**REEMPLAZAR POR:**

```python
    df: pd.DataFrame,
    variables: list[str],
    depth_col: str,
    site_col: Optional[str] = None,
    n_cols: int = 3,
    title: Optional[str] = None,
    width: int = 1100,
    height: Optional[int] = None,
    units: Optional[dict] = None,
) -> go.Figure:
```

Y en el cuerpo, **BUSCAR** el `make_subplots`:

```python
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=variables,
        horizontal_spacing=0.08,
        vertical_spacing=0.10 if n_rows <= 3 else 0.06,
    )
```

**REEMPLAZAR POR:**

```python
    # Build subplot titles that include the unit when known. Falls back to
    # just the variable name for variables without a unit in the dictionary.
    def _title_with_unit(v: str) -> str:
        if units and v in units and units[v]:
            return f"{v} ({units[v]})"
        return v

    subplot_titles = [_title_with_unit(v) for v in variables]

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08,
        vertical_spacing=0.10 if n_rows <= 3 else 0.06,
    )
```

### 10.3 Wire units desde la UI

En `app/views/depth.py`, **BUSCAR**:

```python
    if mode == "Single variable":
        _render_single(raw_df, variable_options, depth_col, site_col)
    else:
        _render_grid(raw_df, variable_options, depth_col, site_col)
```

**REEMPLAZAR POR:**

```python
    # Units dictionary (read from the Excel data dictionary by the parser).
    # Used to label axes with concrete units like "Pb (mg/kg)".
    units = info.units or {}

    if mode == "Single variable":
        _render_single(raw_df, variable_options, depth_col, site_col, units)
    else:
        _render_grid(raw_df, variable_options, depth_col, site_col, units)
```

**BUSCAR** la firma de `_render_single`:

```python
def _render_single(df, variable_options, depth_col, site_col):
    """Render a single-variable depth profile with site and core selection."""
    from aeda.viz.profiles import depth_profile
```

**REEMPLAZAR POR:**

```python
def _render_single(df, variable_options, depth_col, site_col, units=None):
    """Render a single-variable depth profile with site and core selection."""
    from aeda.viz.profiles import depth_profile
```

**BUSCAR** la llamada a `depth_profile`:

```python
    fig = depth_profile(
        filtered_df,
        variable=variable,
        depth_col=depth_col,
        site_col=site_col,
        core_col=core_col,
    )
    st.plotly_chart(fig, use_container_width=True)
```

**REEMPLAZAR POR:**

```python
    fig = depth_profile(
        filtered_df,
        variable=variable,
        depth_col=depth_col,
        site_col=site_col,
        core_col=core_col,
        unit=(units or {}).get(variable),
    )
    st.plotly_chart(fig, use_container_width=True)
```

**BUSCAR** la firma de `_render_grid`:

```python
def _render_grid(df, variable_options, depth_col, site_col):
    """Render a grid of depth profiles for multiple variables."""
    from aeda.viz.profiles import depth_profile_grid
```

**REEMPLAZAR POR:**

```python
def _render_grid(df, variable_options, depth_col, site_col, units=None):
    """Render a grid of depth profiles for multiple variables."""
    from aeda.viz.profiles import depth_profile_grid
```

**BUSCAR** la llamada a `depth_profile_grid`:

```python
    fig = depth_profile_grid(
        df,
        variables=variables,
        depth_col=depth_col,
        site_col=site_col,
        n_cols=n_cols,
    )
    st.plotly_chart(fig, use_container_width=True)
```

**REEMPLAZAR POR:**

```python
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

## 11. Validación

```bash
# 1. Sintaxis de todos los archivos tocados
python -c "
import ast
for f in ['aeda/io/parsers.py', 'aeda/pipeline/runner.py',
          'aeda/viz/profiles.py', 'aeda/viz/base.py',
          'app/main.py', 'app/views/plan.py', 'app/views/audit.py',
          'app/views/depth.py', 'app/views/advanced.py',
          'app/components/params.py']:
    ast.parse(open(f).read())
    print(f'OK {f}')
"
```

```bash
# 2. Suite completa
pytest tests/ -q
```
**Esperado:** `71 passed`.

```bash
# 3. Smoke contra ISOVIDA
python -c "
from aeda.pipeline.runner import AEDAPipeline
EXCLUDE = ['No','Code','Site_Name','Pret_Code','Código_muestra',
           'Sitio_muestreo','Fecha_muestreo','Core','Latitud','Longitud']
r = AEDAPipeline(impute_strategy='median').run(
    'data/BD_ISOVIDA_MANGLARES2023_rectificadaYBA_230326.xlsx',
    exclude_cols=EXCLUDE, sheet_name='DATA',
)
print(f'coordinate_cols: {r.dataset_info.coordinate_cols}')
print(f'Y in measurement_cols: {chr(34)}Y{chr(34) in r.dataset_info.measurement_cols}')
print(f'K applied: {r.clustering.n_clusters}')
print(f'spatial recs: {len([x for x in r.plan.recommendations if x.category == chr(34)spatial{chr(34)}])}')
print(f'units loaded: {len(r.dataset_info.units)}')
"
```

**Esperado:**
```
coordinate_cols: ['Latitud', 'Longitud']
Y in measurement_cols: True
K applied: 7
spatial recs: 1
units loaded: 48
```

```bash
# 4. Smoke visual
streamlit run app/main.py
```

Verificar en el navegador:
- **Plan:** "Correlated pairs (|r|>0.6)" en el dataset profile, sección
  nueva "Spatial analysis" en method recommendations, los parámetros
  aparecen como tabla key-value (no JSON crudo).
- **Results Clustering:** título dice "K-Means (7 clusters)" y la tab
  muestra 7 grupos.
- **Audit Overview:** "Total columns" y "Chemistry variables", la
  metadata detection ya no menciona Y como coordenada.
- **Audit Decisions:** sección "Spatial analysis" aparece, parámetros
  como tabla, K=7 coherente con Technical.
- **Audit Technical:** preprocessing trace con tabla key-value.
- **Depth Profiles:** subplot titles "Pb (mg/kg)", "As (mg/kg)", etc;
  expander de stats con título "Distribution of X per site".
- **Sidebar:** "Columns: 52" con tooltip explicativo.
- **Todos los plots:** las leyendas muestran los nombres completos
  ("Laguna Guanaroca", "Arrollo Inglés"), no "L, A" truncados.

---

## 12. Mensaje de commit sugerido

```
fix(qa): consolidate 10 findings from Yoelvis-perspective walkthrough

Critical:
- parsers: 'Y' (yttrium) no longer mis-detected as coordinate. The
  short patterns 'x'/'y' were removed from KNOWN_COORDINATE_PATTERNS
  and moved to a contextual rule that requires BOTH columns and no
  prior lat/lon detection.
- runner: clustering now respects the brain's primary recommendation
  (e.g. K=7 for ISOVIDA's geographic-validation rule) instead of
  letting auto-K silhouette override it. Decisions ↔ Technical audit
  pages are now consistent.

UI / functional:
- plan & audit: 'spatial' category now appears in the Method
  recommendations and Decisions sections (was emitted by the brain
  but silently dropped by the renderer).
- plan: 'Correlated pairs' label updated from |r|>0.7 to |r|>0.6 to
  match the threshold lowered in Tanda 1.

UX:
- viz/base: increased default right margin (40 → 160) and reduced
  legend font so site names like "Laguna Guanaroca" are no longer
  truncated to single letters in the legends.
- new app/components/params.py: friendly key-value renderer that
  replaces raw st.json in Plan recommendations, Audit Decisions and
  Audit Technical preprocessing trace.
- main/advanced/audit: clarified the "Variables" metrics — sidebar
  shows 'Columns' (total in file), Audit Overview shows 'Total
  columns' + 'Chemistry variables', tooltips explain the difference.

Cosmetic:
- depth.py: 'Variable statistics' expander now has a proper title
  and caption.
- viz/profiles + depth.py: subplot titles in the grid include the
  unit ("Pb (mg/kg)"), read from the data dictionary loaded by the
  parser. Single-variable X-axis also gets the unit when known.

All 71 tests pass. Validated on ISOVIDA.
```
