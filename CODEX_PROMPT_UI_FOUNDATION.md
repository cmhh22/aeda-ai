# CODEX_PROMPT_UI_FOUNDATION

**Tipo:** UI/UX foundation (theming + components + bug fixes)
**Archivos:** 4 nuevos + 7 modificados
**Tiempo estimado:** ~30 min
**Tests esperados después:** 38 tests (sin cambios — esto es UI, no toca el engine)

---

## 1. Contexto

La interfaz Streamlit existente funciona pero usa el tema default (genérico),
los headers de cada página son inconsistentes, los errores muestran tracebacks
crudos que asustan al usuario científico, y hay un bug en `results.py` donde
los `return` dentro de un tab cortan toda la página en vez de solo ese tab.

Este prompt establece la **base visual** del proyecto:

- Paleta cromática "earth science" coherente con el dominio (sediment).
- Theming central con CSS layer aplicado al inicio de la app.
- Componentes reutilizables: `page_header()` y `show_error()`.
- Sidebar rediseñado con branding 🔬 + AEDA-AI.
- Fix del bug de `return` en tabs.

Después de este prompt la app va a tener una identidad visual propia y un
shell consistente. El siguiente prompt (`UI_PAGES_POLISH`) refinará el
contenido específico de cada página.

---

## 2. Archivos NUEVOS a crear

### 2.1 `.streamlit/config.toml`

Crear el directorio `.streamlit/` en la raíz del proyecto si no existe, y
dentro un archivo `config.toml` con este contenido exacto:

```toml
[theme]
base = "light"
primaryColor = "#5C7548"
backgroundColor = "#FAF8F4"
secondaryBackgroundColor = "#F0EDE5"
textColor = "#2A2A2A"
font = "sans serif"

[server]
maxUploadSize = 200

[browser]
gatherUsageStats = false
```

### 2.2 `app/theme.py`

Archivo nuevo con la paleta y el CSS layer. Contenido completo:

```python
"""
Theme module — central source of truth for the AEDA-AI visual identity.

The Streamlit-level theme (primary color, background, font) is configured in
.streamlit/config.toml. This module adds:

1. Named palette constants importable from anywhere (Plotly figures, custom
   widgets, etc.) so colors stay consistent across the app.
2. A small CSS layer applied via `apply_theme()` that refines typography,
   spacing, sidebar styling, metric cards and tabs without fighting the
   Streamlit base theme.

The palette is built around an environmental-sediment metaphor:
- Olive green (vegetation, sediment surface) as primary brand color.
- Ocean blue (aquatic environment) as secondary / informational color.
- Terracotta (warning, anomaly) as accent / alert color.
- Warm off-white background that reads as paper, not as a software UI.
"""

from __future__ import annotations

import streamlit as st


# =============================================================================
# Palette — earth-science scheme
# =============================================================================

# Primary — olive green family (sediment, vegetation)
PRIMARY = "#5C7548"
PRIMARY_DARK = "#3D4F2F"
PRIMARY_LIGHT = "#8BA177"

# Secondary — ocean blue family (aquatic environment, information)
OCEAN = "#2E5266"
OCEAN_DARK = "#1F3845"
OCEAN_LIGHT = "#5B7E94"

# Accent — terracotta family (warning, anomaly, attention)
TERRACOTTA = "#A0522D"
TERRACOTTA_LIGHT = "#C97B2C"

# Neutral grays — warm tone, not pure gray
BACKGROUND = "#FAF8F4"
SURFACE = "#FFFFFF"
SIDEBAR_BG = "#F0EDE5"
BORDER = "#E5E2DA"
TEXT_PRIMARY = "#2A2A2A"
TEXT_SECONDARY = "#5C5C5C"
TEXT_MUTED = "#8A8580"

# Semantic
SUCCESS = PRIMARY
WARNING = TERRACOTTA_LIGHT
ERROR = TERRACOTTA
INFO = OCEAN

# Categorical palette for plots (cluster colors, group comparisons, etc.)
# Designed to be distinguishable for color-blind users and to stay visually
# coherent with the earth-science scheme.
CATEGORICAL = [
    "#5C7548",   # olive
    "#2E5266",   # ocean
    "#A0522D",   # terracotta
    "#8BA177",   # light olive
    "#5B7E94",   # light ocean
    "#C97B2C",   # light terracotta
    "#3D4F2F",   # deep olive
    "#1F3845",   # deep ocean
]


# =============================================================================
# CSS layer
# =============================================================================

_CSS = f"""
<style>
/* ---------- Layout ---------- */
.block-container {{
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}}

/* ---------- Typography ---------- */
h1, h2, h3, h4 {{
    font-weight: 600;
    color: {TEXT_PRIMARY};
    letter-spacing: -0.01em;
}}
h1 {{ font-size: 1.875rem; margin-bottom: 0.25rem; }}
h2 {{ font-size: 1.375rem; margin-top: 1.75rem; margin-bottom: 0.5rem; }}
h3 {{ font-size: 1.125rem; margin-top: 1.25rem; color: {TEXT_SECONDARY}; }}

/* Captions read as supportive subtitles, not afterthoughts */
[data-testid="stCaptionContainer"] {{
    color: {TEXT_SECONDARY};
    font-size: 0.875rem;
    line-height: 1.45;
}}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {{
    background-color: {SIDEBAR_BG};
    border-right: 1px solid {BORDER};
}}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {{
    color: {TEXT_PRIMARY};
}}

/* Sidebar branding row: 🔬 + AEDA-AI */
.sidebar-brand {{
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.25rem 0 0.5rem 0;
}}
.sidebar-brand .brand-icon {{
    font-size: 1.5rem;
    line-height: 1;
}}
.sidebar-brand .brand-name {{
    font-size: 1.375rem;
    font-weight: 700;
    color: {PRIMARY_DARK};
    letter-spacing: -0.02em;
}}

/* Sidebar status block */
.status-empty {{
    color: {TEXT_MUTED};
    font-size: 0.875rem;
    font-style: italic;
    padding: 0.5rem 0;
}}

/* ---------- Metric cards ---------- */
[data-testid="stMetric"] {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    padding: 0.75rem 1rem;
    border-radius: 6px;
}}
[data-testid="stMetricLabel"] {{
    color: {TEXT_SECONDARY};
    font-size: 0.8125rem;
    font-weight: 500;
}}
[data-testid="stMetricValue"] {{
    font-weight: 600;
    color: {TEXT_PRIMARY};
}}

/* ---------- Buttons ---------- */
.stButton > button {{
    border-radius: 6px;
    font-weight: 500;
    border: 1px solid {BORDER};
    transition: all 0.15s ease;
}}
.stButton > button[kind="primary"] {{
    background-color: {PRIMARY};
    border-color: {PRIMARY};
}}
.stButton > button[kind="primary"]:hover {{
    background-color: {PRIMARY_DARK};
    border-color: {PRIMARY_DARK};
}}

/* ---------- Expanders ---------- */
[data-testid="stExpander"] {{
    border: 1px solid {BORDER};
    border-radius: 6px;
    background-color: {SURFACE};
}}
[data-testid="stExpander"] summary {{
    font-weight: 500;
}}

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] {{
    gap: 1rem;
}}
.stTabs [data-baseweb="tab"] {{
    font-weight: 500;
}}
.stTabs [data-baseweb="tab"][aria-selected="true"] {{
    color: {PRIMARY};
}}
.stTabs [data-baseweb="tab-highlight"] {{
    background-color: {PRIMARY};
}}

/* ---------- Dividers (less aggressive) ---------- */
hr {{
    border-color: {BORDER};
    margin: 1.5rem 0;
}}

/* ---------- Page header component (used by app/components/page_header.py) ---------- */
.page-header-row {{
    display: flex;
    align-items: center;
    gap: 0.625rem;
    margin-bottom: 0.25rem;
}}
.page-header-row .page-header-icon {{
    font-size: 1.625rem;
    line-height: 1;
}}
.page-header-row h1 {{
    margin: 0;
    font-size: 1.875rem;
}}

/* ---------- Reduce visual noise from default progress bar ---------- */
[data-testid="stProgress"] > div > div {{
    background-color: {PRIMARY};
}}
</style>
"""


def apply_theme() -> None:
    """Inject the CSS layer. Call once near the top of main.py, after
    `st.set_page_config(...)` but before any page rendering.
    """
    st.markdown(_CSS, unsafe_allow_html=True)
```

### 2.3 `app/components/page_header.py`

Archivo nuevo. Contenido completo:

```python
"""
Page header component.

Used at the top of every page for visual consistency. Replaces the ad-hoc
mix of `st.header() / st.title() / st.write() / st.caption()` calls that
each page currently uses with a single, themed pattern.

Usage
-----
    from app.components.page_header import page_header

    def render():
        page_header(
            title="Upload & Configure",
            description="Upload your environmental dataset and configure the analysis.",
            icon="📤",
        )
        # ... rest of the page ...
"""

from __future__ import annotations

import streamlit as st


def page_header(
    title: str,
    description: str | None = None,
    icon: str | None = None,
) -> None:
    """Render a standard page header.

    Parameters
    ----------
    title : str
        The page title (e.g. "Upload & Configure").
    description : str, optional
        A short subtitle shown below the title in caption style. Keep it under
        ~120 characters so it fits on one line on most screens.
    icon : str, optional
        A leading emoji or single character. Rendered next to the title.
    """
    if icon:
        st.markdown(
            f'<div class="page-header-row">'
            f'<span class="page-header-icon">{icon}</span>'
            f'<h1 class="page-header-title">{title}</h1>'
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.title(title)

    if description:
        st.caption(description)

    st.divider()
```

### 2.4 `app/components/errors.py`

Archivo nuevo. Contenido completo:

```python
"""
Error display helpers.

Replaces the pattern `st.error(msg); st.exception(exc)` used in upload.py and
elsewhere. Showing raw tracebacks scares non-technical users (the intended
audience here is environmental scientists, not developers); these helpers
keep the surface message clean while still allowing technical inspection
on demand via an expander.

Usage
-----
    from app.components.errors import show_error, show_warning

    try:
        ...
    except Exception as e:
        show_error(
            "Could not read the uploaded file. Make sure it is a valid Excel/CSV.",
            exc=e,
        )
"""

from __future__ import annotations

import streamlit as st


def show_error(message: str, exc: Exception | None = None) -> None:
    """Show a user-friendly error message.

    The plain-language `message` is shown as a normal Streamlit error.
    If `exc` is provided, an expandable section labelled "Technical details"
    is added underneath; it contains the exception type, message, and the
    full traceback. Casual users can ignore it; developers can open it.
    """
    st.error(message)
    if exc is not None:
        with st.expander("Technical details (for debugging)"):
            st.code(f"{type(exc).__name__}: {exc}", language="text")
            st.exception(exc)


def show_warning(message: str, detail: str | None = None) -> None:
    """Show a warning with an optional inline detail (no traceback)."""
    st.warning(message)
    if detail:
        st.caption(detail)
```

---

## 3. Archivo REESCRITO: `app/main.py`

Reemplazar el contenido completo de `app/main.py` por:

```python
"""
AEDA-AI — Streamlit interface, main entry point.

Run with:
    streamlit run app/main.py

Responsibilities of this module:
- Bootstrap sys.path so both `aeda.*` and `app.*` are importable.
- Configure the Streamlit page (title, icon, layout).
- Apply the AEDA-AI theme (palette + CSS layer).
- Initialize session state.
- Render the sidebar (branding + navigation + dataset status).
- Route the selected page to its render() function.
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so that both `aeda.*` and `app.*`
# imports work regardless of how Streamlit is invoked (from the project root
# or from inside the app/ directory).
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from app.theme import apply_theme

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AEDA-AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
if "results" not in st.session_state:
    st.session_state.results = None
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "filename" not in st.session_state:
    st.session_state.filename = None
# Persisted run context — used by the Advanced Configuration page to re-run
# the pipeline on the same data with different settings, without forcing the
# user to upload the file again.
if "run_context" not in st.session_state:
    st.session_state.run_context = None

# ---------------------------------------------------------------------------
# Pages registry — single source of truth for navigation.
# Each entry: (label shown in sidebar, dotted import path of the page module)
# ---------------------------------------------------------------------------
PAGES = [
    ("Upload & Configure", "app.pages.upload"),
    ("Analysis Plan", "app.pages.plan"),
    ("Results", "app.pages.results"),
    ("Depth Profiles", "app.pages.depth"),
    ("Audit", "app.pages.audit"),
    ("Advanced Configuration", "app.pages.advanced"),
]


def _render_sidebar() -> str:
    """Render the sidebar (branding + navigation + status block).

    Returns the label of the currently selected page.
    """
    # Branding: 🔬 + AEDA-AI in a single styled row
    st.sidebar.markdown(
        '<div class="sidebar-brand">'
        '<span class="brand-icon">🔬</span>'
        '<span class="brand-name">AEDA-AI</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.caption("Automated EDA for environmental data")
    st.sidebar.divider()

    # Navigation
    page_label = st.sidebar.radio(
        "Navigation",
        options=[label for label, _ in PAGES],
        label_visibility="collapsed",
    )

    # Status block
    st.sidebar.divider()
    _render_status_block()

    return page_label


def _render_status_block() -> None:
    """Show dataset status in the sidebar — empty state or current dataset."""
    results = st.session_state.results
    if results is None:
        st.sidebar.markdown(
            '<div class="status-empty">No dataset loaded</div>',
            unsafe_allow_html=True,
        )
        return

    filename = st.session_state.filename or "—"
    st.sidebar.markdown("**Current dataset**")
    st.sidebar.caption(f"📄 {filename}")

    cols = st.sidebar.columns(2)
    cols[0].metric("Samples", results.raw_data.shape[0])
    cols[1].metric("Variables", results.raw_data.shape[1])

    if results.clustering is not None:
        st.sidebar.metric("Clusters", results.clustering.n_clusters)


def main() -> None:
    page_label = _render_sidebar()

    # Route to the selected page. Imports are lazy to keep page-load fast.
    page_module_path = dict(PAGES)[page_label]
    import importlib
    module = importlib.import_module(page_module_path)
    module.render()


if __name__ == "__main__":
    main()
```

---

## 4. Archivo MODIFICADO: `app/pages/results.py` — fix del bug de `return`

**Problema:** los tabs usan `return` para salir cuando un resultado falta;
esto sale de la función `render()` entera, cortando los demás tabs.
Cambiar a `if/else` para que cada tab sea independiente.

### 4.1 Cambiar el header al inicio de `render()`

**BUSCAR:**

```python
def render():
    st.header("Results")
```

**REEMPLAZAR POR:**

```python
def render():
    from app.components.page_header import page_header

    page_header(
        title="Results",
        description="Interactive dashboard: PCA, correlations, clusters and anomalies.",
        icon="📊",
    )
```

### 4.2 Tab PCA — quitar `return`

**BUSCAR** el bloque que comienza con `# TAB 1: PCA` y termina antes de `# TAB 2: CORRELATIONS`.

**REEMPLAZAR POR:**

```python
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
                color_by = st.selectbox(
                    "Color by", options=["None"] + categorical_cols,
                    index=1 if categorical_cols else 0,
                    key="pca_color",
                )
            with col2:
                n_loadings = st.slider("Loading arrows", min_value=5, max_value=25, value=12, key="pca_loadings")
            with col3:
                n_comp = results.dim_reduction.n_components_selected
                pc_x = st.selectbox("X axis", options=list(range(1, n_comp + 1)), index=0, key="pca_x")
                pc_y = st.selectbox("Y axis", options=list(range(1, n_comp + 1)), index=1, key="pca_y")

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

### 4.3 Tab Correlations — quitar `return`

**BUSCAR** el bloque `# TAB 2: CORRELATIONS` hasta antes de `# TAB 3: CLUSTERING`.

**REEMPLAZAR POR:**

```python
    # ============================================================
    # TAB 2: CORRELATIONS
    # ============================================================
    with tab_corr:
        if results.correlations is None:
            st.warning("Correlation analysis was not executed.")
        else:
            from aeda.viz.correlations import correlation_heatmap, cross_correlation_heatmap

            st.subheader("Correlation matrix")

            col1, col2 = st.columns([1, 1])
            with col1:
                corr_method = st.selectbox(
                    "Method",
                    options=["pearson", "spearman"],
                    index=0,
                    key="corr_method",
                )
            with col2:
                reorder = st.checkbox("Cluster-reorder axes", value=True, key="corr_reorder")

            if isinstance(results.correlations, dict) and corr_method in results.correlations:
                corr_result = results.correlations[corr_method]
            elif isinstance(results.correlations, dict) and "pearson" in results.correlations:
                corr_result = results.correlations["pearson"]
            else:
                corr_result = results.correlations

            fig = correlation_heatmap(corr_result, reorder=reorder)
            st.plotly_chart(fig, use_container_width=True)

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
                        raw_df,
                        group_a=plan.profile.heavy_metal_cols,
                        group_b=plan.profile.granulometry_cols,
                        method="spearman",
                    )
                    st.plotly_chart(fig_cross, use_container_width=True)
```

### 4.4 Tabs Clustering + Anomalies — quitar `return`

**BUSCAR** el bloque desde `# TAB 3: CLUSTERING` hasta el final de la función.

**REEMPLAZAR POR:**

```python
    # ============================================================
    # TAB 3: CLUSTERING
    # ============================================================
    with tab_cluster:
        if results.clustering is None:
            st.warning("Clustering was not executed.")
        elif results.dim_reduction is None:
            st.warning("Dimensionality reduction needed for cluster visualization.")
        else:
            from aeda.viz.clustering import cluster_scatter, cluster_composition

            st.subheader("Cluster analysis")

            compare_col = st.selectbox(
                "Compare clusters with",
                options=["None"] + categorical_cols,
                index=1 if categorical_cols else 0,
                key="cluster_compare",
            )

            fig = cluster_scatter(
                results.clustering,
                results.dim_reduction,
                df=raw_df,
                compare_with=compare_col if compare_col != "None" else None,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Cluster metrics
            metrics = results.clustering.metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Clusters", results.clustering.n_clusters)
            sil = metrics.get("silhouette")
            col2.metric("Silhouette score", f"{sil:.3f}" if sil else "—")
            db = metrics.get("davies_bouldin")
            col3.metric("Davies-Bouldin", f"{db:.3f}" if db else "—")

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

    # ============================================================
    # TAB 4: ANOMALIES
    # ============================================================
    with tab_anomaly:
        if results.anomalies is None:
            st.warning("Anomaly detection was not executed.")
        else:
            st.subheader("Anomaly detection")

            col1, col2 = st.columns(2)
            col1.metric("Method", results.anomalies.method)
            col2.metric("Anomalies detected", results.anomalies.n_anomalies)

            if results.anomalies.n_anomalies > 0 and results.dim_reduction is not None:
                import plotly.graph_objects as go
                from aeda.viz.base import apply_default_layout

                scores = results.dim_reduction.components
                x_col, y_col = scores.columns[0], scores.columns[1]
                is_anomaly = results.anomalies.is_anomaly

                fig = go.Figure()

                # Normal points
                normal_mask = ~is_anomaly
                fig.add_trace(go.Scatter(
                    x=scores.loc[normal_mask, x_col],
                    y=scores.loc[normal_mask, y_col],
                    mode="markers", name="Normal",
                    marker=dict(size=7, color="#2E4057", opacity=0.5),
                ))

                # Anomalous points
                anomaly_mask = is_anomaly
                fig.add_trace(go.Scatter(
                    x=scores.loc[anomaly_mask, x_col],
                    y=scores.loc[anomaly_mask, y_col],
                    mode="markers", name="Anomaly",
                    marker=dict(size=10, color="#A32D2D", symbol="diamond",
                                line=dict(width=1, color="white")),
                ))

                apply_default_layout(fig, title="Anomalies in PCA space")
                fig.update_xaxes(title=x_col)
                fig.update_yaxes(title=y_col)
                st.plotly_chart(fig, use_container_width=True)

            # Anomaly details
            if results.anomalies.n_anomalies > 0:
                with st.expander("Anomalous samples"):
                    import pandas as pd
                    anomaly_idx = results.anomalies.anomaly_indices
                    if raw_df is not None and len(anomaly_idx) > 0:
                        anomaly_rows = raw_df.loc[anomaly_idx]
                        st.dataframe(anomaly_rows, use_container_width=True)
```

---

## 5. Archivos MODIFICADOS: cambios pequeños en las otras 5 páginas

### 5.1 `app/pages/upload.py`

**Cambio A — header:**

BUSCAR:

```python
def render():
    st.header("Upload & Configure")
    st.write("Upload your environmental dataset and configure the analysis.")
```

REEMPLAZAR POR:

```python
def render():
    from app.components.page_header import page_header

    page_header(
        title="Upload & Configure",
        description="Upload your environmental dataset and run the analysis with one click.",
        icon="📤",
    )
```

**Cambio B — error al leer archivo:**

BUSCAR:

```python
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return
```

REEMPLAZAR POR:

```python
    except Exception as e:
        from app.components.errors import show_error
        show_error(
            "Could not read the uploaded file. Make sure it is a valid Excel/CSV "
            "and that the selected sheet contains tabular data.",
            exc=e,
        )
        return
```

**Cambio C — error en pipeline:**

BUSCAR:

```python
    except Exception as e:
        progress.empty()
        st.error(f"Pipeline failed: {e}")
        st.exception(e)
```

REEMPLAZAR POR:

```python
    except Exception as e:
        progress.empty()
        from app.components.errors import show_error
        show_error(
            "The pipeline could not complete. The dataset may have an unexpected "
            "structure or required columns may be missing.",
            exc=e,
        )
```

### 5.2 `app/pages/plan.py`

BUSCAR:

```python
def render():
    st.header("Analysis Plan")
```

REEMPLAZAR POR:

```python
def render():
    from app.components.page_header import page_header

    page_header(
        title="Analysis Plan",
        description="What the system decided to do with your dataset, and why.",
        icon="🧭",
    )
```

### 5.3 `app/pages/depth.py`

BUSCAR:

```python
def render():
    st.header("Depth profiles")
```

REEMPLAZAR POR:

```python
def render():
    from app.components.page_header import page_header

    page_header(
        title="Depth Profiles",
        description="Concentration vs. depth — sediment cores read as temporal series.",
        icon="🌊",
    )
```

### 5.4 `app/pages/audit.py`

BUSCAR:

```python
def render():
    st.header("Audit")
    st.caption(
        "Trace of every decision the pipeline made on this dataset. "
        "Use this page to verify the methodology and defend each choice."
    )
```

REEMPLAZAR POR:

```python
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
```

### 5.5 `app/pages/advanced.py`

**Cambio A — header:**

BUSCAR:

```python
def render():
    st.header("Advanced configuration")
    st.caption(
        "Re-run the analysis on the currently loaded dataset with custom "
        "parameters. Useful for sensitivity analysis and for the scientific "
        "tutor to validate alternative methodological choices."
    )
```

REEMPLAZAR POR:

```python
def render():
    from app.components.page_header import page_header

    page_header(
        title="Advanced Configuration",
        description=(
            "Re-run the analysis on the currently loaded dataset with custom "
            "parameters. Useful for sensitivity analysis and to validate "
            "alternative methodological choices."
        ),
        icon="⚙️",
    )
```

**Cambio B — error en re-run:**

BUSCAR:

```python
    except Exception as e:
        progress.empty()
        st.error(f"Re-run failed: {type(e).__name__}: {e}")
        st.exception(e)
```

REEMPLAZAR POR:

```python
    except Exception as e:
        progress.empty()
        from app.components.errors import show_error
        show_error(
            "The pipeline could not complete with these settings. "
            "Try a different combination of parameters.",
            exc=e,
        )
```

---

## 6. Validación

Ejecutar en orden, desde la raíz del proyecto:

```bash
# 1. Suite completa de tests — debe seguir verde (esto es UI, no toca el engine)
pytest tests/ -q
```

**Esperado:** `38 passed in ~25s` (los 33 originales + los 5 del fix de
`_detect_depth_gradient` agregados en el prompt anterior).

```bash
# 2. Validar que todos los módulos de la app importan sin errores de sintaxis
python -c "
import sys
sys.path.insert(0, '.')
import importlib
modules = [
    'app.theme',
    'app.components.page_header',
    'app.components.errors',
    'app.main',
    'app.pages.upload',
    'app.pages.plan',
    'app.pages.results',
    'app.pages.depth',
    'app.pages.audit',
    'app.pages.advanced',
]
for m in modules:
    importlib.import_module(m)
    print(f'OK  {m}')
print('All modules import cleanly.')
"
```

**Esperado:** todas las líneas terminando con `OK` y `All modules import cleanly.`
(Los warnings de Streamlit `ScriptRunContext` son normales fuera del runtime).

```bash
# 3. Smoke test visual — arrancar la app
streamlit run app/main.py
```

**Verificación visual** (en el navegador):

- ✅ El sidebar muestra `🔬 AEDA-AI` con tipografía bold y subtitle "Automated EDA for environmental data".
- ✅ El fondo de la app es beige cálido (`#FAF8F4`), no blanco puro.
- ✅ Los `st.metric` cards tienen borde fino y padding.
- ✅ Cada página muestra el icono + título + descripción + divider al inicio.
- ✅ Si subís un dataset y vas a Results, los 4 tabs (PCA, Correlations, Clustering, Anomalies) son navegables independientemente — antes si PCA fallaba se cortaba toda la página.
- ✅ Si forzás un error subiendo un Excel inválido, el mensaje aparece en lenguaje claro y el traceback queda en un expander cerrado por defecto.

---

## 7. Si algo falla

- Si los 38 tests dejan de pasar → el cambio tocó algo del engine por error.
  Detenerse y reportar el output de `pytest tests/ -v`.
- Si `streamlit run` arranca pero la página queda en blanco → revisar que
  `apply_theme()` esté llamado **después** de `st.set_page_config()`.
- Si los emojis del header se ven raros → el archivo debe estar guardado en
  UTF-8. En VS Code, abajo a la derecha debería decir `UTF-8`.
- No tocar `aeda/`, `tests/`, ni `pyproject.toml`. Esto es solo `app/` y
  `.streamlit/`.

---

## 8. Mensaje de commit sugerido

```
feat(ui): visual foundation — earth-science theme + reusable components

Establishes the AEDA-AI visual identity:
- Earth-science palette (olive/ocean/terracotta) with named constants in
  app/theme.py for use across the app and Plotly figures.
- CSS layer applied at startup that refines typography, sidebar, metric
  cards, expanders and tabs without fighting the Streamlit base theme.
- New reusable components: page_header() and show_error() / show_warning().
- Sidebar redesigned with branded 🔬 + AEDA-AI header and structured
  dataset status block.
- All 6 pages now use the same page_header() pattern.
- Error handling no longer dumps raw tracebacks; technical detail moved
  behind an expander.

Bug fixes:
- results.py: tab guards used `return` which exited the entire render()
  function instead of just the failing tab. Switched to if/else so each
  tab (PCA, Correlations, Clustering, Anomalies) is independent.

No engine changes. 38 tests still pass.
```
