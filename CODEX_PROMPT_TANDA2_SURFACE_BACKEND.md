# CODEX_PROMPT_TANDA2_SURFACE_BACKEND

**Tipo:** Feature nueva — análisis espacial de fracción superficial (Yoelvis Tanda 2, etapa 1: backend)
**Archivos:** 4 modificados, 2 nuevos
**Tiempo estimado:** ~30 min
**Tests esperados después:** 71 passed (53 actuales + 18 nuevos)

---

## 1. Contexto

Yoelvis Bolaños (LEA-CEAC) aclaró que en estudios multi-sitio el análisis
**entre sitios** se hace típicamente sobre la **fracción superficial**
del sedimento (0-10 cm por defecto, con 5 o 20 cm como alternativas según
autor). La razón geoquímica: comparar tramos profundos entre sitios mezcla
épocas distintas y oscurece patrones espaciales actuales de contaminación.

Esta tanda implementa el backend del análisis. La UI viene en una etapa
posterior (`CODEX_PROMPT_TANDA2_SURFACE_UI.md`).

Decisiones tomadas con el tutor:

| Pregunta | Decisión | Justificación |
|---|---|---|
| Profundidad de la capa superficial | Default 10 cm, configurable | Yoelvis menciona 5/10/20 como cortes habituales |
| Agregación cuando hay varias muestras por sitio | **Promediar por sitio** (1 fila por sitio) | Standard en literatura inter-sitio: Birch (2003), Buchman (2008). Evita sesgo hacia sitios con más cores |
| Algoritmo de clustering | Hierarchical Ward | Más estable con N pequeño (3-15 sitios) que K-Means; el dendrograma es entregable reconocido |
| Mínimo de sitios para clustering | ≥ 3 | Por debajo, agrupar no tiene sentido |

Validado contra ISOVIDA: 7 sitios, 54 muestras a 10 cm, clustering en
3 grupos coherentes.

---

## 2. Archivo nuevo: `aeda/engine/spatial_surface.py`

Crear este archivo con el contenido íntegro siguiente:

```python
"""
Spatial analysis of the surface sediment layer.

Yoelvis Bolaños (LEA-CEAC) clarified that for multi-site environmental
studies, between-site chemistry comparisons are most meaningful when
restricted to the surface layer of each site (typically 0-10 cm, with
0-5 cm or 0-20 cm used by some authors). The surface layer represents
the most recent deposition; comparing deeper sections across sites
mixes different historical periods and obscures present-day spatial
patterns of contamination.

This module:
    1. Filters samples to those within ``max_depth_cm`` of the surface.
    2. Aggregates to one row per site (mean of available samples).
       This is the standard inter-site approach (Birch 2003;
       Buchman 2008) — using individual samples would bias clustering
       toward sites with more cores.
    3. Runs site-level hierarchical clustering (Ward) on the aggregated
       data and, if coordinates are available, attaches them so the UI
       can map the result.

The public entry point is :func:`surface_spatial_analysis`, which
returns a :class:`SurfaceAnalysisResult`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler


# Default surface-layer thresholds. 10 cm matches the most common cut
# in marine and estuarine sediment literature; 5 and 20 are alternative
# definitions encountered in some authors.
DEFAULT_SURFACE_DEPTH_CM = 10.0
COMMON_SURFACE_DEPTHS_CM = (5.0, 10.0, 20.0)

# Minimum number of sites required to attempt clustering.
MIN_SITES_FOR_CLUSTERING = 3


@dataclass
class SurfaceAnalysisResult:
    """Outcome of the surface-layer spatial analysis.

    Attributes
    ----------
    surface_data : DataFrame
        The raw rows that passed the depth filter (before aggregation).
        Useful when the UI wants to show "from which samples is each
        site mean computed".
    site_means : DataFrame
        One row per site, indexed by site name, containing the mean
        concentration of every measurement column across the site's
        surface samples. This is the matrix used for clustering.
    max_depth : float
        The depth threshold (cm) applied to filter samples.
    depth_col : str
        Name of the depth column used to filter.
    site_col : str
        Name of the column identifying sites.
    n_sites_with_data : int
        Number of distinct sites that had at least one sample in the
        surface layer.
    n_samples_in_surface : int
        Total number of samples (rows) that passed the depth filter.
    samples_per_site : dict[str, int]
        For each site, how many surface samples were averaged.
    site_clustering : dict or None
        Result of clustering site_means. ``None`` when fewer than
        ``MIN_SITES_FOR_CLUSTERING`` sites were available.
        Keys when present: ``labels`` (Series indexed by site),
        ``method``, ``n_clusters``, ``linkage``.
    site_coordinates : DataFrame or None
        Per-site mean of latitude/longitude (or whatever coordinate
        columns are passed in). Same index as ``site_means``.
    """

    surface_data: pd.DataFrame
    site_means: pd.DataFrame
    max_depth: float
    depth_col: str
    site_col: str
    n_sites_with_data: int
    n_samples_in_surface: int
    samples_per_site: dict
    site_clustering: Optional[dict] = None
    site_coordinates: Optional[pd.DataFrame] = None

    def summary(self) -> str:
        """Human-readable one-paragraph summary, suitable for the audit page."""
        lines = [
            f"Surface-layer spatial analysis (depth ≤ {self.max_depth:g} cm):",
            f"  Sites with surface data: {self.n_sites_with_data}",
            f"  Total surface samples: {self.n_samples_in_surface}",
        ]
        if self.samples_per_site:
            counts = ", ".join(
                f"{site}: {n}" for site, n in sorted(self.samples_per_site.items())
            )
            lines.append(f"  Samples per site: {counts}")
        if self.site_clustering and self.site_clustering.get("labels") is not None:
            method = self.site_clustering.get("method", "?")
            n_cl = self.site_clustering.get("n_clusters", 0)
            lines.append(f"  Site clustering: {method} → {n_cl} groups")
        return "\n".join(lines)


def filter_surface_layer(
    df: pd.DataFrame,
    depth_col: str,
    max_depth_cm: float = DEFAULT_SURFACE_DEPTH_CM,
) -> pd.DataFrame:
    """Return only the rows whose depth is ≤ ``max_depth_cm``.

    Negative depths and non-numeric values are dropped silently. The
    returned DataFrame keeps the same columns and is a copy (safe to
    modify downstream).

    Parameters
    ----------
    df : DataFrame
        Source dataframe; must contain ``depth_col``.
    depth_col : str
        Name of the depth column (usually 'Profundidad' or 'Depth').
    max_depth_cm : float
        Upper bound; samples with depth above this are excluded.

    Raises
    ------
    ValueError
        If ``depth_col`` is not in ``df.columns`` or ``max_depth_cm <= 0``.
    """
    if depth_col not in df.columns:
        raise ValueError(f"depth column {depth_col!r} not found in dataframe")
    if max_depth_cm <= 0:
        raise ValueError(f"max_depth_cm must be positive, got {max_depth_cm}")

    depths = pd.to_numeric(df[depth_col], errors="coerce")
    mask = depths.notna() & (depths >= 0) & (depths <= max_depth_cm)
    return df.loc[mask].copy()


def aggregate_by_site(
    surface_df: pd.DataFrame,
    site_col: str,
    measurement_cols: list,
) -> pd.DataFrame:
    """Return one row per site, with the mean of each measurement column.

    Standard inter-site approach in environmental sediment studies
    (Birch 2003; Buchman 2008). Using per-sample data would bias the
    spatial comparison toward sites with more cores.

    Parameters
    ----------
    surface_df : DataFrame
        Already filtered to the surface layer.
    site_col : str
        Column identifying the site (e.g. 'Sitio_muestreo').
    measurement_cols : list[str]
        Numeric measurement columns to average. Columns not present in
        ``surface_df`` are silently ignored.

    Returns
    -------
    DataFrame
        Indexed by site name, one numeric column per measurement.
        Sites where all measurements are NaN are dropped.

    Raises
    ------
    ValueError
        If ``site_col`` is missing or no measurement columns are found.
    """
    if site_col not in surface_df.columns:
        raise ValueError(f"site column {site_col!r} not found in dataframe")

    available_cols = [c for c in measurement_cols if c in surface_df.columns]
    if not available_cols:
        raise ValueError("none of the requested measurement columns are in surface_df")

    grouped = surface_df.groupby(site_col)[available_cols].mean(numeric_only=True)
    # Drop sites with no valid measurements at all
    grouped = grouped.dropna(how="all")
    return grouped


def _aggregate_coordinates(
    surface_df: pd.DataFrame,
    site_col: str,
    coordinate_cols: list,
    site_index: pd.Index,
) -> Optional[pd.DataFrame]:
    """Helper: average latitude/longitude per site, aligned to site_index.

    Returns ``None`` if no valid coordinate columns are present.
    """
    avail = [c for c in coordinate_cols if c in surface_df.columns]
    if not avail:
        return None
    coords = surface_df.groupby(site_col)[avail].mean(numeric_only=True)
    return coords.reindex(site_index)


def cluster_sites_hierarchical(
    site_means: pd.DataFrame,
    n_clusters: Optional[int] = None,
) -> dict:
    """Run hierarchical Ward clustering over per-site means.

    Hierarchical Ward is preferred here over K-Means because:
    - The number of "samples" (sites) is small (typically 3-15).
      Ward gives more stable groups in that regime.
    - The dendrogram is a recognized deliverable in environmental
      reports.
    - K-Means' centroid logic is less meaningful at this scale.

    Parameters
    ----------
    site_means : DataFrame
        Indexed by site, numeric columns only.
    n_clusters : int, optional
        If None, defaults to ``max(2, n_sites // 2)`` clamped below
        ``n_sites``.

    Returns
    -------
    dict
        With keys ``labels`` (Series indexed by site, NaN when n_sites
        is too small), ``method``, ``n_clusters``, ``linkage``,
        optionally ``note`` if clustering was skipped.
    """
    n_sites = len(site_means)
    if n_sites < MIN_SITES_FOR_CLUSTERING:
        return {
            "labels": None,
            "method": "hierarchical_ward",
            "n_clusters": 0,
            "linkage": "ward",
            "note": (
                f"Only {n_sites} site(s) — clustering skipped "
                f"(need ≥ {MIN_SITES_FOR_CLUSTERING})."
            ),
        }

    # Simple imputation for any remaining NaN cells. With aggregated
    # per-site means this is typically a non-issue, but stays safe.
    data = site_means.fillna(site_means.mean(numeric_only=True))
    data = data.dropna(axis=1, how="all")

    scaler = StandardScaler()
    X = scaler.fit_transform(data.values)

    if n_clusters is None:
        n_clusters = max(2, min(n_sites - 1, n_sites // 2))
    # Defensive clamp in case the caller passed something out of range
    n_clusters = max(2, min(n_clusters, n_sites - 1))

    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = pd.Series(
        model.fit_predict(X), index=site_means.index, name="surface_cluster"
    )
    return {
        "labels": labels,
        "method": "hierarchical_ward",
        "n_clusters": n_clusters,
        "linkage": "ward",
    }


def surface_spatial_analysis(
    df: pd.DataFrame,
    depth_col: str,
    site_col: str,
    measurement_cols: list,
    max_depth_cm: float = DEFAULT_SURFACE_DEPTH_CM,
    coordinate_cols: Optional[list] = None,
    n_clusters: Optional[int] = None,
) -> SurfaceAnalysisResult:
    """End-to-end surface-layer spatial analysis.

    Workflow:
        1. Filter to depth ≤ max_depth_cm.
        2. Aggregate to one row per site (means).
        3. Cluster sites (hierarchical Ward) when ≥ 3 sites have data.
        4. Compute mean coordinates per site if coordinate_cols is given.

    Parameters
    ----------
    df : DataFrame
        Full raw dataset. Must contain depth_col, site_col, and
        the listed measurement_cols.
    depth_col : str
    site_col : str
    measurement_cols : list[str]
    max_depth_cm : float
    coordinate_cols : list[str], optional
        E.g. ``['Latitud', 'Longitud']``.
    n_clusters : int, optional

    Returns
    -------
    SurfaceAnalysisResult
    """
    surface_df = filter_surface_layer(df, depth_col, max_depth_cm)

    if surface_df.empty:
        # No samples passed the filter — return an empty result rather
        # than raising so the caller can render an informative message.
        return SurfaceAnalysisResult(
            surface_data=surface_df,
            site_means=pd.DataFrame(),
            max_depth=max_depth_cm,
            depth_col=depth_col,
            site_col=site_col,
            n_sites_with_data=0,
            n_samples_in_surface=0,
            samples_per_site={},
            site_clustering=None,
            site_coordinates=None,
        )

    site_means = aggregate_by_site(surface_df, site_col, measurement_cols)

    clustering = None
    if len(site_means) >= MIN_SITES_FOR_CLUSTERING:
        try:
            clustering = cluster_sites_hierarchical(site_means, n_clusters=n_clusters)
        except Exception:
            # Clustering should never crash the rest of the pipeline.
            # If something goes wrong (e.g. degenerate dataframe), we
            # record None and the UI shows the site means without groups.
            clustering = None

    site_coords = None
    if coordinate_cols:
        site_coords = _aggregate_coordinates(
            surface_df, site_col, coordinate_cols, site_means.index
        )

    samples_per_site = surface_df.groupby(site_col).size().to_dict()

    return SurfaceAnalysisResult(
        surface_data=surface_df,
        site_means=site_means,
        max_depth=max_depth_cm,
        depth_col=depth_col,
        site_col=site_col,
        n_sites_with_data=len(site_means),
        n_samples_in_surface=len(surface_df),
        samples_per_site=samples_per_site,
        site_clustering=clustering,
        site_coordinates=site_coords,
    )
```

---

## 3. Cambios en `aeda/pipeline/runner.py`

### 3.1 Importar el módulo nuevo

**BUSCAR:**

```python
from aeda.interpretation import (
    InterpretationReport,
    build_interpretation_report,
)
```

**REEMPLAZAR POR:**

```python
from aeda.interpretation import (
    InterpretationReport,
    build_interpretation_report,
)
from aeda.engine.spatial_surface import (
    SurfaceAnalysisResult,
    surface_spatial_analysis,
    DEFAULT_SURFACE_DEPTH_CM,
)
```

### 3.2 Agregar campo `surface_analysis` al dataclass `AEDAResults`

**BUSCAR:**

```python
    # Environmental interpretation (EF, TEL/PEL, Birch)
    interpretation: Optional[InterpretationReport] = None
```

**REEMPLAZAR POR:**

```python
    # Environmental interpretation (EF, TEL/PEL, Birch)
    interpretation: Optional[InterpretationReport] = None

    # Spatial analysis of the surface layer (0-10 cm by default).
    # Runs after the interpretation step when the dataset has both a site
    # column and a depth column. ``None`` when those prerequisites are not
    # met or the filter yielded no samples.
    surface_analysis: Optional["SurfaceAnalysisResult"] = None
```

### 3.3 Agregar parámetro `surface_depth_cm` al constructor `AEDAPipeline`

**BUSCAR:**

```python
        # Fine-grained method kwargs (advanced/expert overrides). Each dict is
        # forwarded to the corresponding engine function, which already accepts
        # **kwargs and filters the values it understands. Default is None to
        # preserve backwards compatibility — when None, nothing is forwarded.
        dim_kwargs: Optional[dict] = None,
        clustering_kwargs: Optional[dict] = None,
        anomaly_kwargs: Optional[dict] = None,
    ):
        self.scale_method = scale_method
        self.impute_strategy = impute_strategy
        self.dim_method = dim_method
        self.clustering_method = clustering_method
        self.anomaly_method = anomaly_method
        self.correlation_method = correlation_method
        self.apply_clr = apply_clr
        self.contamination = contamination
        self.run_interpretation = run_interpretation
        self.reference_element = reference_element
        self.baseline_strategy = baseline_strategy
        self.custom_baseline = custom_baseline
        self.dim_kwargs = dim_kwargs or {}
        self.clustering_kwargs = clustering_kwargs or {}
        self.anomaly_kwargs = anomaly_kwargs or {}
```

**REEMPLAZAR POR:**

```python
        # Fine-grained method kwargs (advanced/expert overrides). Each dict is
        # forwarded to the corresponding engine function, which already accepts
        # **kwargs and filters the values it understands. Default is None to
        # preserve backwards compatibility — when None, nothing is forwarded.
        dim_kwargs: Optional[dict] = None,
        clustering_kwargs: Optional[dict] = None,
        anomaly_kwargs: Optional[dict] = None,
        # Surface-layer analysis: depth threshold (cm) used to filter samples
        # for the inter-site spatial comparison. Yoelvis (LEA-CEAC) recommends
        # 10 cm as the default; 5 and 20 cm are accepted alternatives. The
        # analysis itself only runs when the dataset has both a site column
        # and a depth column; otherwise this parameter is ignored.
        surface_depth_cm: float = DEFAULT_SURFACE_DEPTH_CM,
    ):
        self.scale_method = scale_method
        self.impute_strategy = impute_strategy
        self.dim_method = dim_method
        self.clustering_method = clustering_method
        self.anomaly_method = anomaly_method
        self.correlation_method = correlation_method
        self.apply_clr = apply_clr
        self.contamination = contamination
        self.run_interpretation = run_interpretation
        self.reference_element = reference_element
        self.baseline_strategy = baseline_strategy
        self.custom_baseline = custom_baseline
        self.dim_kwargs = dim_kwargs or {}
        self.clustering_kwargs = clustering_kwargs or {}
        self.anomaly_kwargs = anomaly_kwargs or {}
        self.surface_depth_cm = float(surface_depth_cm)
```

### 3.4 Ejecutar el análisis al final de `run()`

**BUSCAR:**

```python
            except Exception as e:
                logger.warning(
                    f"Environmental interpretation failed: {type(e).__name__}: {e}"
                )
                results.interpretation = None

        return results
```

**REEMPLAZAR POR:**

```python
            except Exception as e:
                logger.warning(
                    f"Environmental interpretation failed: {type(e).__name__}: {e}"
                )
                results.interpretation = None

        # 11. SURFACE-LAYER SPATIAL ANALYSIS
        # Only runs when both a site column and a depth column are available.
        # The analysis filters to the surface layer (default 10 cm), averages
        # by site, and clusters sites. See aeda/engine/spatial_surface.py.
        if info.site_col is not None and info.depth_col is not None:
            try:
                results.surface_analysis = surface_spatial_analysis(
                    df,
                    depth_col=info.depth_col,
                    site_col=info.site_col,
                    measurement_cols=info.measurement_cols,
                    max_depth_cm=self.surface_depth_cm,
                    coordinate_cols=info.coordinate_cols or None,
                )
            except Exception as e:
                logger.warning(
                    f"Surface spatial analysis failed: {type(e).__name__}: {e}"
                )
                results.surface_analysis = None

        return results
```

---

## 4. Cambios en `aeda/engine/auto_selector.py`

Agregar la recomendación del análisis espacial al plan.

### 4.1 Agregar `_recommend_spatial`

**BUSCAR** la línea exacta:

```python
def _recommend_clustering(p: DataProfile) -> list[MethodRecommendation]:
```

**REEMPLAZAR POR** (inserta una función nueva antes):

```python
def _recommend_spatial(p: DataProfile) -> list[MethodRecommendation]:
    """Recommend a surface-layer inter-site spatial analysis when prerequisites are met.

    The surface-layer analysis (default 0-10 cm) compares sites against each
    other using only their most recent sediment. It requires:
    - A site column (so there is something to group by), and
    - A depth column (so a surface layer can be defined), and
    - At least 3 distinct sites (otherwise clustering of sites is meaningless).

    Yoelvis Bolaños (LEA-CEAC) noted that this is the standard approach in
    inter-site environmental comparisons because deeper layers represent
    different historical periods and can confound present-day spatial
    patterns.
    """
    recs = []
    if p.has_sites and p.has_depth and p.n_sites >= 3:
        recs.append(MethodRecommendation(
            category="spatial",
            method="Surface-layer inter-site analysis (Hierarchical Ward)",
            params={
                "method": "surface_spatial",
                "max_depth_cm": 10.0,
                "aggregation": "site_mean",
                "clustering": "hierarchical_ward",
            },
            reason=(
                "Multi-site dataset with depth: compare sites using only their "
                "surface (recent) sediment to avoid mixing historical periods."
            ),
            priority=1, confidence=Confidence.HIGH,
            evidence=[
                f"{p.n_sites} sites available for inter-site comparison.",
                "Surface layer default is 0-10 cm (Yoelvis 2026, LEA-CEAC).",
                "Aggregates each site to its mean to avoid bias from cores with more samples.",
                "Standard approach: Birch (2003), Buchman (2008).",
            ],
        ))
    return recs


def _recommend_clustering(p: DataProfile) -> list[MethodRecommendation]:
```

### 4.2 Llamar la nueva función al ensamblar el plan

**BUSCAR:**

```python
    recs.extend(_recommend_preprocessing(profile))
    recs.extend(_recommend_dimensionality(profile))
    recs.extend(_recommend_clustering(profile))
    recs.extend(_recommend_anomaly(profile))
    recs.extend(_recommend_correlation(profile))
    recs.extend(_recommend_feature_analysis(profile))
```

**REEMPLAZAR POR:**

```python
    recs.extend(_recommend_preprocessing(profile))
    recs.extend(_recommend_dimensionality(profile))
    recs.extend(_recommend_clustering(profile))
    recs.extend(_recommend_anomaly(profile))
    recs.extend(_recommend_correlation(profile))
    recs.extend(_recommend_feature_analysis(profile))
    recs.extend(_recommend_spatial(profile))
```

---

## 5. Archivo nuevo: `tests/test_spatial_surface.py`

Crear este archivo con el contenido íntegro siguiente:

```python
"""Tests for ``aeda.engine.spatial_surface``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aeda.engine.spatial_surface import (
    DEFAULT_SURFACE_DEPTH_CM,
    COMMON_SURFACE_DEPTHS_CM,
    MIN_SITES_FOR_CLUSTERING,
    SurfaceAnalysisResult,
    aggregate_by_site,
    cluster_sites_hierarchical,
    filter_surface_layer,
    surface_spatial_analysis,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def toy_df() -> pd.DataFrame:
    """Small synthetic dataset: 4 sites, 4 depths each, 3 metals + coords."""
    rng = np.random.default_rng(seed=42)
    sites = ["S1", "S2", "S3", "S4"]
    depths = [2, 8, 18, 35]  # cm: 2 surface (≤10), 1 mid (≤20), 1 deep
    rows = []
    for s_i, site in enumerate(sites):
        for d in depths:
            rows.append({
                "site": site,
                "depth": float(d),
                "Pb": 10.0 + s_i * 20 + rng.normal(0, 1),
                "Zn": 50.0 + s_i * 15 + rng.normal(0, 2),
                "Cu": 5.0 + s_i * 3 + rng.normal(0, 0.5),
                "Lat": 22.0 + s_i * 0.01,
                "Lon": -80.0 - s_i * 0.01,
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# filter_surface_layer
# ----------------------------------------------------------------------


def test_filter_surface_layer_default_keeps_only_shallow(toy_df):
    filtered = filter_surface_layer(toy_df, depth_col="depth")
    # default is 10 cm: keeps depths 2 and 8 (2 per site × 4 = 8 rows)
    assert len(filtered) == 8
    assert (filtered["depth"] <= DEFAULT_SURFACE_DEPTH_CM).all()


def test_filter_surface_layer_custom_threshold(toy_df):
    filtered = filter_surface_layer(toy_df, depth_col="depth", max_depth_cm=20.0)
    # 20 cm keeps depths 2, 8, 18 = 3 per site × 4 = 12 rows
    assert len(filtered) == 12


def test_filter_surface_layer_drops_nan_and_negative(toy_df):
    df = toy_df.copy()
    df.loc[0, "depth"] = np.nan
    df.loc[1, "depth"] = -5.0
    filtered = filter_surface_layer(df, depth_col="depth")
    # Rows 0 and 1 must be gone
    assert 0 not in filtered.index
    assert 1 not in filtered.index


def test_filter_surface_layer_raises_on_missing_column(toy_df):
    with pytest.raises(ValueError, match="not found"):
        filter_surface_layer(toy_df, depth_col="nonexistent")


def test_filter_surface_layer_raises_on_nonpositive_threshold(toy_df):
    with pytest.raises(ValueError, match="positive"):
        filter_surface_layer(toy_df, depth_col="depth", max_depth_cm=0)


# ----------------------------------------------------------------------
# aggregate_by_site
# ----------------------------------------------------------------------


def test_aggregate_by_site_returns_one_row_per_site(toy_df):
    surface = filter_surface_layer(toy_df, "depth", max_depth_cm=10.0)
    agg = aggregate_by_site(surface, site_col="site",
                            measurement_cols=["Pb", "Zn", "Cu"])
    assert agg.shape == (4, 3)  # 4 sites × 3 metals
    assert agg.index.name == "site"
    assert set(agg.index) == {"S1", "S2", "S3", "S4"}


def test_aggregate_by_site_means_match_expectation(toy_df):
    """Verify each site mean is the average of its 2 surface samples."""
    surface = filter_surface_layer(toy_df, "depth", max_depth_cm=10.0)
    agg = aggregate_by_site(surface, "site", ["Pb"])
    for site in surface["site"].unique():
        expected = surface.loc[surface["site"] == site, "Pb"].mean()
        assert agg.loc[site, "Pb"] == pytest.approx(expected)


def test_aggregate_by_site_ignores_missing_measurement_cols(toy_df):
    surface = filter_surface_layer(toy_df, "depth")
    agg = aggregate_by_site(surface, "site", ["Pb", "doesnotexist"])
    assert "Pb" in agg.columns
    assert "doesnotexist" not in agg.columns


def test_aggregate_by_site_raises_on_missing_site_col(toy_df):
    surface = filter_surface_layer(toy_df, "depth")
    with pytest.raises(ValueError, match="site column"):
        aggregate_by_site(surface, "no_such_site", ["Pb"])


def test_aggregate_by_site_raises_when_no_measurement_cols_available(toy_df):
    surface = filter_surface_layer(toy_df, "depth")
    with pytest.raises(ValueError, match="measurement"):
        aggregate_by_site(surface, "site", ["does", "not", "exist"])


# ----------------------------------------------------------------------
# cluster_sites_hierarchical
# ----------------------------------------------------------------------


def test_cluster_sites_hierarchical_returns_labels_for_each_site(toy_df):
    surface = filter_surface_layer(toy_df, "depth")
    agg = aggregate_by_site(surface, "site", ["Pb", "Zn", "Cu"])
    out = cluster_sites_hierarchical(agg)
    assert out["labels"] is not None
    assert len(out["labels"]) == 4
    assert out["method"] == "hierarchical_ward"
    assert out["linkage"] == "ward"
    assert out["n_clusters"] >= 2


def test_cluster_sites_hierarchical_skips_when_too_few_sites():
    """Below MIN_SITES_FOR_CLUSTERING the function returns labels=None."""
    one_site = pd.DataFrame({"Pb": [10.0], "Zn": [50.0]}, index=["S1"])
    one_site.index.name = "site"
    out = cluster_sites_hierarchical(one_site)
    assert out["labels"] is None
    assert "note" in out
    assert out["n_clusters"] == 0


def test_cluster_sites_hierarchical_clamps_n_clusters(toy_df):
    """If the caller asks for more clusters than sites, the function clamps."""
    surface = filter_surface_layer(toy_df, "depth")
    agg = aggregate_by_site(surface, "site", ["Pb", "Zn", "Cu"])
    out = cluster_sites_hierarchical(agg, n_clusters=20)
    # Cannot exceed n_sites - 1 = 3
    assert out["n_clusters"] <= len(agg) - 1


# ----------------------------------------------------------------------
# surface_spatial_analysis (end-to-end)
# ----------------------------------------------------------------------


def test_surface_spatial_analysis_full_pipeline(toy_df):
    result = surface_spatial_analysis(
        toy_df,
        depth_col="depth",
        site_col="site",
        measurement_cols=["Pb", "Zn", "Cu"],
        coordinate_cols=["Lat", "Lon"],
    )
    assert isinstance(result, SurfaceAnalysisResult)
    assert result.n_sites_with_data == 4
    assert result.n_samples_in_surface == 8
    assert result.max_depth == DEFAULT_SURFACE_DEPTH_CM
    assert result.site_clustering is not None
    assert result.site_coordinates is not None
    assert result.site_coordinates.shape == (4, 2)


def test_surface_spatial_analysis_works_without_coordinates(toy_df):
    result = surface_spatial_analysis(
        toy_df, "depth", "site", ["Pb", "Zn", "Cu"],
        coordinate_cols=None,
    )
    assert result.site_coordinates is None
    assert result.n_sites_with_data == 4


def test_surface_spatial_analysis_returns_empty_when_no_samples_in_layer(toy_df):
    """If max_depth_cm is so small nothing passes, return an empty result."""
    result = surface_spatial_analysis(
        toy_df, "depth", "site", ["Pb", "Zn", "Cu"],
        max_depth_cm=0.5,
    )
    assert result.n_samples_in_surface == 0
    assert result.n_sites_with_data == 0
    assert result.site_means.empty
    assert result.site_clustering is None


def test_surface_spatial_analysis_skips_clustering_with_2_sites():
    """Two sites is below MIN_SITES_FOR_CLUSTERING."""
    df = pd.DataFrame({
        "site": ["A", "A", "B", "B"],
        "depth": [2.0, 8.0, 2.0, 8.0],
        "Pb": [10.0, 12.0, 30.0, 32.0],
    })
    result = surface_spatial_analysis(df, "depth", "site", ["Pb"])
    assert result.n_sites_with_data == 2
    # Clustering can be either None or a dict with labels=None
    if result.site_clustering is not None:
        assert result.site_clustering["labels"] is None


def test_surface_depth_thresholds_are_documented():
    """Sanity: the constants used as UI presets match the module's defaults."""
    assert DEFAULT_SURFACE_DEPTH_CM in COMMON_SURFACE_DEPTHS_CM
    assert MIN_SITES_FOR_CLUSTERING >= 3
```

---

## 6. Validación

```bash
# 1. Suite completa
pytest tests/ -q
```
**Esperado:** `71 passed` (53 actuales + 18 nuevos).

```bash
# 2. Sanity check imports
python -c "
from aeda.pipeline.runner import AEDAPipeline, AEDAResults
from aeda.engine.spatial_surface import (
    SurfaceAnalysisResult, surface_spatial_analysis,
    DEFAULT_SURFACE_DEPTH_CM, COMMON_SURFACE_DEPTHS_CM,
)
print('OK imports')
"
```

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
print(r.surface_analysis.summary())
print(f'spatial recs in plan: {len([x for x in r.plan.recommendations if x.category == \"spatial\"])}')
"
```
**Esperado:**
```
Surface-layer spatial analysis (depth ≤ 10 cm):
  Sites with surface data: 7
  Total surface samples: 54
  ...
  Site clustering: hierarchical_ward → 3 groups
spatial recs in plan: 1
```

---

## 7. Si algo falla

- Si los tests existentes empiezan a fallar → verificar que el cambio en
  `auto_selector.py` (agregar `_recommend_spatial` y la línea
  `recs.extend(_recommend_spatial(profile))`) se aplicó correctamente.
  El resto de cambios son aditivos y no deberían afectar nada existente.
- Si `AttributeError: 'AEDAResults' object has no attribute 'surface_analysis'`
  → falta agregar el campo al dataclass (paso 3.2).
- Si `ImportError: cannot import name 'surface_spatial_analysis'` →
  verificar que el archivo `aeda/engine/spatial_surface.py` se creó
  íntegramente (paso 2) y que el módulo está disponible para importar.
- No tocar `app/`, `aeda/interpretation/`, `aeda/io/`. Esta etapa es solo
  backend del análisis espacial.

---

## 8. Mensaje de commit sugerido

```
feat(engine): surface-layer inter-site spatial analysis (Yoelvis Tanda 2)

Implements Yoelvis Bolaños' (LEA-CEAC) recommendation for multi-site
sediment studies: compare sites using only their surface layer (0-10 cm
by default, configurable to 5/10/20 cm) so present-day spatial patterns
of contamination are not confounded by deeper, historically-different
samples.

New module aeda/engine/spatial_surface.py:
- filter_surface_layer: depth-based row filter
- aggregate_by_site: averages measurements per site (Birch 2003,
  Buchman 2008 standard — avoids bias from sites with more cores)
- cluster_sites_hierarchical: Ward clustering over per-site means;
  skips gracefully when fewer than 3 sites are available
- surface_spatial_analysis: end-to-end entry point
- SurfaceAnalysisResult dataclass

Pipeline integration (aeda/pipeline/runner.py):
- AEDAResults.surface_analysis field
- AEDAPipeline.surface_depth_cm constructor parameter (default 10.0)
- Runs after the interpretation step when site_col + depth_col are
  available; failures are logged but do not break the run

Brain integration (aeda/engine/auto_selector.py):
- New _recommend_spatial function emits a 'spatial' recommendation
  when multi-site + depth + ≥3 sites are detected
- Registered in the plan-assembly extension list

Tests: 18 new in tests/test_spatial_surface.py. Total 71 pass.

Validated on ISOVIDA: 7 sites, 54 samples in the 0-10 cm layer,
3 groups identified by hierarchical clustering.
```
