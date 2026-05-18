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
# in marine and estuarine sediment literature; 5 and 20 cm are alternative
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
