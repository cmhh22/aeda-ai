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
