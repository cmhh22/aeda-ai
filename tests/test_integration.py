# -*- coding: utf-8 -*-
"""Integration tests for critical default-pipeline regression scenarios."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aeda.pipeline.runner import AEDAPipeline


def _build_realistic_df(n: int = 100) -> pd.DataFrame:
    """Build synthetic environmental-like data with structure and missingness."""
    rng = np.random.default_rng(42)

    df = pd.DataFrame(
        {
            # Major elements in %
            "Al": rng.normal(7.0, 1.5, size=n),
            "Fe": rng.normal(4.5, 1.0, size=n),
            "Ca": rng.normal(5.0, 2.0, size=n),
            "K": rng.normal(2.0, 0.5, size=n),
            "Mg": rng.normal(1.5, 0.3, size=n),
            # Trace elements in mg/kg (skewed)
            "Cr": rng.lognormal(4.0, 0.5, size=n),
            "Cu": rng.lognormal(3.5, 0.6, size=n),
            "Ni": rng.lognormal(3.8, 0.4, size=n),
            "Pb": rng.lognormal(3.2, 0.5, size=n),
            "Zn": rng.lognormal(4.5, 0.5, size=n),
            "As": rng.lognormal(2.5, 0.7, size=n),
            # Compositional granulometry (sum to 100%)
            "clay": rng.uniform(20, 40, size=n),
            "silt": rng.uniform(30, 50, size=n),
        }
    )
    # Close the composition
    df["sand"] = 100 - df["clay"] - df["silt"]

    # Structured missingness: first 21 rows lack granulometry
    df.loc[:20, ["clay", "silt", "sand"]] = np.nan

    return df


def test_pipeline_defaults_produce_all_results(tmp_path):
    """With all defaults, every pipeline stage should produce a result."""
    df = _build_realistic_df()
    filepath = tmp_path / "synthetic.xlsx"
    df.to_excel(filepath, index=False, sheet_name="DATA")

    pipeline = AEDAPipeline()
    results = pipeline.run(str(filepath), sheet_name="DATA")

    assert results.raw_data is not None, "raw_data should not be None"
    assert results.processed_data is not None, "processed_data should not be None"
    assert results.plan is not None, "analysis plan should not be None"

    # Regression checks for prior silent failures
    assert results.dim_reduction is not None, "dim_reduction failed silently"
    assert results.clustering is not None, "clustering failed silently"
    assert results.anomalies is not None, "anomalies failed silently"
    assert results.correlations is not None, "correlations failed silently"
    assert results.feature_importance is not None, "feature_importance failed silently"


def test_default_dim_method_is_pca():
    """With method='auto', PCA should be selected by default."""
    df = _build_realistic_df(n=200).dropna()

    from aeda.io.preprocessor import preprocess
    from aeda.engine.dimensionality import reduce

    processed, _, _ = preprocess(df, scale_method="standard", impute_strategy="median")
    result = reduce(processed, method="auto")

    assert result.method == "PCA", f"Expected PCA default, got {result.method}"


def test_cluster_auto_with_kmeans_kwargs():
    """Regression: cluster(method='auto') must not pass K-Means kwargs to DBSCAN."""
    from aeda.engine.clustering import cluster

    rng = np.random.default_rng(42)
    df = pd.DataFrame(rng.normal(size=(50, 3)), columns=list("abc"))

    # k_range belongs to K-Means, not DBSCAN.
    # Before this fix, this call would raise TypeError.
    result = cluster(df, method="auto", k_range=(2, 8))

    assert result is not None
    assert result.diagnostics.get("auto_selected") is True
    assert "compared_methods" in result.diagnostics


def test_cluster_auto_with_dbscan_kwargs():
    """Regression: cluster(method='auto') must accept DBSCAN-specific kwargs without breaking K-Means."""
    from aeda.engine.clustering import cluster

    rng = np.random.default_rng(42)
    df = pd.DataFrame(rng.normal(size=(50, 3)), columns=list("abc"))

    # min_samples belongs to DBSCAN, not K-Means.
    result = cluster(df, method="auto", min_samples=10)

    assert result is not None
    assert result.diagnostics.get("auto_selected") is True


def test_cluster_explicit_method_filters_kwargs():
    """Explicit methods must also tolerate kwargs that don't belong to them."""
    from aeda.engine.clustering import cluster

    rng = np.random.default_rng(42)
    df = pd.DataFrame(rng.normal(size=(50, 3)), columns=list("abc"))

    # Pass DBSCAN's min_samples to explicit kmeans — must be silently ignored.
    result = cluster(df, method="kmeans", n_clusters=3, min_samples=10)
    assert result.method == "K-Means"
    assert result.n_clusters == 3


def test_feature_importance_values_are_sane():
    """Fallback feature ranking should remain finite and non-explosive."""
    df = _build_realistic_df(n=100).dropna()

    from aeda.io.preprocessor import preprocess
    from aeda.engine.feature_analysis import analyze_features

    processed, _, _ = preprocess(df, scale_method="standard", impute_strategy="median")
    # No cluster labels -> fallback to variance ranking
    result = analyze_features(processed)

    max_value = result.importances.max()
    assert max_value < 100, f"Importance values too large: max={max_value}"
    assert not result.importances.isna().any(), "Importance contains NaN values"


def test_unknown_impute_strategy_raises():
    """Unknown preprocessor strategy should raise instead of failing silently."""
    df = _build_realistic_df(n=50)

    from aeda.io.preprocessor import preprocess

    with pytest.raises(ValueError, match="Unsupported impute_strategy"):
        preprocess(df, impute_strategy="bogus_strategy")
