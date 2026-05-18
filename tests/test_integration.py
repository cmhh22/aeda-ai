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


def test_trace_elements_includes_isovida_metals():
    """Trace elements must include all metals present in ISOVIDA."""
    from aeda.engine.auto_selector import TRACE_ELEMENTS

    required = {"Cr", "Co", "Ni", "Cu", "Zn", "As", "Mo", "Pb"}
    missing = required - TRACE_ELEMENTS
    assert not missing, f"Missing trace elements in ISOVIDA: {missing}"


def test_ancillary_variables_renamed():
    """Sediment indicators were renamed to ancillary variables (Yoelvis feedback)."""
    from aeda.engine import auto_selector

    assert hasattr(auto_selector, "ANCILLARY_VARIABLES")
    expected = {"TOC", "OM", "PPI550", "PPI950", "HC", "CaCO3"}
    assert auto_selector.ANCILLARY_VARIABLES == expected


def test_frx_typical_rule_removed():
    """Rule 7 (FRX typical detection by sum) was removed per Yoelvis feedback."""
    import inspect

    from aeda.engine.auto_selector import profile_dataset

    src = inspect.getsource(profile_dataset)
    assert "frx_typical" not in src.lower(), "Rule 7 (FRX typical sum) should be removed"


def test_correlation_block_threshold_is_06():
    """Per Yoelvis feedback, threshold lowered from 0.7 to 0.6."""
    from aeda.engine.auto_selector import CORRELATION_BLOCK_THRESHOLD

    assert CORRELATION_BLOCK_THRESHOLD == 0.6


def test_mixed_units_from_dictionary_preferred_over_heuristic():
    """When a units dictionary is provided, it takes precedence over the heuristic."""
    from aeda.engine.auto_selector import detect_mixed_units

    df = pd.DataFrame(
        {
            "Al": [5.0, 6.0, 7.0],
            "Pb": [10.0, 12.0, 15.0],
        }
    )
    units = {"Al": "%", "Pb": "mg/kg"}
    result = detect_mixed_units(df, major_cols=["Al"], trace_cols=["Pb"], units_dict=units)
    assert result["mixed_units_detected"] is True
    assert result["method"] == "dictionary"


def test_mixed_units_falls_back_to_heuristic_when_no_dictionary():
    """When no units dictionary is available, fall back to numeric heuristic."""
    from aeda.engine.auto_selector import detect_mixed_units

    df = pd.DataFrame(
        {
            "Al": [5.0, 6.0, 7.0],
            "Pb": [40.0, 50.0, 60.0],
        }
    )
    result = detect_mixed_units(df, major_cols=["Al"], trace_cols=["Pb"], units_dict=None)
    assert result["method"] == "heuristic"


def test_isovida_units_loaded_from_dictionary():
    """Integration: ISOVIDA dataset dictionary specifies units for each variable."""
    EXCLUDE = [
        "No",
        "Code",
        "Site_Name",
        "Pret_Code",
        "Código_muestra",
        "Sitio_muestreo",
        "Fecha_muestreo",
        "Core",
        "Latitud",
        "Longitud",
        "Profundidad",
    ]
    p = AEDAPipeline(impute_strategy="median")
    r = p.run(
        "data/BD_ISOVIDA_MANGLARES2023_rectificadaYBA_230326.xlsx",
        exclude_cols=EXCLUDE,
        sheet_name="DATA",
    )
    assert hasattr(r.dataset_info, "units")
    units = r.dataset_info.units
    if units:
        major_in_pct = any("%" in units.get(c, "") for c in ["Al", "Fe", "Si", "Ca"] if c in units)
        assert major_in_pct, "Expected at least one major element to be in % per ISOVIDA dictionary"


def test_clr_not_applied_automatically():
    """Per Yoelvis feedback, CLR must NOT appear in preprocessing log unless explicitly enabled."""
    import os
    import tempfile

    rng = np.random.default_rng(42)
    n = 50
    a = rng.uniform(20, 40, n)
    b = rng.uniform(20, 40, n)
    c = 100 - a - b
    other = rng.normal(50, 10, n)
    df = pd.DataFrame({"clay": a, "silt": b, "sand": c, "other": other})

    p = AEDAPipeline(impute_strategy="median", apply_clr=False, run_interpretation=False)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        path = f.name
    try:
        r = p.run(path)
        clr_entries = [
            s for s in (r.preprocessing_log.steps if r.preprocessing_log else [])
            if "clr" in str(s).lower()
        ]
        assert len(clr_entries) == 0, "CLR should not appear in log when apply_clr=False"
    finally:
        os.unlink(path)


def test_clr_applied_when_explicitly_enabled():
    """When apply_clr=True, CLR should appear in preprocessing log."""
    import os
    import tempfile

    rng = np.random.default_rng(42)
    n = 50
    a = rng.uniform(20, 40, n)
    b = rng.uniform(20, 40, n)
    df = pd.DataFrame({"clay": a, "silt": b, "sand": 100 - a - b, "other": rng.normal(50, 10, n)})
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        path = f.name
    try:
        p = AEDAPipeline(impute_strategy="median", apply_clr=True, run_interpretation=False)
        r = p.run(path)
        clr_entries = [
            s for s in (r.preprocessing_log.steps if r.preprocessing_log else [])
            if "clr" in str(s).lower()
        ]
        assert len(clr_entries) > 0, "CLR should appear in log when apply_clr=True"
    finally:
        os.unlink(path)
