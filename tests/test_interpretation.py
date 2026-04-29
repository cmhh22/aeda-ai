"""Regression tests for the environmental interpretation module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aeda.interpretation import (
    EFClass,
    TELPELClass,
    TEL_PEL_MARINE_SEDIMENT,
    build_interpretation_report,
    classify_ef_birch,
    classify_tel_pel,
    compute_enrichment_factor,
    handle_lod_values,
)


def _synthetic_sediment_data(n_per_site: int = 20, n_sites: int = 3):
    """Build synthetic sediment data: multiple sites, depth gradient, Pb trend."""
    rng = np.random.default_rng(42)
    rows = []
    for site_idx in range(n_sites):
        site_name = f"Site_{site_idx}"
        for depth in np.linspace(0, 50, n_per_site):
            pb_surface_factor = 1 + 4 * (1 - depth / 50)
            rows.append(
                {
                    "Site_Name": site_name,
                    "Depth": depth,
                    "Al": 7.0 + rng.normal(0, 0.5),
                    "Fe": 4.5 + rng.normal(0, 0.3),
                    "Pb": 30 * pb_surface_factor + rng.normal(0, 3),
                    "Cr": 55 + rng.normal(0, 5),
                    "Cu": 15 + rng.normal(0, 2),
                    "Zn": 100 * pb_surface_factor ** 0.5 + rng.normal(0, 5),
                }
            )
    return pd.DataFrame(rows)


def test_lod_imputation_normal_uses_half_lod():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"X": rng.normal(10, 1, size=200)})
    df.loc[:10, "X"] = np.nan
    out, log = handle_lod_values(df, lod_values={"X": 5.0})
    assert log.variable_decisions["X"]["method"].startswith("LOD/2")
    imputed_vals = out.loc[:10, "X"].unique()
    assert len(imputed_vals) == 1
    assert abs(imputed_vals[0] - 2.5) < 1e-9


def test_lod_imputation_lognormal_uses_sqrt_lod():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"X": rng.lognormal(0, 1.5, size=200)})
    df.loc[:10, "X"] = np.nan
    out, log = handle_lod_values(df, lod_values={"X": 4.0})
    assert "sqrt" in log.variable_decisions["X"]["method"]
    expected = 4.0 / np.sqrt(2.0)
    imputed_vals = out.loc[:10, "X"].unique()
    assert abs(imputed_vals[0] - expected) < 1e-9


def test_lod_raises_on_negative_lod():
    df = pd.DataFrame({"X": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="must be positive"):
        handle_lod_values(df, lod_values={"X": -1.0})


def test_tel_pel_classification_bands():
    concentrations = pd.Series([10, 50, 200, np.nan], name="Pb")
    result = classify_tel_pel(concentrations, "Pb")
    assert result.iloc[0] == TELPELClass.BELOW_TEL.value
    assert result.iloc[1] == TELPELClass.BETWEEN_TEL_PEL.value
    assert result.iloc[2] == TELPELClass.ABOVE_PEL.value
    assert pd.isna(result.iloc[3])


def test_tel_pel_unknown_metal():
    result = classify_tel_pel(pd.Series([10, 20]), "Unobtanium")
    assert all(v == TELPELClass.NO_THRESHOLDS.value for v in result)


def test_tel_pel_has_required_metals():
    required = {"As", "Cd", "Cr", "Cu", "Hg", "Ni", "Pb", "Zn"}
    assert required.issubset(TEL_PEL_MARINE_SEDIMENT.keys())


def test_ef_classification_birch_bands():
    ef = pd.Series([1.5, 2.5, 4.0, 8.0, 20.0, 40.0, 100.0])
    result = classify_ef_birch(ef)
    assert result.iloc[0] == EFClass.NO_ENRICHMENT.value
    assert result.iloc[1] == EFClass.MINOR.value
    assert result.iloc[2] == EFClass.MODERATE.value
    assert result.iloc[3] == EFClass.MODERATELY_SEVERE.value
    assert result.iloc[4] == EFClass.SEVERE.value
    assert result.iloc[5] == EFClass.VERY_SEVERE.value
    assert result.iloc[6] == EFClass.EXTREMELY_SEVERE.value


def test_ef_uses_deepest_section_as_baseline():
    df = _synthetic_sediment_data()
    result = compute_enrichment_factor(
        df,
        metals=["Pb", "Cr", "Cu", "Zn"],
        reference_element="Al",
        site_col="Site_Name",
        depth_col="Depth",
        baseline_strategy="deepest",
    )

    surface = df[df["Depth"] < 3].index
    deep = df[df["Depth"] > 45].index

    for site_idx in range(3):
        site_df = df[df["Site_Name"] == f"Site_{site_idx}"]
        deepest_row = site_df.loc[site_df["Depth"].idxmax()]
        ef_at_baseline = result.ef_values.loc[deepest_row.name, "Pb"]
        assert abs(ef_at_baseline - 1.0) < 1e-9

    surface_pb_ef_mean = result.ef_values.loc[surface, "Pb"].mean()
    deep_pb_ef_mean = result.ef_values.loc[deep, "Pb"].mean()
    assert surface_pb_ef_mean > deep_pb_ef_mean
    assert surface_pb_ef_mean > 2.5


def test_ef_rejects_reference_in_metals():
    df = _synthetic_sediment_data()
    with pytest.raises(ValueError, match="cannot be in the metals list"):
        compute_enrichment_factor(
            df,
            metals=["Pb", "Al"],
            reference_element="Al",
            site_col="Site_Name",
            depth_col="Depth",
        )


def test_ef_rejects_missing_reference():
    df = _synthetic_sediment_data()
    with pytest.raises(ValueError, match="not found"):
        compute_enrichment_factor(
            df,
            metals=["Pb"],
            reference_element="Scandium",
            site_col="Site_Name",
            depth_col="Depth",
        )


def test_ef_strategy_deepest_requires_depth_col():
    df = _synthetic_sediment_data()
    with pytest.raises(ValueError, match="deepest.*depth_col"):
        compute_enrichment_factor(
            df,
            metals=["Pb"],
            reference_element="Al",
            baseline_strategy="deepest",
            depth_col=None,
        )


def test_ef_custom_baseline():
    df = _synthetic_sediment_data()
    custom = {"__global__": {"Al": 7.0, "Pb": 30.0}}
    result = compute_enrichment_factor(
        df,
        metals=["Pb"],
        reference_element="Al",
        baseline_strategy="user",
        custom_baseline=custom,
    )
    surface_ef = result.ef_values.loc[df[df["Depth"] < 3].index, "Pb"].mean()
    assert 3 < surface_ef < 7


def test_interpretation_report_end_to_end():
    df = _synthetic_sediment_data()
    report = build_interpretation_report(
        df,
        metals=["Pb", "Cr", "Cu", "Zn"],
        reference_element="Al",
        site_col="Site_Name",
        depth_col="Depth",
        baseline_strategy="deepest",
    )
    assert report.ef_result is not None
    assert report.ef_classifications is not None
    assert "Pb" in report.tel_pel_classifications.columns
    assert len(report.summary()) > 0

    surface_idx = df[df["Depth"] < 3].index
    pb_classes = report.tel_pel_classifications.loc[surface_idx, "Pb"]
    assert (pb_classes != TELPELClass.BELOW_TEL.value).sum() > 0


def test_ef_custom_baseline_per_site_uses_correct_site():
    """Regression: per-site custom_baseline must apply the right baseline per row."""
    df = pd.DataFrame({
        "site": ["A", "A", "B", "B", "C", "C"],
        "depth": [10, 50, 10, 50, 10, 50],
        "Al": [5.0, 6.0, 4.0, 5.0, 6.0, 7.0],
        "Pb": [30, 20, 50, 25, 100, 30],
    })
    custom = {
        "A": {"Al": 5.5, "Pb": 25.0},
        "B": {"Al": 4.5, "Pb": 30.0},
        "C": {"Al": 6.5, "Pb": 28.0},
    }
    result = compute_enrichment_factor(
        df, metals=["Pb"], reference_element="Al",
        site_col="site", depth_col="depth",
        baseline_strategy="user",
        custom_baseline=custom,
    )
    # Row 0 (site A, Al=5.0, Pb=30): EF = (30/5.0) / (25/5.5) = 6.0 / 4.5454... = 1.32
    # Row 4 (site C, Al=6.0, Pb=100): EF = (100/6.0) / (28/6.5) = 16.66... / 4.3076... = 3.87
    # If the bug is present, both rows would use baseline A and produce different (wrong) values.
    assert abs(result.ef_values.loc[0, "Pb"] - 1.32) < 0.01
    assert abs(result.ef_values.loc[4, "Pb"] - 3.87) < 0.01


def test_ef_custom_baseline_global_flat_dict_still_works():
    """Backwards compat: flat global custom_baseline must still work."""
    df = pd.DataFrame({
        "site": ["A", "A", "B", "B"],
        "depth": [10, 50, 10, 50],
        "Al": [5.0, 6.0, 4.0, 5.0],
        "Pb": [30, 20, 50, 25],
    })
    result = compute_enrichment_factor(
        df, metals=["Pb"], reference_element="Al",
        baseline_strategy="user",
        custom_baseline={"Al": 5.0, "Pb": 25.0},
    )
    assert result.ef_values["Pb"].notna().all()
    # Row 0: (30/5.0) / (25/5.0) = 6/5 = 1.2
    assert abs(result.ef_values.loc[0, "Pb"] - 1.2) < 0.01


def test_ef_custom_baseline_missing_site_raises():
    """custom_baseline must cover all sites in the dataset."""
    df = pd.DataFrame({
        "site": ["A", "B", "C"],
        "depth": [10, 10, 10],
        "Al": [5.0, 4.0, 6.0],
        "Pb": [30, 50, 100],
    })
    incomplete = {
        "A": {"Al": 5.0, "Pb": 25.0},
        "B": {"Al": 4.0, "Pb": 30.0},
        # missing "C"
    }
    with pytest.raises(ValueError, match="does not cover"):
        compute_enrichment_factor(
            df, metals=["Pb"], reference_element="Al",
            site_col="site", depth_col="depth",
            baseline_strategy="user",
            custom_baseline=incomplete,
        )


def test_ef_custom_baseline_missing_metal_raises():
    """Per-site baseline missing a required metal must raise."""
    df = pd.DataFrame({
        "site": ["A", "B"],
        "depth": [10, 10],
        "Al": [5.0, 4.0],
        "Pb": [30, 50],
    })
    bad = {
        "A": {"Al": 5.0},  # missing Pb
        "B": {"Al": 4.0, "Pb": 30.0},
    }
    with pytest.raises(ValueError, match="missing required keys"):
        compute_enrichment_factor(
            df, metals=["Pb"], reference_element="Al",
            site_col="site", depth_col="depth",
            baseline_strategy="user",
            custom_baseline=bad,
        )
