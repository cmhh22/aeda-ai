"""Tests for the upper continental crust reference module."""

import pytest


def test_crust_table_includes_main_metals():
    """Table must include the metals relevant to environmental sediment studies."""
    from aeda.interpretation.crust_reference import UPPER_CONTINENTAL_CRUST

    required = {"Al", "Fe", "Cr", "Cu", "Ni", "Pb", "Zn", "As", "Cd", "Hg", "Mn"}
    missing = required - set(UPPER_CONTINENTAL_CRUST.keys())
    assert not missing, f"Missing main metals in Rudnick & Gao table: {missing}"


def test_get_crust_reference_returns_dataclass():
    from aeda.interpretation.crust_reference import get_crust_reference, CrustReferenceValue

    ref = get_crust_reference("Pb")
    assert isinstance(ref, CrustReferenceValue)
    assert ref.unit == "mg/kg"
    assert ref.value == 17.0


def test_get_crust_reference_unknown_raises():
    from aeda.interpretation.crust_reference import get_crust_reference

    with pytest.raises(KeyError):
        get_crust_reference("Unobtanium")


def test_compare_to_crust_labels():
    from aeda.interpretation.crust_reference import compare_to_crust

    assert compare_to_crust(5.0, "Pb")["label"] == "below_crust"
    assert compare_to_crust(20.0, "Pb")["label"] == "similar_to_crust"
    assert compare_to_crust(100.0, "Pb")["label"] == "enriched"
    assert compare_to_crust(500.0, "Pb")["label"] == "highly_enriched"


def test_compare_unit_conversion_wtpct_to_mgkg():
    from aeda.interpretation.crust_reference import compare_to_crust

    result = compare_to_crust(40000.0, "Al", sample_unit="mg/kg")
    assert 0.4 < result["ratio"] < 0.6


def test_module_is_exported_from_interpretation():
    from aeda.interpretation import (
        UPPER_CONTINENTAL_CRUST,
        get_crust_reference,
        compare_to_crust,
    )

    assert UPPER_CONTINENTAL_CRUST is not None
    assert callable(get_crust_reference)
    assert callable(compare_to_crust)