"""Regression tests for heavy metals alignment between auto_selector and interpretation module."""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np


def test_heavy_metals_aligned_with_noaa_table():
    """Regression: HEAVY_METALS in auto_selector must equal the NOAA thresholds table."""
    from aeda.engine.auto_selector import HEAVY_METALS
    from aeda.interpretation.thresholds import TEL_PEL_MARINE_SEDIMENT

    assert HEAVY_METALS == set(TEL_PEL_MARINE_SEDIMENT.keys()), (
        f"HEAVY_METALS={sorted(HEAVY_METALS)} does not match "
        f"NOAA={sorted(TEL_PEL_MARINE_SEDIMENT.keys())}"
    )


def test_mn_only_in_major_elements_not_heavy():
    """Mn is a major (lithogenic) element, not a regulated heavy metal in NOAA."""
    from aeda.engine.auto_selector import MAJOR_ELEMENTS, HEAVY_METALS
    assert "Mn" in MAJOR_ELEMENTS
    assert "Mn" not in HEAVY_METALS


def test_cd_hg_ag_sb_now_recognized_as_heavy_metals():
    """These metals have NOAA thresholds and must be detected by the brain."""
    from aeda.engine.auto_selector import HEAVY_METALS
    for metal in ("Cd", "Hg", "Ag", "Sb"):
        assert metal in HEAVY_METALS, f"{metal} should be in HEAVY_METALS (has NOAA TEL/PEL)"


def test_isovida_heavy_metal_detection_includes_relevant_metals():
    """Validation against the real ISOVIDA dataset."""
    from pathlib import Path

    # Check if ISOVIDA data file exists
    data_path = Path("data/BD_ISOVIDA_MANGLARES2023_rectificadaYBA_230326.xlsx")
    if not data_path.exists():
        pytest.skip(f"ISOVIDA data file not found at {data_path}")

    from aeda.pipeline.runner import AEDAPipeline

    EXCLUDE = ["No", "Code", "Site_Name", "Pret_Code", "Código_muestra",
               "Sitio_muestreo", "Fecha_muestreo", "Core",
               "Latitud", "Longitud", "Profundidad"]
    p = AEDAPipeline(impute_strategy="median")
    r = p.run(str(data_path),
              exclude_cols=EXCLUDE, sheet_name="DATA")

    detected = set(r.plan.profile.heavy_metal_cols)
    # ISOVIDA's FRX measurements include at least these heavy metals
    expected_detected = {"As", "Cr", "Cu", "Ni", "Pb", "Zn"}
    assert expected_detected.issubset(detected), (
        f"Expected at least {expected_detected} to be detected as heavy metals, got {detected}"
    )

    # Mn must NOT appear here (it's in MAJOR)
    assert "Mn" not in detected
