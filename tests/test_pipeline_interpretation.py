"""Tests for pipeline interpretation integration."""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from aeda.pipeline.runner import AEDAPipeline


def test_pipeline_runs_interpretation_on_isovida():
    """Regression: pipeline must produce an interpretation report when prerequisites are met."""
    data_path = Path("data/BD_ISOVIDA_MANGLARES2023_rectificadaYBA_230326.xlsx")
    if not data_path.exists():
        pytest.skip(f"ISOVIDA data file not found at {data_path}")

    EXCLUDE = ["No", "Code", "Site_Name", "Pret_Code", "Código_muestra",
               "Sitio_muestreo", "Fecha_muestreo", "Core",
               "Latitud", "Longitud", "Profundidad"]

    p = AEDAPipeline(impute_strategy="median")
    r = p.run(str(data_path),
              exclude_cols=EXCLUDE, sheet_name="DATA")

    assert r.interpretation is not None, "Interpretation must run on ISOVIDA"
    assert r.interpretation.ef_result is not None
    assert r.interpretation.ef_result.reference_element == "Al"
    assert len(r.interpretation.metals_analyzed) > 0
    # TEL/PEL should be computed for at least Pb
    assert "Pb" in r.interpretation.tel_pel_classifications.columns


def test_pipeline_skips_interpretation_without_reference_element():
    """If the reference element is not in the dataset, interpretation must be skipped, not crash."""
    data_path = Path("data/BD_ISOVIDA_MANGLARES2023_rectificadaYBA_230326.xlsx")
    if not data_path.exists():
        pytest.skip(f"ISOVIDA data file not found at {data_path}")

    EXCLUDE = ["No", "Code", "Site_Name", "Pret_Code", "Código_muestra",
               "Sitio_muestreo", "Fecha_muestreo", "Core",
               "Latitud", "Longitud", "Profundidad"]

    # Use a non-existent reference element
    p = AEDAPipeline(impute_strategy="median", reference_element="Unobtanium")
    r = p.run(str(data_path),
              exclude_cols=EXCLUDE, sheet_name="DATA")

    # Pipeline must still run successfully but with no interpretation
    assert r.dim_reduction is not None  # other steps still work
    assert r.interpretation is None


def test_pipeline_with_custom_reference_element():
    """The user can override the reference element via the constructor."""
    data_path = Path("data/BD_ISOVIDA_MANGLARES2023_rectificadaYBA_230326.xlsx")
    if not data_path.exists():
        pytest.skip(f"ISOVIDA data file not found at {data_path}")

    EXCLUDE = ["No", "Code", "Site_Name", "Pret_Code", "Código_muestra",
               "Sitio_muestreo", "Fecha_muestreo", "Core",
               "Latitud", "Longitud", "Profundidad"]

    p = AEDAPipeline(impute_strategy="median", reference_element="Fe")
    r = p.run(str(data_path),
              exclude_cols=EXCLUDE, sheet_name="DATA")

    assert r.interpretation is not None
    assert r.interpretation.ef_result.reference_element == "Fe"
    # Fe must not appear among the metals analyzed (it is the reference)
    assert "Fe" not in r.interpretation.metals_analyzed


def test_run_interpretation_flag_disables_step():
    """run_interpretation=False must skip the step entirely."""
    data_path = Path("data/BD_ISOVIDA_MANGLARES2023_rectificadaYBA_230326.xlsx")
    if not data_path.exists():
        pytest.skip(f"ISOVIDA data file not found at {data_path}")

    EXCLUDE = ["No", "Code", "Site_Name", "Pret_Code", "Código_muestra",
               "Sitio_muestreo", "Fecha_muestreo", "Core",
               "Latitud", "Longitud", "Profundidad"]

    p = AEDAPipeline(impute_strategy="median", run_interpretation=False)
    r = p.run(str(data_path),
              exclude_cols=EXCLUDE, sheet_name="DATA")

    assert r.interpretation is None
