"""Regression test: structural columns (depth/site/coordinates) must never be
analyzed as ML variables, even when the user forgets to exclude them."""
from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from aeda.pipeline.runner import AEDAPipeline

warnings.filterwarnings("ignore")

DATA = Path("data/BD_ISOVIDA_MANGLARES2023_rectificadaYBA_230326.xlsx")

# Deliberately omit "Profundidad" to simulate a user who forgot to exclude it.
EXCLUDE_WITHOUT_DEPTH = [
    "No", "Code", "Site_Name", "Pret_Code", "CÃ³digo_muestra",
    "Sitio_muestreo", "Fecha_muestreo", "Core", "Latitud", "Longitud",
]


@pytest.fixture(scope="module")
def results():
    if not DATA.exists():
        pytest.skip("ISOVIDA dataset not available")
    return AEDAPipeline().run(str(DATA), exclude_cols=EXCLUDE_WITHOUT_DEPTH, sheet_name="DATA")


def test_depth_is_detected(results):
    assert results.dataset_info.depth_col == "Profundidad"


def test_depth_not_in_analyzed_features(results):
    """Profundidad must be absent from the PCA feature set despite not being
    listed in exclude_cols by the user."""
    features = list(results.dim_reduction.feature_names)
    assert "Profundidad" not in features
    # Sanity: the analyzed set is the 36 measurement variables, not 37.
    assert len(features) == 36
