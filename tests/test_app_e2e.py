"""End-to-end smoke tests for the Streamlit interface (third validation level).

These tests exercise the web interface with Streamlit's native ``AppTest``
runner, which executes the app in-process without a browser. They verify two
critical states of every page:

* the empty state (no dataset loaded), and
* the loaded state (a real ISOVIDA analysis seeded into ``session_state``).

The file-upload step is bypassed by seeding ``session_state`` directly, which
is the standard approach for end-to-end testing of Streamlit apps whose state
flows through ``session_state``. The goal is to guard the interface layer
against regressions: any page that raises on render is caught here.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pytest

# Streamlit is an optional ("ui") dependency; skip cleanly where it is absent.
pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from aeda.pipeline.runner import AEDAPipeline  # noqa: E402

warnings.filterwarnings("ignore")

APP = "app/main.py"
DATA = Path("data/BD_ISOVIDA_MANGLARES2023_rectificadaYBA_230326.xlsx")
EXCLUDE = ["No", "Code", "Site_Name", "Pret_Code", "CÃ³digo_muestra",
           "Sitio_muestreo", "Fecha_muestreo", "Core",
           "Latitud", "Longitud", "Profundidad"]
PAGES = ["Upload & Configure", "Analysis Plan", "Results",
         "Depth Profiles", "Audit", "Advanced Configuration"]


@pytest.fixture(scope="module")
def results():
    if not DATA.exists():
        pytest.skip("ISOVIDA dataset not available")
    return AEDAPipeline().run(str(DATA), exclude_cols=EXCLUDE, sheet_name="DATA")


def _seeded(results):
    """Build an AppTest with a completed analysis seeded into session_state."""
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["results"] = results
    at.session_state["raw_df"] = results.raw_data
    at.session_state["filename"] = "ISOVIDA.xlsx"
    at.session_state["run_context"] = {
        "tmp_path": str(DATA),
        "sheet_name": "DATA",
        "exclude_cols": EXCLUDE,
        "settings": results.effective_settings,
    }
    return at


def test_app_boots_empty():
    """The app boots in the empty state without raising."""
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    # The navigation radio is present in the sidebar.
    assert len(at.sidebar.radio) == 1
    # Six navigation entries are present. Labels may be displayed translated
    # (format_func), so we assert the count rather than the raw English values.
    assert len(at.sidebar.radio[0].options) == len(PAGES)


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_with_results(results, page):
    """Every page renders without raising when a full analysis is loaded."""
    at = _seeded(results).run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, f"page {page!r} raised: {[e.value for e in at.exception]}"
