"""Smoke tests for the visualization layer (aeda.viz).

These tests verify that every public figure-builder runs end-to-end on the
real ISOVIDA results and returns a Plotly Figure without raising. They are
deliberately lightweight (no pixel assertions): their purpose is to guard the
viz layer against regressions and to exercise its code paths.
"""
from __future__ import annotations
import warnings
from pathlib import Path
import pytest
import plotly.graph_objects as go

from aeda.pipeline.runner import AEDAPipeline

warnings.filterwarnings("ignore")

EXCLUDE = ["No", "Code", "Site_Name", "Pret_Code", "Código_muestra",
           "Sitio_muestreo", "Fecha_muestreo", "Core",
           "Latitud", "Longitud", "Profundidad"]
DATA = Path("data/BD_ISOVIDA_MANGLARES2023_rectificadaYBA_230326.xlsx")


@pytest.fixture(scope="module")
def results():
    if not DATA.exists():
        pytest.skip("ISOVIDA dataset not available")
    return AEDAPipeline().run(str(DATA), exclude_cols=EXCLUDE, sheet_name="DATA")


def _is_fig(obj):
    return isinstance(obj, go.Figure)


def test_pca_biplot(results):
    from aeda.viz.dimensionality import pca_biplot
    fig = pca_biplot(results.dim_reduction, df=results.raw_data,
                     color_by="Sitio_muestreo")
    assert _is_fig(fig)


def test_pca_scree(results):
    from aeda.viz.dimensionality import pca_scree_plot
    assert _is_fig(pca_scree_plot(results.dim_reduction))


def test_embedding_scatter(results):
    from aeda.viz.dimensionality import embedding_scatter
    fig = embedding_scatter(results.dim_reduction, df=results.raw_data,
                            color_by="Sitio_muestreo")
    assert _is_fig(fig)


def test_cluster_scatter(results):
    from aeda.viz.clustering import cluster_scatter
    fig = cluster_scatter(results.clustering, results.dim_reduction,
                          df=results.raw_data, compare_with="Sitio_muestreo")
    assert _is_fig(fig)


def test_cluster_composition(results):
    from aeda.viz.clustering import cluster_composition
    fig = cluster_composition(results.clustering, results.raw_data,
                              category_col="Sitio_muestreo")
    assert _is_fig(fig)


def test_correlation_heatmap(results):
    from aeda.viz.correlations import correlation_heatmap
    assert _is_fig(correlation_heatmap(results.correlations["pearson"]))
    assert _is_fig(correlation_heatmap(results.correlations["pearson"],
                                       reorder=False, show_values=True))


def test_cross_correlation_heatmap(results):
    from aeda.viz.correlations import cross_correlation_heatmap
    metals = ["Pb", "Ni", "Cr", "Cu", "Zn", "As"]
    grain = ["< 2 µm", "2 < G < 63 µm", "> 63 µm"]
    fig = cross_correlation_heatmap(results.raw_data, metals, grain)
    assert _is_fig(fig)


def test_ef_depth_plot(results):
    from aeda.viz.interpretation import enrichment_factor_depth_plot
    fig = enrichment_factor_depth_plot(
        results.interpretation.ef_result, results.raw_data,
        depth_col="Profundidad", site_col="Sitio_muestreo")
    assert _is_fig(fig)


def test_contamination_classification_plots(results):
    from aeda.viz.interpretation import contamination_classification_plot
    assert _is_fig(contamination_classification_plot(
        results.interpretation.tel_pel_classifications, kind="tel_pel"))
    assert _is_fig(contamination_classification_plot(
        results.interpretation.ef_classifications, kind="ef"))


def test_depth_profile(results):
    from aeda.viz.profiles import depth_profile
    assert _is_fig(depth_profile(results.raw_data, "Pb",
                                 depth_col="Profundidad",
                                 site_col="Sitio_muestreo"))


def test_depth_profile_grid(results):
    from aeda.viz.profiles import depth_profile_grid
    fig = depth_profile_grid(results.raw_data, ["Pb", "Ni", "Cr"],
                             depth_col="Profundidad", site_col="Sitio_muestreo")
    assert _is_fig(fig)


def test_depth_profile_with_thresholds(results):
    from aeda.viz.profiles import depth_profile_with_thresholds
    fig = depth_profile_with_thresholds(results.raw_data, "Pb",
                                        depth_col="Profundidad",
                                        site_col="Sitio_muestreo")
    assert _is_fig(fig)
