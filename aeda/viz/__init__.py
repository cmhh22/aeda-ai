"""
aeda.viz
Visualization module for AEDA-AI results.

Each submodule focuses on a specific type of plot:

* ``dimensionality`` — PCA biplot, scree plot, t-SNE/UMAP scatter
* ``correlations``   — correlation heatmaps (square and cross-group)
* ``clustering``     — cluster scatters, cluster composition
* ``profiles``       — depth profile plots for sediment cores

All plot functions return a :class:`plotly.graph_objects.Figure`. Use
``fig.show()`` in Jupyter, or :func:`aeda.viz.base.save_figure` to export
to HTML, PNG, SVG, or PDF.

Example
-------
>>> from aeda.pipeline.runner import AEDAPipeline
>>> from aeda.viz import pca_biplot, correlation_heatmap
>>> pipeline = AEDAPipeline()
>>> results = pipeline.run("data.xlsx", exclude_cols=[...])
>>> fig = pca_biplot(results.dim_reduction, df=results.raw_data, color_by="Sitio_muestreo")
>>> fig.show()
"""

from aeda.viz.base import (
    CATEGORICAL_PALETTE,
    DIVERGING_PALETTE,
    apply_default_layout,
    get_categorical_colors,
    save_figure,
)
from aeda.viz.clustering import cluster_composition, cluster_scatter
from aeda.viz.correlations import correlation_heatmap, cross_correlation_heatmap
from aeda.viz.dimensionality import embedding_scatter, pca_biplot, pca_scree_plot
from aeda.viz.interpretation import (
    contamination_classification_plot,
    enrichment_factor_depth_plot,
)
from aeda.viz.profiles import (
    depth_profile,
    depth_profile_grid,
    depth_profile_with_thresholds,
)

__all__ = [
    # Base utilities
    "CATEGORICAL_PALETTE",
    "DIVERGING_PALETTE",
    "apply_default_layout",
    "get_categorical_colors",
    "save_figure",
    # Dimensionality
    "pca_biplot",
    "pca_scree_plot",
    "embedding_scatter",
    # Correlations
    "correlation_heatmap",
    "cross_correlation_heatmap",
    # Clustering
    "cluster_scatter",
    "cluster_composition",
    # Profiles
    "depth_profile",
    "depth_profile_grid",
    "depth_profile_with_thresholds",
    # Interpretation
    "enrichment_factor_depth_plot",
    "contamination_classification_plot",
]
