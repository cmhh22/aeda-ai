"""
aeda.viz.clustering
Visualizations for clustering results.

The headline plot is :func:`cluster_scatter`, which projects samples into 2D
(typically via PCA or UMAP) and colors them by cluster assignment. An optional
side-by-side comparison with a ground-truth grouping (e.g. sampling site) makes
it easy to see whether the chemistry-driven clusters align with geography.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from aeda.engine.clustering import ClusteringResult
from aeda.engine.dimensionality import DimReductionResult
from aeda.viz.base import (
    CATEGORICAL_PALETTE,
    apply_default_layout,
    get_categorical_colors,
    resolve_color_column,
)


# ============================================================
#  CLUSTER SCATTER
# ============================================================

def cluster_scatter(
    clustering: ClusteringResult,
    dim_reduction: DimReductionResult,
    df: Optional[pd.DataFrame] = None,
    compare_with: Optional[str] = None,
    title: Optional[str] = None,
    width: int = 1000,
    height: int = 600,
) -> go.Figure:
    """
    Scatter of samples in 2D coordinates, colored by cluster assignment.

    If ``compare_with`` is given, the plot becomes two side-by-side scatters
    sharing the same coordinates: one colored by cluster, the other by the
    ground-truth column. The visual comparison answers directly whether the
    chemistry-driven clustering recovers the expected geographic grouping.

    Parameters
    ----------
    clustering : ClusteringResult
        Clustering output from ``aeda.engine.clustering``.
    dim_reduction : DimReductionResult
        2D embedding used for plotting (PCA, UMAP, t-SNE). The first two
        components are used as x, y.
    df : pd.DataFrame, optional
        Original DataFrame, needed when ``compare_with`` is used.
    compare_with : str, optional
        Column name in ``df`` (e.g. ``"Sitio_muestreo"``) to plot side-by-side
        against the clustering result.
    title : str, optional
        Plot title.
    width, height : int
        Figure dimensions in pixels.

    Returns
    -------
    plotly.graph_objects.Figure
        Single or dual scatter plot.
    """
    scores = dim_reduction.components
    x_col, y_col = scores.columns[0], scores.columns[1]

    labels = clustering.label_series(index=scores.index)

    # Determine axis labels with variance when PCA
    if dim_reduction.method == "PCA" and dim_reduction.explained_variance is not None:
        x_label = f"{x_col} ({dim_reduction.explained_variance[0]*100:.1f}%)"
        y_label = f"{y_col} ({dim_reduction.explained_variance[1]*100:.1f}%)"
    else:
        x_label, y_label = x_col, y_col

    if compare_with is None or df is None or compare_with not in df.columns:
        fig = go.Figure()
        _add_cluster_traces(fig, scores, x_col, y_col, labels)
        fig.update_xaxes(title=x_label, gridcolor="#F0F0F0")
        fig.update_yaxes(title=y_label, gridcolor="#F0F0F0")
        if title is None:
            title = f"Clustering: {clustering.method} ({clustering.n_clusters} clusters)"
        apply_default_layout(fig, title=title, width=width, height=height)
        return fig

    # Side-by-side comparison
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"{clustering.method} ({clustering.n_clusters} clusters)",
            f"Ground truth: {compare_with}",
        ),
        horizontal_spacing=0.12,
    )

    _add_cluster_traces(fig, scores, x_col, y_col, labels, row=1, col=1)

    color_values = df.loc[scores.index, compare_with]
    color_map = get_categorical_colors(color_values.unique())
    for group, color in color_map.items():
        mask = color_values == group
        fig.add_trace(
            go.Scatter(
                x=scores.loc[mask, x_col], y=scores.loc[mask, y_col],
                mode="markers", name=str(group),
                legendgroup=str(group), legendgrouptitle_text=compare_with,
                marker=dict(size=7, color=color, line=dict(width=0.5, color="white")),
                hovertemplate=f"<b>{group}</b><br>{x_col}: %{{x:.2f}}<br>{y_col}: %{{y:.2f}}<extra></extra>",
            ),
            row=1, col=2,
        )

    fig.update_xaxes(title=x_label, gridcolor="#F0F0F0", row=1, col=1)
    fig.update_xaxes(title=x_label, gridcolor="#F0F0F0", row=1, col=2)
    fig.update_yaxes(title=y_label, gridcolor="#F0F0F0", row=1, col=1)
    fig.update_yaxes(title=y_label, gridcolor="#F0F0F0", row=1, col=2)

    if title is None:
        title = "Clustering vs. ground truth"
    apply_default_layout(fig, title=title, width=width, height=height)

    return fig


def _add_cluster_traces(
    fig: go.Figure,
    scores: pd.DataFrame,
    x_col: str,
    y_col: str,
    labels: pd.Series,
    row: Optional[int] = None,
    col: Optional[int] = None,
) -> None:
    """Add one scatter trace per cluster, with a special style for DBSCAN noise."""
    unique_labels = sorted(labels.unique())
    # Put noise last so it renders under the clusters
    if -1 in unique_labels:
        unique_labels.remove(-1)
        unique_labels.append(-1)

    for i, label in enumerate(unique_labels):
        mask = labels == label
        if label == -1:
            name = "Noise"
            color = "#AAAAAA"
            marker_opts = dict(size=6, color=color, symbol="x", line=dict(width=0.5))
        else:
            name = f"Cluster {label}"
            color = CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)]
            marker_opts = dict(size=8, color=color, line=dict(width=0.5, color="white"))

        trace = go.Scatter(
            x=scores.loc[mask, x_col], y=scores.loc[mask, y_col],
            mode="markers", name=name,
            legendgroup=name, legendgrouptitle_text="Clusters",
            marker=marker_opts,
            hovertemplate=f"<b>{name}</b><br>{x_col}: %{{x:.2f}}<br>{y_col}: %{{y:.2f}}<extra></extra>",
        )

        if row is not None and col is not None:
            fig.add_trace(trace, row=row, col=col)
        else:
            fig.add_trace(trace)


# ============================================================
#  CLUSTER COMPOSITION
# ============================================================

def cluster_composition(
    clustering: ClusteringResult,
    df: pd.DataFrame,
    category_col: str,
    title: Optional[str] = None,
    width: int = 900,
    height: int = 500,
) -> go.Figure:
    """
    Stacked bar chart showing which categories fall into each cluster.

    Complements :func:`cluster_scatter` with a quantitative view: if cluster 0
    contains 80% of samples from one site, the bar for that cluster will be
    mostly one color. Useful for validating that clusters correspond to meaningful
    groups in the data.

    Parameters
    ----------
    clustering : ClusteringResult
        Clustering output.
    df : pd.DataFrame
        DataFrame containing the category column.
    category_col : str
        Column to compare against (e.g. ``"Sitio_muestreo"``).
    title : str, optional
        Plot title.
    width, height : int
        Figure dimensions in pixels.

    Returns
    -------
    plotly.graph_objects.Figure
        Stacked bar chart.
    """
    labels = clustering.label_series(index=df.index[:len(clustering.labels)])
    categories = df.loc[labels.index, category_col]

    # Cross-tabulation: rows = clusters, columns = categories
    crosstab = pd.crosstab(labels, categories, normalize="index") * 100

    color_map = get_categorical_colors(crosstab.columns)

    fig = go.Figure()
    for category in crosstab.columns:
        fig.add_trace(go.Bar(
            x=[f"Cluster {c}" if c != -1 else "Noise" for c in crosstab.index],
            y=crosstab[category],
            name=str(category),
            marker_color=color_map[category],
            hovertemplate=f"<b>{category}</b><br>%{{y:.1f}}%<extra></extra>",
        ))

    fig.update_layout(barmode="stack")
    fig.update_xaxes(title="")
    fig.update_yaxes(title=f"% of samples in cluster", range=[0, 100], gridcolor="#F0F0F0")

    if title is None:
        title = f"Cluster composition by {category_col}"
    apply_default_layout(fig, title=title, width=width, height=height)

    return fig
