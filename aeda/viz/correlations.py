"""
aeda.viz.correlations
Correlation heatmaps with hierarchical clustering of variables.

A plain correlation matrix becomes much more informative when its rows and columns
are reordered so that correlated variables sit next to each other. The reordering is
computed via hierarchical clustering on ``1 − |corr|`` as distance.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

from aeda.engine.correlations import CorrelationResult
from aeda.viz.base import DIVERGING_PALETTE, apply_default_layout


# ============================================================
#  CORRELATION HEATMAP
# ============================================================

def correlation_heatmap(
    result: CorrelationResult | pd.DataFrame,
    reorder: bool = True,
    show_values: bool = False,
    value_threshold: float = 0.3,
    title: Optional[str] = None,
    width: int = 900,
    height: int = 800,
) -> go.Figure:
    """
    Heatmap of a correlation matrix, optionally reordered via hierarchical clustering.

    When ``reorder=True``, rows and columns are permuted so that correlated variables
    cluster together visually, making blocks of related variables immediately obvious.

    Parameters
    ----------
    result : CorrelationResult or pd.DataFrame
        Correlation output from ``aeda.engine.correlations.correlate``, or a raw
        correlation DataFrame.
    reorder : bool
        If True, reorder rows/columns using Ward hierarchical clustering.
    show_values : bool
        If True, overlay numerical correlation values on each cell (above the threshold).
    value_threshold : float
        Only show numerical values whose absolute correlation exceeds this threshold
        (avoids clutter on large matrices).
    title : str, optional
        Plot title.
    width, height : int
        Figure dimensions in pixels.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive heatmap.
    """
    matrix = result.matrix if isinstance(result, CorrelationResult) else result
    method_label = result.method if isinstance(result, CorrelationResult) else "Correlation"

    if reorder and len(matrix) > 2:
        matrix = _reorder_by_clustering(matrix)

    # Build the annotation matrix if values are shown
    text_values = None
    if show_values:
        text_values = matrix.round(2).astype(str).values
        # Blank cells below the threshold
        mask = matrix.abs() < value_threshold
        text_values = np.where(mask.values, "", text_values)

    fig = go.Figure(data=go.Heatmap(
        z=matrix.values,
        x=matrix.columns,
        y=matrix.index,
        colorscale=DIVERGING_PALETTE,
        zmin=-1, zmax=1,
        colorbar=dict(title="r", thickness=15, len=0.7),
        text=text_values,
        texttemplate="%{text}" if show_values else None,
        textfont=dict(size=9, color="#333333"),
        hovertemplate="<b>%{y}</b> × <b>%{x}</b><br>r = %{z:.3f}<extra></extra>",
    ))

    fig.update_xaxes(side="bottom", tickangle=-45, showgrid=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)

    if title is None:
        suffix = " (clustered)" if reorder else ""
        title = f"{method_label} correlation matrix{suffix}"

    apply_default_layout(fig, title=title, width=width, height=height)

    return fig


def _reorder_by_clustering(corr_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Reorder a correlation matrix using hierarchical clustering on ``1 - |corr|``.

    Uses Ward linkage to minimize within-cluster variance, producing visually tight
    blocks of related variables.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Square, symmetric correlation matrix.

    Returns
    -------
    pd.DataFrame
        Same matrix with rows and columns permuted.
    """
    # Use 1 - |corr| as distance: perfectly correlated variables (|r|=1) have distance 0
    distances_values = (1 - corr_matrix.abs()).values.copy()
    # squareform requires zero diagonal and full symmetry
    np.fill_diagonal(distances_values, 0)
    condensed = squareform(distances_values, checks=False)

    linkage_matrix = linkage(condensed, method="average")
    order = leaves_list(linkage_matrix)

    reordered_labels = corr_matrix.index[order]
    return corr_matrix.loc[reordered_labels, reordered_labels]


# ============================================================
#  CROSS-CORRELATION BETWEEN TWO VARIABLE GROUPS
# ============================================================

def cross_correlation_heatmap(
    df: pd.DataFrame,
    group_a: list[str],
    group_b: list[str],
    method: str = "spearman",
    title: Optional[str] = None,
    width: int = 800,
    height: int = 600,
) -> go.Figure:
    """
    Non-square heatmap showing correlations between two variable groups.

    Useful for targeted questions like "do heavy metals correlate with grain size?"
    where a full correlation matrix would be wasteful and harder to read.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the variables.
    group_a : list of str
        Variable names for the y-axis (rows).
    group_b : list of str
        Variable names for the x-axis (columns).
    method : str
        Correlation method: ``"pearson"``, ``"spearman"``, or ``"kendall"``.
    title : str, optional
        Plot title.
    width, height : int
        Figure dimensions in pixels.

    Returns
    -------
    plotly.graph_objects.Figure
        Rectangular correlation heatmap.
    """
    # Compute the full correlation, then slice to the two groups
    full_corr = df[group_a + group_b].corr(method=method)
    sub = full_corr.loc[group_a, group_b]

    fig = go.Figure(data=go.Heatmap(
        z=sub.values,
        x=sub.columns, y=sub.index,
        colorscale=DIVERGING_PALETTE,
        zmin=-1, zmax=1,
        colorbar=dict(title="r", thickness=15, len=0.7),
        text=sub.round(2).astype(str).values,
        texttemplate="%{text}",
        textfont=dict(size=10),
        hovertemplate="<b>%{y}</b> × <b>%{x}</b><br>r = %{z:.3f}<extra></extra>",
    ))

    fig.update_xaxes(side="bottom", tickangle=-45, showgrid=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)

    if title is None:
        title = f"{method.capitalize()} correlation — group comparison"

    apply_default_layout(fig, title=title, width=width, height=height)

    return fig
