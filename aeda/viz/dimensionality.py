"""
aeda.viz.dimensionality
Visualizations for dimensionality reduction results (PCA, t-SNE, UMAP).

The most important plot in this module is :func:`pca_biplot`, which combines
the sample scatter with variable loadings on a single figure. This is typically
the headline plot of an environmental EDA study because it answers two questions
at once: how do samples group, and which variables drive the grouping.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from aeda.engine.dimensionality import DimReductionResult
from aeda.viz.base import (
    CATEGORICAL_PALETTE,
    apply_default_layout,
    get_categorical_colors,
    resolve_color_column,
)


# ============================================================
#  PCA BIPLOT
# ============================================================

def pca_biplot(
    result: DimReductionResult,
    df: Optional[pd.DataFrame] = None,
    color_by: Optional[str] = None,
    pc_x: int = 1,
    pc_y: int = 2,
    top_n_loadings: int = 15,
    show_labels: bool = True,
    title: Optional[str] = None,
    width: int = 1000,
    height: int = 700,
) -> go.Figure:
    """
    Build a PCA biplot: sample scatter overlaid with loading arrows.

    Samples appear as points in the PC space. Variables appear as arrows from
    the origin — the direction indicates correlation with each PC, and the
    length indicates contribution. Variables pointing the same way are
    correlated; variables pointing opposite ways are anti-correlated.

    Parameters
    ----------
    result : DimReductionResult
        PCA result from ``aeda.engine.dimensionality.run_pca``. Must contain
        ``loadings`` (i.e. method must be ``"PCA"``).
    df : pd.DataFrame, optional
        Original DataFrame, required if ``color_by`` is used for sample coloring
        (e.g. to color by site name).
    color_by : str, optional
        Column name in ``df`` used to color samples categorically.
    pc_x, pc_y : int
        Principal components to plot on the x and y axes (1-indexed).
    top_n_loadings : int
        Number of variables to display as arrows. Variables are ranked by the
        magnitude of their combined contribution to the two selected PCs, so
        arrows never clutter the figure regardless of how many variables exist.
    show_labels : bool
        Whether to label the loading arrows with variable names.
    title : str, optional
        Plot title. A default is generated if omitted.
    width, height : int
        Figure dimensions in pixels.

    Returns
    -------
    plotly.graph_objects.Figure
        Ready-to-display biplot.

    Raises
    ------
    ValueError
        If the result does not come from PCA or if the requested PCs are out of range.
    """
    if result.method != "PCA" or result.loadings is None:
        raise ValueError("pca_biplot requires a PCA result with loadings available.")

    n_components = result.n_components_selected
    if pc_x > n_components or pc_y > n_components:
        raise ValueError(
            f"Requested PC{pc_x} and PC{pc_y}, but only {n_components} components are available."
        )

    x_col = f"PC{pc_x}"
    y_col = f"PC{pc_y}"

    # Variance percentages for the axis labels
    var_x = result.explained_variance[pc_x - 1] * 100
    var_y = result.explained_variance[pc_y - 1] * 100

    fig = go.Figure()

    # ------------------------------------------------------------
    # Sample scatter
    # ------------------------------------------------------------
    scores = result.components

    if df is not None and color_by is not None:
        color_values, color_map = resolve_color_column(df.loc[scores.index], color_by)
        if color_values is not None:
            for group, color in color_map.items():
                mask = color_values == group
                fig.add_trace(go.Scatter(
                    x=scores.loc[mask, x_col],
                    y=scores.loc[mask, y_col],
                    mode="markers",
                    name=str(group),
                    marker=dict(size=8, color=color, line=dict(width=0.5, color="white")),
                    hovertemplate=(
                        f"<b>{color_by}:</b> {group}<br>"
                        f"<b>{x_col}:</b> %{{x:.2f}}<br>"
                        f"<b>{y_col}:</b> %{{y:.2f}}<extra></extra>"
                    ),
                ))
        else:
            _add_plain_scatter(fig, scores, x_col, y_col)
    else:
        _add_plain_scatter(fig, scores, x_col, y_col)

    # ------------------------------------------------------------
    # Loading arrows
    # ------------------------------------------------------------
    loadings = result.loadings[[x_col, y_col]].copy()
    loadings["magnitude"] = np.sqrt(loadings[x_col] ** 2 + loadings[y_col] ** 2)
    top_loadings = loadings.nlargest(top_n_loadings, "magnitude")

    # Scale arrows to sit within the sample cloud for readability
    max_score = max(abs(scores[x_col]).max(), abs(scores[y_col]).max())
    max_loading = max(abs(top_loadings[x_col]).max(), abs(top_loadings[y_col]).max())
    arrow_scale = 0.75 * max_score / max_loading if max_loading > 0 else 1.0

    for var_name, row in top_loadings.iterrows():
        x_end = row[x_col] * arrow_scale
        y_end = row[y_col] * arrow_scale

        fig.add_annotation(
            x=x_end, y=y_end, ax=0, ay=0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=1.5,
            arrowcolor="#555555", opacity=0.8,
        )

        if show_labels:
            # Offset the label slightly past the arrow tip to avoid overlap
            label_offset = 1.08
            fig.add_annotation(
                x=x_end * label_offset, y=y_end * label_offset,
                text=f"<b>{var_name}</b>",
                showarrow=False,
                font=dict(size=11, color="#333333"),
                bgcolor="rgba(255,255,255,0.7)",
                borderpad=2,
            )

    # ------------------------------------------------------------
    # Axes and styling
    # ------------------------------------------------------------
    fig.add_hline(y=0, line=dict(color="#CCCCCC", width=1, dash="dot"))
    fig.add_vline(x=0, line=dict(color="#CCCCCC", width=1, dash="dot"))

    fig.update_xaxes(
        title=f"{x_col} ({var_x:.1f}%)",
        showgrid=True, gridcolor="#F0F0F0", zeroline=False,
    )
    fig.update_yaxes(
        title=f"{y_col} ({var_y:.1f}%)",
        showgrid=True, gridcolor="#F0F0F0", zeroline=False,
        scaleanchor="x", scaleratio=1,  # Equal aspect ratio is correct for PCA
    )

    if title is None:
        title = f"PCA biplot — {x_col} vs {y_col}"
    apply_default_layout(fig, title=title, width=width, height=height)

    return fig


def _add_plain_scatter(fig: go.Figure, scores: pd.DataFrame, x_col: str, y_col: str) -> None:
    """Add a non-colored scatter trace when no ``color_by`` is given."""
    fig.add_trace(go.Scatter(
        x=scores[x_col],
        y=scores[y_col],
        mode="markers",
        name="Samples",
        marker=dict(size=8, color=CATEGORICAL_PALETTE[0], line=dict(width=0.5, color="white")),
        hovertemplate=f"<b>{x_col}:</b> %{{x:.2f}}<br><b>{y_col}:</b> %{{y:.2f}}<extra></extra>",
    ))


# ============================================================
#  SCREE PLOT
# ============================================================

def pca_scree_plot(
    result: DimReductionResult,
    title: Optional[str] = None,
    width: int = 800,
    height: int = 500,
) -> go.Figure:
    """
    Bar chart of variance explained per principal component, with cumulative curve.

    Answers the question: how many components do we actually need? The cumulative
    curve shows when we hit 80% / 90% / 95% of total variance explained.

    Parameters
    ----------
    result : DimReductionResult
        PCA result. Must contain ``explained_variance``.
    title : str, optional
        Plot title.
    width, height : int
        Figure dimensions in pixels.

    Returns
    -------
    plotly.graph_objects.Figure
        Scree plot with dual y-axis (bars for individual variance, line for cumulative).
    """
    if result.explained_variance is None:
        raise ValueError("pca_scree_plot requires a result with explained variance.")

    n = len(result.explained_variance)
    components = [f"PC{i+1}" for i in range(n)]
    variance_pct = result.explained_variance * 100
    cumulative_pct = np.cumsum(variance_pct)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=components, y=variance_pct,
        name="Variance explained",
        marker_color=CATEGORICAL_PALETTE[0],
        hovertemplate="<b>%{x}</b><br>Variance: %{y:.2f}%<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=components, y=cumulative_pct,
        mode="lines+markers",
        name="Cumulative",
        yaxis="y2",
        line=dict(color=CATEGORICAL_PALETTE[1], width=2),
        marker=dict(size=7),
        hovertemplate="<b>%{x}</b><br>Cumulative: %{y:.2f}%<extra></extra>",
    ))

    # Threshold reference lines on the cumulative axis
    for threshold in (80, 90, 95):
        fig.add_shape(
            type="line",
            x0=0, x1=1, xref="paper",
            y0=threshold, y1=threshold, yref="y2",
            line=dict(color="#AAAAAA", width=1, dash="dash"),
        )
        fig.add_annotation(
            text=f"{threshold}%",
            xref="paper", x=1.02,
            yref="y2", y=threshold,
            showarrow=False,
            font=dict(size=10, color="#888888"),
            xanchor="left",
        )

    fig.update_xaxes(title="Principal component")
    fig.update_yaxes(title="Variance explained (%)", gridcolor="#F0F0F0")
    fig.update_layout(
        yaxis2=dict(
            title="Cumulative (%)",
            overlaying="y", side="right",
            range=[0, 105], showgrid=False,
        ),
    )

    if title is None:
        title = "PCA scree plot"
    apply_default_layout(fig, title=title, width=width, height=height)

    return fig


# ============================================================
#  2D SCATTER FOR t-SNE / UMAP
# ============================================================

def embedding_scatter(
    result: DimReductionResult,
    df: Optional[pd.DataFrame] = None,
    color_by: Optional[str] = None,
    title: Optional[str] = None,
    width: int = 800,
    height: int = 700,
) -> go.Figure:
    """
    Generic 2D scatter of any dimensionality-reduction embedding (t-SNE, UMAP, PCA).

    Simpler than :func:`pca_biplot` because no loadings are overlaid. Use it for
    non-linear embeddings (t-SNE, UMAP) where loadings do not exist, or when a
    clean scatter without arrows is preferred.

    Parameters
    ----------
    result : DimReductionResult
        Any dimensionality-reduction result with at least 2 components.
    df : pd.DataFrame, optional
        Original DataFrame for coloring.
    color_by : str, optional
        Column in ``df`` used to color samples.
    title : str, optional
        Plot title.
    width, height : int
        Figure dimensions in pixels.

    Returns
    -------
    plotly.graph_objects.Figure
        2D scatter of the embedding.
    """
    scores = result.components
    x_col, y_col = scores.columns[0], scores.columns[1]

    fig = go.Figure()

    if df is not None and color_by is not None:
        color_values, color_map = resolve_color_column(df.loc[scores.index], color_by)
        if color_values is not None:
            for group, color in color_map.items():
                mask = color_values == group
                fig.add_trace(go.Scatter(
                    x=scores.loc[mask, x_col], y=scores.loc[mask, y_col],
                    mode="markers", name=str(group),
                    marker=dict(size=9, color=color, line=dict(width=0.5, color="white")),
                ))
        else:
            _add_plain_scatter(fig, scores, x_col, y_col)
    else:
        _add_plain_scatter(fig, scores, x_col, y_col)

    fig.update_xaxes(title=x_col, showgrid=True, gridcolor="#F0F0F0")
    fig.update_yaxes(title=y_col, showgrid=True, gridcolor="#F0F0F0")

    if title is None:
        title = f"{result.method} embedding"
    apply_default_layout(fig, title=title, width=width, height=height)

    return fig
