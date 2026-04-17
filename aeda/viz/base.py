"""
aeda.viz.base
Shared theming, color palettes, and utility functions for all visualization modules.

All plots in AEDA-AI use a consistent visual style to produce cohesive, publication-ready
figures. This module centralizes the style so individual plot functions stay focused on
the data logic.
"""

from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd
import plotly.graph_objects as go


# ============================================================
#  DEFAULT LAYOUT
# ============================================================

#: Standard layout applied to every figure. Values can be overridden per-plot.
DEFAULT_LAYOUT = dict(
    template="simple_white",
    font=dict(family="Arial, sans-serif", size=13, color="#2E4057"),
    title=dict(x=0.5, xanchor="center", font=dict(size=16)),
    margin=dict(l=60, r=40, t=70, b=60),
    plot_bgcolor="white",
    paper_bgcolor="white",
    hoverlabel=dict(
        bgcolor="white",
        bordercolor="#CCCCCC",
        font=dict(family="Arial, sans-serif", size=12),
    ),
    legend=dict(
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#CCCCCC",
        borderwidth=1,
    ),
)


# ============================================================
#  COLOR PALETTES
# ============================================================

#: Categorical palette for discrete groupings (sites, clusters, categories).
#: Colorblind-friendly, high contrast on white background.
CATEGORICAL_PALETTE = [
    "#2E4057",  # Dark blue
    "#D85A30",  # Coral
    "#1D9E75",  # Teal
    "#BA7517",  # Amber
    "#7F77DD",  # Purple
    "#D4537E",  # Pink
    "#639922",  # Green
    "#888780",  # Gray
    "#185FA5",  # Blue
    "#A32D2D",  # Red
]

#: Diverging palette for correlations and anomaly scores (−1 to +1).
DIVERGING_PALETTE = [
    [0.0, "#185FA5"],   # Strong negative
    [0.25, "#85B7EB"],  # Mild negative
    [0.5, "#F5F5F5"],   # Neutral
    [0.75, "#F09595"],  # Mild positive
    [1.0, "#A32D2D"],   # Strong positive
]

#: Sequential palette for continuous variables (e.g. depth, concentration).
SEQUENTIAL_PALETTE = "Viridis"


def get_categorical_colors(categories: Sequence[str]) -> dict[str, str]:
    """
    Assign a stable color to each category.

    The same category always gets the same color within a dataset, so plots
    across the pipeline stay visually consistent (e.g. ``Delfinario`` is always
    the same color in every figure).

    Parameters
    ----------
    categories : sequence of str
        Unique category labels (e.g. site names, cluster IDs).

    Returns
    -------
    dict[str, str]
        Mapping from category label to hex color.
    """
    unique_cats = list(dict.fromkeys(categories))  # Preserve insertion order
    colors = {}
    for i, cat in enumerate(unique_cats):
        colors[cat] = CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)]
    return colors


# ============================================================
#  FIGURE UTILITIES
# ============================================================

def apply_default_layout(
    fig: go.Figure,
    title: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    **overrides,
) -> go.Figure:
    """
    Apply the AEDA-AI default layout to a Plotly figure.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        The figure to style.
    title : str, optional
        Plot title.
    width, height : int, optional
        Figure dimensions in pixels. If omitted, Plotly auto-sizes.
    **overrides
        Additional layout parameters to merge on top of the defaults.

    Returns
    -------
    plotly.graph_objects.Figure
        The same figure, styled in place and returned for chaining.
    """
    layout = {**DEFAULT_LAYOUT}
    if title is not None:
        layout["title"] = {**layout["title"], "text": title}
    if width is not None:
        layout["width"] = width
    if height is not None:
        layout["height"] = height
    layout.update(overrides)
    fig.update_layout(**layout)
    return fig


def save_figure(
    fig: go.Figure,
    path: str,
    format: Optional[str] = None,
    width: int = 1000,
    height: int = 700,
    scale: int = 2,
) -> None:
    """
    Save a figure to disk as HTML, PNG, SVG, or PDF.

    The format is inferred from the file extension if not given explicitly.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Figure to save.
    path : str
        Destination path.
    format : str, optional
        One of ``html``, ``png``, ``svg``, ``pdf``. Inferred from ``path`` if omitted.
    width, height : int
        Output dimensions in pixels (images only).
    scale : int
        Resolution multiplier for raster formats (higher = sharper).
    """
    if format is None:
        format = path.rsplit(".", 1)[-1].lower()

    if format == "html":
        fig.write_html(path, include_plotlyjs="cdn")
    elif format in ("png", "svg", "pdf", "jpeg"):
        fig.write_image(path, format=format, width=width, height=height, scale=scale)
    else:
        raise ValueError(f"Unsupported format: {format}")


def resolve_color_column(
    df: pd.DataFrame,
    color_by: Optional[str],
) -> tuple[Optional[pd.Series], Optional[dict[str, str]]]:
    """
    Validate a ``color_by`` column and compute its categorical color mapping.

    Helper used by plotting functions that accept a ``color_by`` parameter.
    Returns ``(None, None)`` if ``color_by`` is None or missing from the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the potential color column.
    color_by : str, optional
        Column name.

    Returns
    -------
    tuple
        ``(values, color_map)`` — the column as a Series and a label→hex dict.
    """
    if color_by is None or color_by not in df.columns:
        return None, None
    values = df[color_by]
    color_map = get_categorical_colors(values.unique())
    return values, color_map
