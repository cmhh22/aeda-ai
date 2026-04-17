"""
aeda.viz.profiles
Depth profile plots for sediment core data.

In sediment studies, depth is a proxy for time: deeper layers were deposited
earlier. Plotting concentration vs. depth for each site reveals whether a
contaminant has been accumulating, declining, or staying constant over time.
These plots are the natural complement to the spatial analysis done in PCA
and clustering.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from aeda.viz.base import (
    CATEGORICAL_PALETTE,
    apply_default_layout,
    get_categorical_colors,
)


# ============================================================
#  SINGLE VARIABLE DEPTH PROFILE
# ============================================================

def depth_profile(
    df: pd.DataFrame,
    variable: str,
    depth_col: str,
    site_col: Optional[str] = None,
    core_col: Optional[str] = None,
    title: Optional[str] = None,
    width: int = 900,
    height: int = 600,
) -> go.Figure:
    """
    Plot concentration of a single variable against depth, one line per site.

    Depth is plotted on the y-axis (inverted, so surface is at the top, which is
    the geological convention). Concentration is on the x-axis. Each site gets
    its own line and color.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the variable, depth, and site columns.
    variable : str
        Column name of the concentration to plot.
    depth_col : str
        Column name containing depth (e.g. ``"Profundidad"``).
    site_col : str, optional
        Column to group by. If ``None``, all samples are plotted as one line.
    core_col : str, optional
        Optional secondary grouping (e.g. core A vs. B). When given, different
        cores within the same site use different line dash patterns.
    title : str, optional
        Plot title.
    width, height : int
        Figure dimensions in pixels.

    Returns
    -------
    plotly.graph_objects.Figure
        Depth profile plot.
    """
    fig = go.Figure()

    if site_col is not None and site_col in df.columns:
        sites = df[site_col].unique()
        color_map = get_categorical_colors(sites)

        for site in sites:
            site_mask = df[site_col] == site
            if core_col is not None and core_col in df.columns:
                for i, core in enumerate(sorted(df.loc[site_mask, core_col].unique())):
                    mask = site_mask & (df[core_col] == core)
                    subset = df.loc[mask, [depth_col, variable]].dropna().sort_values(depth_col)
                    if subset.empty:
                        continue
                    dash = "solid" if i == 0 else "dash"
                    fig.add_trace(go.Scatter(
                        x=subset[variable], y=subset[depth_col],
                        mode="lines+markers",
                        name=f"{site} ({core})",
                        line=dict(color=color_map[site], width=2, dash=dash),
                        marker=dict(size=6),
                        hovertemplate=(
                            f"<b>{site}</b> (Core {core})<br>"
                            f"Depth: %{{y}} cm<br>"
                            f"{variable}: %{{x:.2f}}<extra></extra>"
                        ),
                    ))
            else:
                subset = df.loc[site_mask, [depth_col, variable]].dropna().sort_values(depth_col)
                if subset.empty:
                    continue
                fig.add_trace(go.Scatter(
                    x=subset[variable], y=subset[depth_col],
                    mode="lines+markers",
                    name=str(site),
                    line=dict(color=color_map[site], width=2),
                    marker=dict(size=6),
                    hovertemplate=(
                        f"<b>{site}</b><br>"
                        f"Depth: %{{y}} cm<br>"
                        f"{variable}: %{{x:.2f}}<extra></extra>"
                    ),
                ))
    else:
        subset = df[[depth_col, variable]].dropna().sort_values(depth_col)
        fig.add_trace(go.Scatter(
            x=subset[variable], y=subset[depth_col],
            mode="lines+markers",
            name=variable,
            line=dict(color=CATEGORICAL_PALETTE[0], width=2),
            marker=dict(size=6),
        ))

    fig.update_xaxes(title=variable, gridcolor="#F0F0F0")
    fig.update_yaxes(
        title=f"{depth_col} (cm)",
        autorange="reversed",  # Surface at top — geological convention
        gridcolor="#F0F0F0",
    )

    if title is None:
        title = f"Depth profile — {variable}"
    apply_default_layout(fig, title=title, width=width, height=height)

    return fig


# ============================================================
#  MULTI-VARIABLE DEPTH PROFILE GRID
# ============================================================

def depth_profile_grid(
    df: pd.DataFrame,
    variables: list[str],
    depth_col: str,
    site_col: Optional[str] = None,
    n_cols: int = 3,
    title: Optional[str] = None,
    width: int = 1100,
    height: Optional[int] = None,
) -> go.Figure:
    """
    Grid of depth profiles, one panel per variable.

    The same set of sites appears in every panel, letting the reader compare
    trends across variables at a glance. Ideal for showing the behavior of all
    heavy metals in a single figure.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with all variables, depth, and site columns.
    variables : list of str
        Variables to plot (one subplot per variable).
    depth_col : str
        Depth column name.
    site_col : str, optional
        Column to group and color by.
    n_cols : int
        Number of columns in the subplot grid.
    title : str, optional
        Plot title.
    width : int
        Figure width in pixels.
    height : int, optional
        Figure height. Auto-computed from ``n_rows * 300`` if omitted.

    Returns
    -------
    plotly.graph_objects.Figure
        Grid of depth profiles.
    """
    n = len(variables)
    n_rows = (n + n_cols - 1) // n_cols

    if height is None:
        height = max(400, n_rows * 320)

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=variables,
        horizontal_spacing=0.08,
        vertical_spacing=0.10 if n_rows <= 3 else 0.06,
    )

    if site_col is not None and site_col in df.columns:
        sites = df[site_col].unique()
        color_map = get_categorical_colors(sites)
    else:
        sites = [None]
        color_map = {None: CATEGORICAL_PALETTE[0]}

    # Track which sites have been added to the legend to avoid duplicates
    legend_added = set()

    for idx, variable in enumerate(variables):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        for site in sites:
            if site is None:
                subset = df[[depth_col, variable]].dropna().sort_values(depth_col)
                name = variable
                color = color_map[None]
            else:
                mask = df[site_col] == site
                subset = df.loc[mask, [depth_col, variable]].dropna().sort_values(depth_col)
                name = str(site)
                color = color_map[site]

            if subset.empty:
                continue

            show_legend = (site not in legend_added) if site is not None else (idx == 0)
            if site is not None:
                legend_added.add(site)

            fig.add_trace(go.Scatter(
                x=subset[variable], y=subset[depth_col],
                mode="lines+markers",
                name=name,
                legendgroup=name,
                showlegend=show_legend,
                line=dict(color=color, width=1.5),
                marker=dict(size=5),
                hovertemplate=f"<b>{name}</b><br>Depth: %{{y}} cm<br>{variable}: %{{x:.2f}}<extra></extra>",
            ), row=row, col=col)

        fig.update_xaxes(title_text=variable, row=row, col=col, gridcolor="#F0F0F0")
        fig.update_yaxes(
            title_text=f"{depth_col} (cm)" if col == 1 else None,
            autorange="reversed",
            gridcolor="#F0F0F0",
            row=row, col=col,
        )

    if title is None:
        title = f"Depth profiles — {len(variables)} variables"
    apply_default_layout(fig, title=title, width=width, height=height)

    return fig
