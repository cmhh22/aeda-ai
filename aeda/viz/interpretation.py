"""Visualization functions for environmental interpretation results."""

from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from aeda.interpretation.normalization import EnrichmentFactorResult
from aeda.viz.base import CATEGORICAL_PALETTE, apply_default_layout, get_categorical_colors


EF_BAND_COLORS = {
    "no_enrichment": "#2ca02c",
    "minor": "#bcbd22",
    "moderate": "#ff7f0e",
    "moderately_severe": "#d62728",
    "severe": "#8c564b",
    "very_severe": "#9467bd",
    "extremely_severe": "#1f1f1f",
}

TEL_PEL_COLORS = {
    "below_TEL": "#2ca02c",
    "TEL_to_PEL": "#ff7f0e",
    "above_PEL": "#d62728",
    "no_thresholds": "#999999",
}


def enrichment_factor_depth_plot(
    ef_result: EnrichmentFactorResult,
    df: pd.DataFrame,
    depth_col: str,
    site_col: Optional[str] = None,
    metals: Optional[list[str]] = None,
    n_cols: int = 3,
    height_per_row: int = 260,
) -> go.Figure:
    """Plot EF vs depth for multiple metals with Birch threshold bands."""
    if metals is None:
        metals = list(ef_result.ef_values.columns)

    n_metals = len(metals)
    n_rows = (n_metals + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=metals,
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    bands = [2, 3, 5, 10, 25, 50]

    if site_col and site_col in df.columns:
        site_values = list(df[site_col].dropna().unique())
    else:
        site_values = [None]

    if site_col:
        color_map = get_categorical_colors(site_values)
    else:
        color_map = {None: CATEGORICAL_PALETTE[0]}

    for i, metal in enumerate(metals):
        row = i // n_cols + 1
        col = i % n_cols + 1

        merged = df[[depth_col]].copy()
        if site_col and site_col in df.columns:
            merged[site_col] = df[site_col]
        merged["ef"] = ef_result.ef_values[metal]
        merged = merged.dropna(subset=["ef", depth_col])

        if site_col and site_col in merged.columns:
            show_legend_for_this_subplot = i == 0
            for site in site_values:
                site_data = merged[merged[site_col] == site].sort_values(depth_col)
                if len(site_data) == 0:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=site_data["ef"],
                        y=site_data[depth_col],
                        mode="lines+markers",
                        name=str(site),
                        line=dict(color=color_map[site]),
                        marker=dict(size=5),
                        showlegend=show_legend_for_this_subplot,
                        legendgroup=str(site),
                    ),
                    row=row,
                    col=col,
                )
        else:
            site_data = merged.sort_values(depth_col)
            fig.add_trace(
                go.Scatter(
                    x=site_data["ef"],
                    y=site_data[depth_col],
                    mode="lines+markers",
                    line=dict(color=CATEGORICAL_PALETTE[0]),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

        for b in bands:
            fig.add_vline(
                x=b,
                line=dict(color="gray", dash="dot", width=0.8),
                row=row,
                col=col,
            )

        fig.update_xaxes(title_text="EF", type="log", row=row, col=col)
        fig.update_yaxes(title_text=depth_col, autorange="reversed", row=row, col=col)

    fig.update_layout(
        height=height_per_row * n_rows,
        title_text=f"Enrichment Factor vs {depth_col} (reference: {ef_result.reference_element})",
    )
    apply_default_layout(fig)
    return fig


def contamination_classification_plot(
    classifications: pd.DataFrame,
    kind: str = "tel_pel",
    title: Optional[str] = None,
) -> go.Figure:
    """Stacked bar chart of classification counts per metal."""
    if kind == "tel_pel":
        color_map = TEL_PEL_COLORS
        default_title = "TEL/PEL Classification by Metal"
        order = ["below_TEL", "TEL_to_PEL", "above_PEL", "no_thresholds"]
    elif kind == "ef":
        color_map = EF_BAND_COLORS
        default_title = "Enrichment Factor Classification by Metal (Birch 2003)"
        order = [
            "no_enrichment",
            "minor",
            "moderate",
            "moderately_severe",
            "severe",
            "very_severe",
            "extremely_severe",
        ]
    else:
        raise ValueError(f"Unknown kind '{kind}'. Use 'tel_pel' or 'ef'.")

    counts = pd.DataFrame(0, index=classifications.columns, columns=order, dtype=int)
    for metal in classifications.columns:
        vc = classifications[metal].value_counts(dropna=True)
        for label in order:
            if label in vc.index:
                counts.at[metal, label] = int(vc[label])

    fig = go.Figure()
    for label in order:
        if counts[label].sum() == 0:
            continue
        fig.add_trace(
            go.Bar(
                name=label.replace("_", " "),
                x=counts.index,
                y=counts[label],
                marker_color=color_map.get(label, "#999999"),
            )
        )

    fig.update_layout(
        barmode="stack",
        title=title or default_title,
        xaxis_title="Metal",
        yaxis_title="Number of samples",
    )
    apply_default_layout(fig)
    return fig
