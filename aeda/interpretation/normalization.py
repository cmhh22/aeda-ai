"""Enrichment Factor (EF) computation for geochemical normalization.

Formula:

    EF = (Metal / Reference)_sample / (Metal / Reference)_baseline

Baseline strategy
-----------------
Without radiometric dating, this module uses the deepest core section per site
as an approximation of pre-anthropogenic baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class EnrichmentFactorResult:
    """Result of EF computation for one or more metals."""

    ef_values: pd.DataFrame
    reference_element: str
    baseline_concentrations: dict
    baseline_strategy: str
    diagnostics: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = ["=" * 60, "ENRICHMENT FACTOR RESULT", "=" * 60]
        lines.append(f"Reference element: {self.reference_element}")
        lines.append(f"Baseline strategy: {self.baseline_strategy}")
        lines.append(f"Metals analyzed: {list(self.ef_values.columns)}")
        lines.append(f"Samples: {len(self.ef_values)}")
        lines.append("\nEF descriptive statistics per metal:")
        lines.append(self.ef_values.describe().T.to_string())
        return "\n".join(lines)


def compute_enrichment_factor(
    df: pd.DataFrame,
    metals: list[str],
    reference_element: str = "Al",
    site_col: Optional[str] = None,
    depth_col: Optional[str] = None,
    baseline_strategy: str = "deepest",
    custom_baseline: Optional[dict] = None,
) -> EnrichmentFactorResult:
    """Compute EF = (Metal/Ref)_sample / (Metal/Ref)_baseline per sample."""
    if reference_element not in df.columns:
        raise ValueError(
            f"Reference element '{reference_element}' not found in DataFrame. "
            f"Available numeric columns: {list(df.select_dtypes(include='number').columns)}"
        )
    if reference_element in metals:
        raise ValueError(
            f"Reference element '{reference_element}' cannot be in the metals list."
        )
    missing = [m for m in metals if m not in df.columns]
    if missing:
        raise ValueError(f"Metal columns not found in DataFrame: {missing}")

    if baseline_strategy == "deepest" and depth_col is None:
        raise ValueError(
            "baseline_strategy='deepest' requires depth_col to identify deepest samples"
        )
    if baseline_strategy == "user" and custom_baseline is None:
        raise ValueError(
            "baseline_strategy='user' requires custom_baseline dictionary"
        )

    baseline_concs: dict = {}

    if baseline_strategy == "deepest":
        if site_col is not None:
            for site, sub in df.groupby(site_col):
                deepest_idx = sub[depth_col].idxmax()
                site_baseline = {reference_element: float(df.at[deepest_idx, reference_element])}
                for m in metals:
                    site_baseline[m] = float(df.at[deepest_idx, m])
                baseline_concs[site] = site_baseline
        else:
            deepest_idx = df[depth_col].idxmax()
            global_baseline = {reference_element: float(df.at[deepest_idx, reference_element])}
            for m in metals:
                global_baseline[m] = float(df.at[deepest_idx, m])
            baseline_concs["__global__"] = global_baseline

    elif baseline_strategy == "global_min_depth":
        if depth_col is None:
            raise ValueError("baseline_strategy='global_min_depth' requires depth_col")
        deepest_idx = df[depth_col].idxmax()
        global_baseline = {reference_element: float(df.at[deepest_idx, reference_element])}
        for m in metals:
            global_baseline[m] = float(df.at[deepest_idx, m])
        baseline_concs["__global__"] = global_baseline

    elif baseline_strategy == "user":
        baseline_concs = custom_baseline
    else:
        raise ValueError(
            f"Unknown baseline_strategy '{baseline_strategy}'. "
            "Valid: 'deepest', 'global_min_depth', 'user'"
        )

    ef_df = pd.DataFrame(index=df.index, columns=metals, dtype=float)

    for idx, row in df.iterrows():
        ref_val_sample = row[reference_element]
        if pd.isna(ref_val_sample) or ref_val_sample == 0:
            ef_df.loc[idx, :] = np.nan
            continue

        if site_col is not None and baseline_strategy == "deepest":
            site = row[site_col]
            if site not in baseline_concs:
                ef_df.loc[idx, :] = np.nan
                continue
            base = baseline_concs[site]
        else:
            base = baseline_concs.get("__global__", next(iter(baseline_concs.values())))

        ref_val_base = base.get(reference_element)
        if ref_val_base is None or ref_val_base == 0:
            ef_df.loc[idx, :] = np.nan
            continue

        for m in metals:
            metal_val_sample = row[m]
            metal_val_base = base.get(m)
            if pd.isna(metal_val_sample) or metal_val_base is None or metal_val_base == 0:
                ef_df.at[idx, m] = np.nan
            else:
                ef_df.at[idx, m] = (metal_val_sample / ref_val_sample) / (
                    metal_val_base / ref_val_base
                )

    return EnrichmentFactorResult(
        ef_values=ef_df,
        reference_element=reference_element,
        baseline_concentrations=baseline_concs,
        baseline_strategy=baseline_strategy,
        diagnostics={
            "n_samples": len(df),
            "n_metals": len(metals),
            "n_sites": len(baseline_concs) if site_col else 1,
        },
    )
