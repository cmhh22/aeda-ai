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
        baseline_concs = _validate_user_baseline(
            custom_baseline,
            metals=metals,
            reference_element=reference_element,
            df=df,
            site_col=site_col,
        )
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

        # Determine which baseline applies to this row.
        if site_col is not None and baseline_strategy in ("deepest", "user") and "__global__" not in baseline_concs:
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


def _validate_user_baseline(
    custom_baseline: dict,
    metals: list[str],
    reference_element: str,
    df: pd.DataFrame,
    site_col: Optional[str],
) -> dict:
    """Validate and normalize a user-provided custom_baseline.

    Accepted formats:
    1. Global flat dict: {reference_element: float, metal1: float, ...}
       → Wrapped as {"__global__": {...}}.
    2. Per-site dict (when site_col is provided):
       {site1: {reference_element: float, metal1: float, ...}, ...}
       → Returned as-is, after validating coverage.

    Raises ValueError if the structure does not match either format or is incomplete.
    """
    if not isinstance(custom_baseline, dict) or not custom_baseline:
        raise ValueError("custom_baseline must be a non-empty dictionary.")

    required_keys = {reference_element, *metals}

    # Heuristic: if all top-level keys are column names of the dataset, this is a
    # flat global dict; otherwise we assume keys are site identifiers.
    top_level_keys = set(custom_baseline.keys())
    looks_global = required_keys.issubset(top_level_keys) or "__global__" in top_level_keys

    if looks_global:
        if "__global__" in custom_baseline:
            global_base = custom_baseline["__global__"]
        else:
            global_base = custom_baseline
        missing = required_keys - set(global_base.keys())
        if missing:
            raise ValueError(
                f"custom_baseline (global) is missing required keys: {sorted(missing)}"
            )
        return {"__global__": dict(global_base)}

    # Per-site format
    if site_col is None:
        raise ValueError(
            "custom_baseline appears to be per-site but site_col was not provided. "
            "Either provide site_col or pass a flat global dict."
        )

    available_sites = set(df[site_col].dropna().unique())
    provided_sites = set(custom_baseline.keys())

    missing_sites = available_sites - provided_sites
    if missing_sites:
        raise ValueError(
            f"custom_baseline does not cover the following sites in the dataset: "
            f"{sorted(missing_sites)}. Provide a baseline entry for each site, "
            f"or use a single global baseline."
        )

    validated: dict = {}
    for site, site_base in custom_baseline.items():
        if not isinstance(site_base, dict):
            raise ValueError(
                f"custom_baseline['{site}'] must be a dictionary mapping element to value."
            )
        missing = required_keys - set(site_base.keys())
        if missing:
            raise ValueError(
                f"custom_baseline['{site}'] is missing required keys: {sorted(missing)}"
            )
        validated[site] = dict(site_base)

    return validated
