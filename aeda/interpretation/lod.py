"""Handle values below the limit of detection (LOD) using Succop et al. (2004).

This module implements the criterion recommended by the project's scientific
advisor (Yoelvis Bolanos-Alvarez):

- If the variable follows a Normal distribution: replace <LOD values by LOD/2
- If the variable does NOT follow a Normal distribution: replace by LOD/sqrt(2)

Normality is tested per-variable using the Shapiro-Wilk test at alpha=0.05.

References
----------
Succop, P. A., Clark, S., Chen, M., & Galke, W. (2004). Imputation of data
values that are less than a detection limit. J. Occupational and Environmental
Hygiene, 1(7), 436-441.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class LODImputationLog:
    """Record of LOD imputation decisions for audit and reporting."""

    variable_decisions: dict = field(default_factory=dict)
    alpha: float = 0.05

    def summary(self) -> str:
        lines = ["=" * 60, "LOD IMPUTATION LOG", "=" * 60]
        if not self.variable_decisions:
            lines.append("No variables had LOD values to impute.")
            return "\n".join(lines)
        for var, info in self.variable_decisions.items():
            p_value = info["p_value"]
            p_text = f"{p_value:.4f}" if pd.notna(p_value) else "nan"
            lines.append(
                f"  {var}: n_imputed={info['n_imputed']}, "
                f"normal={info['normal']} (p={p_text}), "
                f"method={info['method']}"
            )
        return "\n".join(lines)


def handle_lod_values(
    df: pd.DataFrame,
    lod_values: dict[str, float],
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, LODImputationLog]:
    """Replace values below the limit of detection (LOD) using Yoelvis criterion.

    For each variable in lod_values:
    1. Test whether non-missing, non-below-LOD values are Normal (Shapiro-Wilk).
    2. If Normal (p > alpha): replace values flagged as <LOD by LOD/2.
    3. If not Normal: replace by LOD/sqrt(2).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with numeric columns. Values below LOD can be pre-marked as
        NaN; values less than the LOD itself are also treated as below-LOD.
    lod_values : dict[str, float]
        Mapping from variable name to limit of detection in the same units as data.
    alpha : float
        Significance level for Shapiro-Wilk normality test.

    Returns
    -------
    tuple[pd.DataFrame, LODImputationLog]
        DataFrame with imputed values and a per-variable decision log.
    """
    out = df.copy()
    log = LODImputationLog(alpha=alpha)

    for var, lod in lod_values.items():
        if var not in out.columns:
            continue
        if lod <= 0:
            raise ValueError(f"LOD for variable '{var}' must be positive, got {lod}")

        col = out[var]
        below_lod_mask = col.isna() | (col < lod)
        valid_values = col[~below_lod_mask].dropna()

        if len(valid_values) < 3:
            is_normal = False
            p_value = float("nan")
        else:
            if len(valid_values) > 5000:
                rng = np.random.default_rng(42)
                sample = rng.choice(valid_values.values, size=5000, replace=False)
            else:
                sample = valid_values.values
            try:
                _, p_value = stats.shapiro(sample)
                is_normal = p_value > alpha
            except Exception:
                is_normal = False
                p_value = float("nan")

        if is_normal:
            replacement = lod / 2.0
            method = "LOD/2 (Normal)"
        else:
            replacement = lod / np.sqrt(2.0)
            method = "LOD/sqrt(2) (non-Normal)"

        n_imputed = int(below_lod_mask.sum())
        if n_imputed > 0:
            out.loc[below_lod_mask, var] = replacement

        log.variable_decisions[var] = {
            "normal": is_normal,
            "p_value": float(p_value),
            "method": method,
            "n_imputed": n_imputed,
            "replacement_value": float(replacement),
            "lod": float(lod),
        }

    return out, log
