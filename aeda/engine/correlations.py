"""
aeda.engine.correlations
Análisis de correlaciones multi-método para matrices ambientales.
Pearson, Spearman, y detección de relaciones no-lineales.
"""

import numpy as np
import pandas as pd
from typing import Optional, Literal
from dataclasses import dataclass, field


@dataclass
class CorrelationResult:
    """Resultado de un análisis de correlaciones."""
    method: str
    matrix: pd.DataFrame
    significant_pairs: list = field(default_factory=list)  # pares con |r| > threshold
    n_strong: int = 0
    n_moderate: int = 0
    diagnostics: dict = field(default_factory=dict)


def _extract_significant_pairs(
    corr_matrix: pd.DataFrame,
    threshold_strong: float = 0.7,
    threshold_moderate: float = 0.5,
) -> tuple[list[dict], int, int]:
    """Extrae pares de variables con correlaciones significativas."""
    pairs = []
    n_strong = 0
    n_moderate = 0
    cols = corr_matrix.columns

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr_matrix.iloc[i, j]
            abs_r = abs(r)
            if abs_r >= threshold_moderate:
                pair = {
                    "var1": cols[i],
                    "var2": cols[j],
                    "correlation": float(r),
                    "abs_correlation": float(abs_r),
                    "strength": "strong" if abs_r >= threshold_strong else "moderate",
                }
                pairs.append(pair)
                if abs_r >= threshold_strong:
                    n_strong += 1
                else:
                    n_moderate += 1

    pairs.sort(key=lambda x: x["abs_correlation"], reverse=True)
    return pairs, n_strong, n_moderate


def run_correlation(
    df: pd.DataFrame,
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
    threshold_strong: float = 0.7,
    threshold_moderate: float = 0.5,
) -> CorrelationResult:
    """
    Calcula matriz de correlación y extrae pares significativos.
    """
    matrix = df.corr(method=method)
    pairs, n_strong, n_moderate = _extract_significant_pairs(
        matrix, threshold_strong, threshold_moderate
    )

    return CorrelationResult(
        method=method.capitalize(),
        matrix=matrix,
        significant_pairs=pairs,
        n_strong=n_strong,
        n_moderate=n_moderate,
        diagnostics={
            "n_variables": len(df.columns),
            "n_pairs_total": len(df.columns) * (len(df.columns) - 1) // 2,
            "threshold_strong": threshold_strong,
            "threshold_moderate": threshold_moderate,
        },
    )


def compare_methods(
    df: pd.DataFrame,
) -> dict[str, CorrelationResult]:
    """
    Ejecuta Pearson y Spearman, y compara los resultados.
    Diferencias grandes entre ambos sugieren relaciones no-lineales.
    """
    pearson = run_correlation(df, method="pearson")
    spearman = run_correlation(df, method="spearman")

    # Detectar pares donde Spearman >> Pearson (relación no-lineal)
    diff_matrix = spearman.matrix.abs() - pearson.matrix.abs()
    nonlinear_candidates = []

    cols = df.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            d = diff_matrix.iloc[i, j]
            if d > 0.15:  # Spearman much higher than Pearson
                nonlinear_candidates.append({
                    "var1": cols[i],
                    "var2": cols[j],
                    "pearson": float(pearson.matrix.iloc[i, j]),
                    "spearman": float(spearman.matrix.iloc[i, j]),
                    "difference": float(d),
                })

    return {
        "pearson": pearson,
        "spearman": spearman,
        "nonlinear_candidates": nonlinear_candidates,
    }


def correlate(
    df: pd.DataFrame,
    method: Literal["pearson", "spearman", "kendall", "compare", "auto"] = "compare",
    **kwargs,
) -> CorrelationResult | dict:
    """
    Interfaz unificada.
    method='compare' o 'auto' ejecuta Pearson + Spearman y detecta no-linealidad.
    """
    if method in ("compare", "auto"):
        return compare_methods(df)
    return run_correlation(df, method=method, **kwargs)
