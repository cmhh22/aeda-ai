"""
aeda.engine.feature_analysis
Feature importance analysis and contribution to data structure.
Random Forest importance and permutation importance.
"""

import numpy as np
import pandas as pd
from typing import Optional, Literal
from dataclasses import dataclass, field
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance


@dataclass
class FeatureImportanceResult:
    """Result of a feature importance analysis."""
    method: str
    importances: pd.Series  # feature -> importance (sorted)
    target_variable: str
    diagnostics: dict = field(default_factory=dict)

    def top_n(self, n: int = 10) -> pd.Series:
        return self.importances.head(n)


def rank_by_variance(df: pd.DataFrame) -> pd.Series:
    """
    Rank features by variance.

    Note: if data is already standardized (mean ~ 0, std ~ 1),
    CV is not meaningful. Plain variance is always well-defined and
    remains interpretable regardless of scaling.
    """
    variance = df.var()
    return variance.sort_values(ascending=False).rename("variance")


def rank_by_rf_importance(
    df: pd.DataFrame,
    target: str,
    task: str = "auto",
    n_estimators: int = 100,
    random_state: int = 42,
) -> FeatureImportanceResult:
    """
    Compute feature importance using Random Forest.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with all variables (features + target).
    target : str
        Target column name.
    task : str
        'classification', 'regression', or 'auto' (inferred from target).
    """
    X = df.drop(columns=[target])
    y = df[target]

    if task == "auto":
        task = "classification" if y.nunique() < 15 else "regression"

    if task == "classification":
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    else:
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)

    model.fit(X, y)

    importances = pd.Series(
        model.feature_importances_,
        index=X.columns,
    ).sort_values(ascending=False)

    # Permutation importance for cross-checking
    perm = permutation_importance(model, X, y, n_repeats=10, random_state=random_state)
    perm_importances = pd.Series(
        perm.importances_mean,
        index=X.columns,
    ).sort_values(ascending=False)

    return FeatureImportanceResult(
        method=f"Random Forest ({task})",
        importances=importances,
        target_variable=target,
        diagnostics={
            "task": task,
            "n_estimators": n_estimators,
            "oob_score": getattr(model, "oob_score_", None),
            "permutation_importances": perm_importances.to_dict(),
            "n_features": len(X.columns),
        },
    )


def rank_by_cluster_discrimination(
    df: pd.DataFrame,
    labels: np.ndarray,
) -> FeatureImportanceResult:
    """
    Compute which variables contribute most to separating clusters.
    Uses Random Forest with cluster labels as target.
    """
    df_with_labels = df.copy()
    df_with_labels["_cluster"] = labels

    return rank_by_rf_importance(
        df_with_labels,
        target="_cluster",
        task="classification",
    )


def analyze_features(
    df: pd.DataFrame,
    target: Optional[str] = None,
    cluster_labels: Optional[np.ndarray] = None,
    method: Literal["auto", "variance", "random_forest"] = "auto",
) -> FeatureImportanceResult:
    """
    Unified interface.
    If target is provided, uses Random Forest with that target.
    If cluster_labels is provided, analyzes discrimination between clusters.
    If neither is provided, uses variance as a proxy.
    """
    if method == "variance":
        variance_ranking = rank_by_variance(df)
        return FeatureImportanceResult(
            method="Variance",
            importances=variance_ranking,
            target_variable="(none - variability-based ranking)",
            diagnostics={"method": "variance", "n_features": len(df.columns)},
        )

    if target and target in df.columns:
        return rank_by_rf_importance(df, target=target)
    elif cluster_labels is not None:
        return rank_by_cluster_discrimination(df, cluster_labels)
    else:
        variance_ranking = rank_by_variance(df)
        return FeatureImportanceResult(
            method="Variance",
            importances=variance_ranking,
            target_variable="(none - variability-based ranking)",
            diagnostics={"method": "variance", "n_features": len(df.columns)},
        )
