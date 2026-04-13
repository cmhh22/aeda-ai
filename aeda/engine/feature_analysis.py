"""
aeda.engine.feature_analysis
Análisis de importancia de variables y su contribución al patrón de datos.
Random Forest importance, permutation importance.
"""

import numpy as np
import pandas as pd
from typing import Optional, Literal
from dataclasses import dataclass, field
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance


@dataclass
class FeatureImportanceResult:
    """Resultado del análisis de importancia de variables."""
    method: str
    importances: pd.Series  # variable -> importancia (ordenada)
    target_variable: str
    diagnostics: dict = field(default_factory=dict)

    def top_n(self, n: int = 10) -> pd.Series:
        return self.importances.head(n)


def rank_by_variance(df: pd.DataFrame) -> pd.Series:
    """Ranking simple por varianza (coeficiente de variación)."""
    cv = df.std() / df.mean().abs().replace(0, np.nan)
    return cv.sort_values(ascending=False).rename("cv")


def rank_by_rf_importance(
    df: pd.DataFrame,
    target: str,
    task: str = "auto",
    n_estimators: int = 100,
    random_state: int = 42,
) -> FeatureImportanceResult:
    """
    Calcula importancia de variables usando Random Forest.

    Parameters
    ----------
    df : DataFrame con todas las variables (features + target)
    target : Nombre de la columna target
    task : 'classification', 'regression', o 'auto' (infiere del target)
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

    # Permutation importance para validar
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
    Calcula qué variables contribuyen más a discriminar entre clusters.
    Usa RF con los cluster labels como target.
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
    Interfaz unificada.
    Si target es dado, usa RF con ese target.
    Si cluster_labels es dado, analiza discriminación entre clusters.
    Si ninguno, usa varianza como proxy.
    """
    if method == "variance":
        variance_ranking = rank_by_variance(df)
        return FeatureImportanceResult(
            method="Coefficient of Variation",
            importances=variance_ranking,
            target_variable="(ninguno - ranking por variabilidad)",
            diagnostics={"method": "cv", "n_features": len(df.columns)},
        )

    if target and target in df.columns:
        return rank_by_rf_importance(df, target=target)
    elif cluster_labels is not None:
        return rank_by_cluster_discrimination(df, cluster_labels)
    else:
        variance_ranking = rank_by_variance(df)
        return FeatureImportanceResult(
            method="Coefficient of Variation",
            importances=variance_ranking,
            target_variable="(ninguno - ranking por variabilidad)",
            diagnostics={"method": "cv", "n_features": len(df.columns)},
        )
