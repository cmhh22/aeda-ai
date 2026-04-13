"""
aeda.engine.anomalies
Detección de anomalías en datos ambientales.
Isolation Forest, LOF, y métodos estadísticos.
"""

import numpy as np
import pandas as pd
from typing import Optional, Literal
from dataclasses import dataclass, field
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


@dataclass
class AnomalyResult:
    """Resultado de detección de anomalías."""
    method: str
    is_anomaly: np.ndarray  # boolean array
    scores: np.ndarray  # anomaly scores (más negativo = más anómalo)
    n_anomalies: int = 0
    anomaly_indices: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)

    def anomaly_mask(self) -> pd.Series:
        return pd.Series(self.is_anomaly, name="is_anomaly")


def run_isolation_forest(
    df: pd.DataFrame,
    contamination: float = 0.05,
    random_state: int = 42,
) -> AnomalyResult:
    """
    Isolation Forest: eficiente para alta dimensionalidad.
    
    Parameters
    ----------
    df : DataFrame escalado
    contamination : Proporción esperada de anomalías (0-0.5)
    """
    iso = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=200,
    )
    predictions = iso.fit_predict(df)
    scores = iso.decision_function(df)

    is_anomaly = predictions == -1
    anomaly_idx = df.index[is_anomaly].tolist()

    return AnomalyResult(
        method="Isolation Forest",
        is_anomaly=is_anomaly,
        scores=scores,
        n_anomalies=int(is_anomaly.sum()),
        anomaly_indices=anomaly_idx,
        diagnostics={
            "contamination": contamination,
            "n_estimators": 200,
            "score_threshold": float(np.percentile(scores, contamination * 100)),
        },
    )


def run_lof(
    df: pd.DataFrame,
    n_neighbors: int = 20,
    contamination: float = 0.05,
) -> AnomalyResult:
    """
    Local Outlier Factor: detecta anomalías basándose en densidad local.
    Bueno para clusters de diferente densidad.
    """
    n_neighbors = min(n_neighbors, len(df) - 1)

    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
    )
    predictions = lof.fit_predict(df)
    scores = lof.negative_outlier_factor_

    is_anomaly = predictions == -1
    anomaly_idx = df.index[is_anomaly].tolist()

    return AnomalyResult(
        method="LOF",
        is_anomaly=is_anomaly,
        scores=scores,
        n_anomalies=int(is_anomaly.sum()),
        anomaly_indices=anomaly_idx,
        diagnostics={
            "n_neighbors": n_neighbors,
            "contamination": contamination,
        },
    )


def run_statistical(
    df: pd.DataFrame,
    method: Literal["zscore", "iqr"] = "zscore",
    threshold: float = 3.0,
) -> AnomalyResult:
    """
    Detección estadística simple: Z-score o IQR.
    Una muestra es anómala si CUALQUIER variable excede el umbral.
    """
    if method == "zscore":
        z = np.abs((df - df.mean()) / df.std())
        is_anomaly_per_col = z > threshold
    elif method == "iqr":
        q1 = df.quantile(0.25)
        q3 = df.quantile(0.75)
        iqr = q3 - q1
        is_anomaly_per_col = (df < (q1 - threshold * iqr)) | (df > (q3 + threshold * iqr))
    else:
        raise ValueError(f"Método no soportado: {method}")

    is_anomaly = is_anomaly_per_col.any(axis=1).values
    # Score: número de variables anómalas por muestra
    scores = -is_anomaly_per_col.sum(axis=1).values.astype(float)

    anomaly_idx = df.index[is_anomaly].tolist()

    return AnomalyResult(
        method=f"Statistical ({method})",
        is_anomaly=is_anomaly,
        scores=scores,
        n_anomalies=int(is_anomaly.sum()),
        anomaly_indices=anomaly_idx,
        diagnostics={
            "statistical_method": method,
            "threshold": threshold,
            "anomalous_variables": {
                col: int(is_anomaly_per_col[col].sum())
                for col in df.columns
                if is_anomaly_per_col[col].sum() > 0
            },
        },
    )


def detect_anomalies(
    df: pd.DataFrame,
    method: Literal["isolation_forest", "lof", "zscore", "iqr", "auto"] = "auto",
    contamination: float = 0.05,
    **kwargs,
) -> AnomalyResult:
    """
    Interfaz unificada para detección de anomalías.

    Si method='auto', usa Isolation Forest (mejor rendimiento general).
    """
    if method == "auto":
        method = "isolation_forest"

    if method == "isolation_forest":
        return run_isolation_forest(df, contamination=contamination, **kwargs)
    elif method == "lof":
        return run_lof(df, contamination=contamination, **kwargs)
    elif method in ("zscore", "iqr"):
        return run_statistical(df, method=method, **kwargs)
    else:
        raise ValueError(f"Método no soportado: {method}")
