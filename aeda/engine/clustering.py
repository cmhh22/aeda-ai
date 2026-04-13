"""
aeda.engine.clustering
Módulo de clustering para datos ambientales.
K-Means, DBSCAN, Hierarchical con selección automática de parámetros.
"""

import numpy as np
import pandas as pd
from typing import Optional, Literal
from dataclasses import dataclass, field
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors


@dataclass
class ClusteringResult:
    """Resultado de un análisis de clustering."""
    method: str
    labels: np.ndarray
    n_clusters: int
    metrics: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    centroids: Optional[np.ndarray] = None

    def label_series(self, index=None) -> pd.Series:
        return pd.Series(self.labels, index=index, name="cluster")


def _evaluate_clustering(X: np.ndarray, labels: np.ndarray) -> dict:
    """Calcula métricas de calidad del clustering."""
    unique_labels = set(labels)
    unique_labels.discard(-1)  # DBSCAN noise
    n_clusters = len(unique_labels)

    if n_clusters < 2 or n_clusters >= len(X):
        return {"n_clusters": n_clusters, "silhouette": None}

    mask = labels != -1
    if mask.sum() < n_clusters + 1:
        return {"n_clusters": n_clusters, "silhouette": None}

    return {
        "n_clusters": n_clusters,
        "silhouette": float(silhouette_score(X[mask], labels[mask])),
        "calinski_harabasz": float(calinski_harabasz_score(X[mask], labels[mask])),
        "davies_bouldin": float(davies_bouldin_score(X[mask], labels[mask])),
        "n_noise": int((labels == -1).sum()),
    }


def find_optimal_k(
    X: np.ndarray,
    k_range: tuple[int, int] = (2, 10),
    method: str = "silhouette",
) -> dict:
    """
    Encuentra el K óptimo para K-Means usando silhouette o elbow.

    Returns
    -------
    dict con 'optimal_k', 'scores', y 'all_results'
    """
    k_min, k_max = k_range
    k_max = min(k_max, len(X) - 1)
    scores = {}
    inertias = {}

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        scores[k] = silhouette_score(X, labels)
        inertias[k] = km.inertia_

    optimal_k = max(scores, key=scores.get)

    return {
        "optimal_k": optimal_k,
        "silhouette_scores": scores,
        "inertias": inertias,
    }


def run_kmeans(
    df: pd.DataFrame,
    n_clusters: Optional[int] = None,
    k_range: tuple[int, int] = (2, 10),
) -> ClusteringResult:
    """
    K-Means con selección automática de K si n_clusters es None.
    """
    X = df.values

    if n_clusters is None:
        opt = find_optimal_k(X, k_range)
        n_clusters = opt["optimal_k"]
        diag = {"auto_k": True, **opt}
    else:
        diag = {"auto_k": False}

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    metrics = _evaluate_clustering(X, labels)

    return ClusteringResult(
        method="K-Means",
        labels=labels,
        n_clusters=n_clusters,
        metrics=metrics,
        diagnostics=diag,
        centroids=km.cluster_centers_,
    )


def _estimate_eps(X: np.ndarray, k: int = 5) -> float:
    """Estima eps para DBSCAN usando el método del k-NN distance plot."""
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    sorted_distances = np.sort(distances[:, -1])

    # Buscar el "codo" usando la segunda derivada
    diffs = np.diff(sorted_distances)
    diffs2 = np.diff(diffs)
    if len(diffs2) > 0:
        knee_idx = np.argmax(diffs2) + 1
        return float(sorted_distances[knee_idx])

    return float(np.percentile(sorted_distances, 90))


def run_dbscan(
    df: pd.DataFrame,
    eps: Optional[float] = None,
    min_samples: int = 5,
) -> ClusteringResult:
    """
    DBSCAN con estimación automática de eps si no se provee.
    Útil para datos con clusters de forma irregular y detección de noise.
    """
    X = df.values

    if eps is None:
        eps = _estimate_eps(X, k=min_samples)
        auto_eps = True
    else:
        auto_eps = False

    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)

    unique_labels = set(labels)
    unique_labels.discard(-1)
    n_clusters = len(unique_labels)

    metrics = _evaluate_clustering(X, labels)

    return ClusteringResult(
        method="DBSCAN",
        labels=labels,
        n_clusters=n_clusters,
        metrics=metrics,
        diagnostics={
            "eps": eps,
            "auto_eps": auto_eps,
            "min_samples": min_samples,
        },
    )


def run_hierarchical(
    df: pd.DataFrame,
    n_clusters: Optional[int] = None,
    linkage: Literal["ward", "complete", "average", "single"] = "ward",
) -> ClusteringResult:
    """
    Clustering jerárquico aglomerativo.
    """
    X = df.values

    if n_clusters is None:
        # Usar silhouette para encontrar K óptimo
        best_k, best_score = 2, -1
        for k in range(2, min(11, len(X))):
            hc = AgglomerativeClustering(n_clusters=k, linkage=linkage)
            labels = hc.fit_predict(X)
            score = silhouette_score(X, labels)
            if score > best_score:
                best_k, best_score = k, score
        n_clusters = best_k

    hc = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = hc.fit_predict(X)
    metrics = _evaluate_clustering(X, labels)

    return ClusteringResult(
        method=f"Hierarchical ({linkage})",
        labels=labels,
        n_clusters=n_clusters,
        metrics=metrics,
        diagnostics={"linkage": linkage},
    )


def cluster(
    df: pd.DataFrame,
    method: Literal["kmeans", "dbscan", "hierarchical", "auto"] = "auto",
    n_clusters: Optional[int] = None,
    **kwargs,
) -> ClusteringResult:
    """
    Interfaz unificada para clustering.

    Si method='auto', ejecuta K-Means y DBSCAN, y retorna el de mejor silhouette.
    """
    if method == "auto":
        results = []
        results.append(run_kmeans(df, n_clusters=n_clusters))
        results.append(run_dbscan(df, **kwargs))

        # Seleccionar el mejor por silhouette
        best = max(
            [r for r in results if r.metrics.get("silhouette") is not None],
            key=lambda r: r.metrics["silhouette"],
            default=results[0],
        )
        best.diagnostics["auto_selected"] = True
        best.diagnostics["compared_methods"] = [
            {"method": r.method, "silhouette": r.metrics.get("silhouette")}
            for r in results
        ]
        return best

    if method == "kmeans":
        return run_kmeans(df, n_clusters=n_clusters, **kwargs)
    elif method == "dbscan":
        return run_dbscan(df, **kwargs)
    elif method == "hierarchical":
        return run_hierarchical(df, n_clusters=n_clusters, **kwargs)
    else:
        raise ValueError(f"Método no soportado: {method}")
