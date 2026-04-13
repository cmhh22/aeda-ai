"""
aeda.engine.dimensionality
Módulo de reducción dimensional para datos ambientales multivariados.
PCA, t-SNE, UMAP con selección automática y diagnósticos.
"""

import numpy as np
import pandas as pd
from typing import Optional, Literal
from dataclasses import dataclass, field
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


@dataclass
class DimReductionResult:
    """Resultado de una reducción dimensional."""
    method: str
    components: pd.DataFrame  # coordenadas transformadas
    explained_variance: Optional[np.ndarray] = None  # solo PCA
    cumulative_variance: Optional[np.ndarray] = None  # solo PCA
    n_components_selected: int = 0
    loadings: Optional[pd.DataFrame] = None  # solo PCA: contribución de cada variable
    feature_names: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


def run_pca(
    df: pd.DataFrame,
    n_components: Optional[int] = None,
    variance_threshold: float = 0.85,
) -> DimReductionResult:
    """
    PCA con selección automática de componentes.

    Si n_components es None, selecciona el número mínimo que explique
    al menos variance_threshold de la varianza total.

    Parameters
    ----------
    df : DataFrame escalado con variables numéricas
    n_components : Número de componentes. None = auto-selección.
    variance_threshold : Varianza acumulada mínima para auto-selección (0-1).

    Returns
    -------
    DimReductionResult con componentes, varianza explicada y loadings.
    """
    # PCA completo primero para decidir n_components
    pca_full = PCA()
    pca_full.fit(df)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)

    if n_components is None:
        n_components = int(np.argmax(cumvar >= variance_threshold) + 1)
        n_components = max(2, min(n_components, df.shape[1]))

    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(df)

    col_names = [f"PC{i+1}" for i in range(n_components)]
    components_df = pd.DataFrame(transformed, columns=col_names, index=df.index)

    # Loadings: contribución de cada variable original a cada PC
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=col_names,
        index=df.columns,
    )

    # Kaiser criterion: componentes con eigenvalue > 1
    eigenvalues = pca.explained_variance_
    kaiser_n = int(np.sum(eigenvalues > 1.0))

    return DimReductionResult(
        method="PCA",
        components=components_df,
        explained_variance=pca.explained_variance_ratio_,
        cumulative_variance=cumvar[:n_components],
        n_components_selected=n_components,
        loadings=loadings,
        feature_names=df.columns.tolist(),
        diagnostics={
            "total_variance_explained": float(cumvar[n_components - 1]),
            "kaiser_criterion_n": kaiser_n,
            "eigenvalues": eigenvalues[:n_components].tolist(),
            "n_features": df.shape[1],
            "n_samples": df.shape[0],
        },
    )


def run_tsne(
    df: pd.DataFrame,
    n_components: int = 2,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> DimReductionResult:
    """
    t-SNE para visualización no-lineal de clusters.

    Parameters
    ----------
    df : DataFrame escalado
    n_components : 2 o 3
    perplexity : Controla el balance local/global (5-50 típico)
    random_state : Semilla para reproducibilidad
    """
    # Ajustar perplexity si hay pocas muestras
    effective_perplexity = min(perplexity, (len(df) - 1) / 3)

    tsne = TSNE(
        n_components=n_components,
        perplexity=effective_perplexity,
        random_state=random_state,
        n_iter=1000,
    )
    transformed = tsne.fit_transform(df)

    col_names = [f"tSNE{i+1}" for i in range(n_components)]
    components_df = pd.DataFrame(transformed, columns=col_names, index=df.index)

    return DimReductionResult(
        method="t-SNE",
        components=components_df,
        n_components_selected=n_components,
        feature_names=df.columns.tolist(),
        diagnostics={
            "perplexity": effective_perplexity,
            "kl_divergence": float(tsne.kl_divergence_),
            "n_iter": tsne.n_iter_,
            "n_features": df.shape[1],
            "n_samples": df.shape[0],
        },
    )


def run_umap(
    df: pd.DataFrame,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> DimReductionResult:
    """
    UMAP para reducción dimensional no-lineal.
    Preserva mejor la estructura global que t-SNE.

    Parameters
    ----------
    df : DataFrame escalado
    n_components : Dimensiones del embedding
    n_neighbors : Tamaño del vecindario local (5-50)
    min_dist : Distancia mínima en el embedding (0.0-1.0)
    """
    try:
        import umap
    except ImportError:
        raise ImportError(
            "umap-learn no está instalado. Instálalo con: pip install umap-learn"
        )

    n_neighbors = min(n_neighbors, len(df) - 1)

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    transformed = reducer.fit_transform(df)

    col_names = [f"UMAP{i+1}" for i in range(n_components)]
    components_df = pd.DataFrame(transformed, columns=col_names, index=df.index)

    return DimReductionResult(
        method="UMAP",
        components=components_df,
        n_components_selected=n_components,
        feature_names=df.columns.tolist(),
        diagnostics={
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "n_features": df.shape[1],
            "n_samples": df.shape[0],
        },
    )


def reduce(
    df: pd.DataFrame,
    method: Literal["pca", "tsne", "umap", "auto"] = "auto",
    n_components: Optional[int] = None,
    **kwargs,
) -> DimReductionResult:
    """
    Interfaz unificada para reducción dimensional.

    Si method='auto':
    - Usa PCA si n_features < 15 o n_samples < 50
    - Usa UMAP si disponible, sino t-SNE

    Parameters
    ----------
    df : DataFrame escalado con variables numéricas
    method : 'pca', 'tsne', 'umap', o 'auto'
    n_components : Número de componentes (None = automático)
    **kwargs : Parámetros adicionales para el método específico
    """
    if method == "auto":
        n_feat, n_samp = df.shape[1], df.shape[0]
        if n_feat < 15 or n_samp < 50:
            method = "pca"
        else:
            try:
                import umap  # noqa: F401
                method = "umap"
            except ImportError:
                method = "tsne"

    if method == "pca":
        return run_pca(df, n_components=n_components, **kwargs)
    elif method == "tsne":
        return run_tsne(df, n_components=n_components or 2, **kwargs)
    elif method == "umap":
        return run_umap(df, n_components=n_components or 2, **kwargs)
    else:
        raise ValueError(f"Método no soportado: {method}")
