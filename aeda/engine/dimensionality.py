"""
aeda.engine.dimensionality
Dimensionality reduction module for multivariate environmental data.
PCA, t-SNE, and UMAP with automatic selection and diagnostics.
"""

import numpy as np
import pandas as pd
from typing import Optional, Literal
from dataclasses import dataclass, field
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


@dataclass
class DimReductionResult:
    """Result of a dimensionality reduction run."""
    method: str
    components: pd.DataFrame  # transformed coordinates
    explained_variance: Optional[np.ndarray] = None  # PCA only
    cumulative_variance: Optional[np.ndarray] = None  # PCA only
    n_components_selected: int = 0
    loadings: Optional[pd.DataFrame] = None  # PCA only: contribution of each variable
    feature_names: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


def run_pca(
    df: pd.DataFrame,
    n_components: Optional[int] = None,
    variance_threshold: float = 0.85,
) -> DimReductionResult:
    """
    Run PCA with automatic component selection.

    If n_components is None, it selects the minimum number of components
    that explain at least variance_threshold of total variance.

    Parameters
    ----------
    df : pd.DataFrame
        Scaled DataFrame with numeric variables.
    n_components : int, optional
        Number of components. None enables auto-selection.
    variance_threshold : float
        Minimum cumulative explained variance for auto-selection (0-1).

    Returns
    -------
    DimReductionResult
        Result with components, explained variance, and loadings.
    """
    # Fit full PCA first to determine n_components
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

    # Loadings: contribution of each original variable to each PC
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=col_names,
        index=df.columns,
    )

    # Kaiser criterion: components with eigenvalue > 1
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
    Run t-SNE for nonlinear cluster visualization.

    Parameters
    ----------
    df : pd.DataFrame
        Scaled DataFrame.
    n_components : int
        Number of output dimensions (typically 2 or 3).
    perplexity : float
        Controls local/global balance (typical range: 5-50).
    random_state : int
        Seed for reproducibility.
    """
    # Adjust perplexity for small sample sizes
    effective_perplexity = min(perplexity, (len(df) - 1) / 3)

    tsne = TSNE(
        n_components=n_components,
        perplexity=effective_perplexity,
        random_state=random_state,
        max_iter=1000,
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
            "n_iter": getattr(tsne, "n_iter_", getattr(tsne, "n_iter", None)),
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
    Run UMAP for nonlinear dimensionality reduction.
    Typically preserves global structure better than t-SNE.

    Parameters
    ----------
    df : pd.DataFrame
        Scaled DataFrame.
    n_components : int
        Embedding dimensions.
    n_neighbors : int
        Local neighborhood size (5-50).
    min_dist : float
        Minimum distance in the embedding (0.0-1.0).
    """
    try:
        import umap
    except ImportError:
        raise ImportError(
            "umap-learn is not installed. Install it with: pip install umap-learn"
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
    Unified interface for dimensionality reduction.

    If method='auto', PCA is always used. PCA is the canonical EDA method:
    it explains variance, produces interpretable loadings for biplots, and
    works reliably on typical environmental datasets. For nonlinear
    visualizations, pass method='tsne' or method='umap' explicitly.

    Parameters
    ----------
    df : pd.DataFrame
        Scaled DataFrame with numeric variables.
    method : Literal["pca", "tsne", "umap", "auto"]
        Dimensionality reduction method. 'auto' resolves to 'pca'.
    n_components : int, optional
        Number of components (None = automatic behavior).
    **kwargs
        Extra parameters for the selected method.
    """
    if method == "auto":
        # For environmental EDA, PCA is the canonical first choice:
        # it provides interpretable loadings for biplots, explains variance,
        # and works reliably on typical dataset sizes.
        method = "pca"

    if method == "pca":
        return run_pca(df, n_components=n_components, **kwargs)
    elif method == "tsne":
        return run_tsne(df, n_components=n_components or 2, **kwargs)
    elif method == "umap":
        return run_umap(df, n_components=n_components or 2, **kwargs)
    else:
        raise ValueError(f"Unsupported method: {method}")
