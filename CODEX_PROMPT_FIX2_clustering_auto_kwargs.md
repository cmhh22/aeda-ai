# BUG CRÍTICO 2 — cluster(method='auto') falla cuando se pasa k_range

## Contexto

En `aeda/engine/clustering.py`, la función `cluster(df, method="auto", **kwargs)` corre tanto K-Means como DBSCAN y elige el de mejor silhouette. El problema es que cuando se pasan kwargs específicos de K-Means (como `k_range`), estos también se reenvían a `run_dbscan(df, **kwargs)`, que no los acepta y arroja `TypeError: run_dbscan() got an unexpected keyword argument 'k_range'`.

El auto-selector genera recomendaciones con `params={"method": "kmeans", "n_clusters": None, "k_range": (2, 10)}`, así que esta llamada es realista y aparecerá en flujos reales del usuario.

## Diagnóstico verificado

Reproducción:

```python
import pandas as pd, numpy as np
from aeda.engine.clustering import cluster

df = pd.DataFrame(np.random.randn(50, 3), columns=list("abc"))
cluster(df, method="auto", k_range=(2, 10))
# TypeError: run_dbscan() got an unexpected keyword argument 'k_range'
```

## Cambios requeridos

### 1. Modificar `aeda/engine/clustering.py`

Reemplazar la rama `method == "auto"` en la función `cluster` (líneas ~214-230) por una versión que separe kwargs por método.

**Implementación:**

```python
# Parameters that belong specifically to each clustering method
KMEANS_KWARGS = {"k_range"}
DBSCAN_KWARGS = {"eps", "min_samples"}
HIERARCHICAL_KWARGS = {"linkage"}


def _split_kwargs(kwargs: dict, allowed: set[str]) -> dict:
    """Return only the kwargs that are accepted by the target function."""
    return {k: v for k, v in kwargs.items() if k in allowed}


def cluster(
    df: pd.DataFrame,
    method: Literal["kmeans", "dbscan", "hierarchical", "auto"] = "auto",
    n_clusters: Optional[int] = None,
    **kwargs,
) -> ClusteringResult:
    """
    Unified interface for clustering.

    If method='auto', runs K-Means and DBSCAN and returns the one with
    the best silhouette score. Each method receives only its own kwargs.
    """
    if method == "auto":
        results = []
        # K-Means accepts n_clusters and k_range; DBSCAN accepts eps and min_samples.
        # In auto mode we filter kwargs so each function only receives its own.
        kmeans_kwargs = _split_kwargs(kwargs, KMEANS_KWARGS)
        dbscan_kwargs = _split_kwargs(kwargs, DBSCAN_KWARGS)

        results.append(run_kmeans(df, n_clusters=n_clusters, **kmeans_kwargs))
        results.append(run_dbscan(df, **dbscan_kwargs))

        # Select the best result by silhouette score
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
        return run_kmeans(df, n_clusters=n_clusters, **_split_kwargs(kwargs, KMEANS_KWARGS))
    elif method == "dbscan":
        return run_dbscan(df, **_split_kwargs(kwargs, DBSCAN_KWARGS))
    elif method == "hierarchical":
        return run_hierarchical(df, n_clusters=n_clusters, **_split_kwargs(kwargs, HIERARCHICAL_KWARGS))
    else:
        raise ValueError(f"Unsupported method: {method}")
```

**Importante:** las constantes `KMEANS_KWARGS`, `DBSCAN_KWARGS`, `HIERARCHICAL_KWARGS` y la función `_split_kwargs` deben colocarse al nivel del módulo, antes de la función `cluster` (no dentro de ella).

### 2. Agregar tests de regresión en `tests/test_integration.py`

Agregar al final del archivo:

```python
def test_cluster_auto_with_kmeans_kwargs():
    """Regression: cluster(method='auto') must not pass K-Means kwargs to DBSCAN."""
    import numpy as np
    import pandas as pd
    from aeda.engine.clustering import cluster

    rng = np.random.default_rng(42)
    df = pd.DataFrame(rng.normal(size=(50, 3)), columns=list("abc"))

    # k_range belongs to K-Means, not DBSCAN.
    # Before this fix, this call would raise TypeError.
    result = cluster(df, method="auto", k_range=(2, 8))

    assert result is not None
    assert result.diagnostics.get("auto_selected") is True
    assert "compared_methods" in result.diagnostics


def test_cluster_auto_with_dbscan_kwargs():
    """Regression: cluster(method='auto') must accept DBSCAN-specific kwargs without breaking K-Means."""
    import numpy as np
    import pandas as pd
    from aeda.engine.clustering import cluster

    rng = np.random.default_rng(42)
    df = pd.DataFrame(rng.normal(size=(50, 3)), columns=list("abc"))

    # min_samples belongs to DBSCAN, not K-Means.
    result = cluster(df, method="auto", min_samples=10)

    assert result is not None
    assert result.diagnostics.get("auto_selected") is True


def test_cluster_explicit_method_filters_kwargs():
    """Explicit methods must also tolerate kwargs that don't belong to them."""
    import numpy as np
    import pandas as pd
    from aeda.engine.clustering import cluster

    rng = np.random.default_rng(42)
    df = pd.DataFrame(rng.normal(size=(50, 3)), columns=list("abc"))

    # Pass DBSCAN's min_samples to explicit kmeans — must be silently ignored.
    result = cluster(df, method="kmeans", n_clusters=3, min_samples=10)
    assert result.method == "K-Means"
    assert result.n_clusters == 3
```

## Verificación

Después de aplicar los cambios:

1. Ejecutar `pytest tests/ -v` y verificar que los tests previos siguen pasando, más los 3 nuevos.
2. Probar manualmente desde el repositorio:

```python
import numpy as np, pandas as pd
from aeda.engine.clustering import cluster

df = pd.DataFrame(np.random.randn(100, 4), columns=list("abcd"))

# Caso 1: auto con kwargs de K-Means (antes fallaba)
r1 = cluster(df, method="auto", k_range=(2, 6))
print(r1.method, r1.n_clusters, r1.metrics.get("silhouette"))

# Caso 2: auto con kwargs de DBSCAN
r2 = cluster(df, method="auto", min_samples=5, eps=0.5)
print(r2.method, r2.n_clusters)

# Caso 3: kmeans explícito ignora kwargs ajenos
r3 = cluster(df, method="kmeans", n_clusters=3, min_samples=99)
print(r3.method, r3.n_clusters)
```

## Commit

Mensaje sugerido:

```
fix(clustering): cluster(method='auto') no longer leaks kwargs across methods

Previously, when cluster() ran in auto mode it passed **kwargs unchanged to both
run_kmeans and run_dbscan. K-Means-specific kwargs like k_range made run_dbscan
raise TypeError, breaking the whole auto path. The auto-selector emits exactly
this kind of recommendation, so this was triggered in normal usage.

This fix adds method-specific kwargs whitelists (KMEANS_KWARGS, DBSCAN_KWARGS,
HIERARCHICAL_KWARGS) and a small helper that filters incoming kwargs to only
those accepted by the target function. Auto mode and explicit method calls
both benefit from this.

Adds 3 regression tests covering:
- auto mode with K-Means-specific kwargs (k_range)
- auto mode with DBSCAN-specific kwargs (min_samples)
- explicit method tolerates ignored kwargs
```
