# BUG CRÍTICO 1 — Baseline personalizado por sitio se ignora

## Contexto

En `aeda/interpretation/normalization.py`, la función `compute_enrichment_factor` tiene una rama para `baseline_strategy="user"` con `custom_baseline`. Cuando el usuario provee un diccionario organizado por sitio (ej: `{"SiteA": {...}, "SiteB": {...}}`), el código actual selecciona el primer sitio para todas las muestras, ignorando el sitio real de cada fila.

## Diagnóstico verificado

Reproducción:

```python
df = pd.DataFrame({
    "site": ["A", "A", "B", "B", "C", "C"],
    "depth": [10, 50, 10, 50, 10, 50],
    "Al": [5.0, 6.0, 4.0, 5.0, 6.0, 7.0],
    "Pb": [30, 20, 50, 25, 100, 30],
})
result = compute_enrichment_factor(
    df, metals=["Pb"], reference_element="Al",
    site_col="site", depth_col="depth",
    baseline_strategy="user",
    custom_baseline={
        "A": {"Al": 5.5, "Pb": 25},
        "B": {"Al": 4.5, "Pb": 30},
        "C": {"Al": 6.5, "Pb": 28},
    }
)
# Bug: todos los EF se calculan contra el baseline de A.
```

Causa raíz: en la línea 124, el código hace `base = baseline_concs.get("__global__", next(iter(baseline_concs.values())))` para todas las ramas que no son `"deepest"`, sin distinguir si `custom_baseline` está organizado por sitio o globalmente.

## Cambios requeridos

### 1. Modificar `aeda/interpretation/normalization.py`

En la función `compute_enrichment_factor`, modificar la lógica de selección del baseline para que la rama `"user"` también soporte baselines por sitio cuando `site_col` está provisto y las claves del `custom_baseline` coinciden con valores únicos del `site_col`.

**Lógica de detección de formato del custom_baseline:**

- Si `custom_baseline` contiene la clave `"__global__"` → es global, una para todas las muestras.
- Si las claves de `custom_baseline` coinciden con valores de `df[site_col].unique()` → es por sitio.
- Si `custom_baseline` parece tener estructura plana `{reference_element: ..., metal1: ..., metal2: ...}` (las claves son nombres de columnas, no nombres de sitios) → tratarla como global.

**Reemplazo del bloque actual (líneas ~101-107):**

```python
    elif baseline_strategy == "user":
        baseline_concs = custom_baseline
```

debe transformarse en una validación más estricta:

```python
    elif baseline_strategy == "user":
        baseline_concs = _validate_user_baseline(
            custom_baseline,
            metals=metals,
            reference_element=reference_element,
            df=df,
            site_col=site_col,
        )
```

**Reemplazo del bloque de selección por fila (líneas ~117-124):**

El bloque actual es:

```python
        if site_col is not None and baseline_strategy == "deepest":
            site = row[site_col]
            if site not in baseline_concs:
                ef_df.loc[idx, :] = np.nan
                continue
            base = baseline_concs[site]
        else:
            base = baseline_concs.get("__global__", next(iter(baseline_concs.values())))
```

Debe transformarse en:

```python
        # Determine which baseline applies to this row.
        if site_col is not None and baseline_strategy in ("deepest", "user") and "__global__" not in baseline_concs:
            site = row[site_col]
            if site not in baseline_concs:
                ef_df.loc[idx, :] = np.nan
                continue
            base = baseline_concs[site]
        else:
            base = baseline_concs.get("__global__", next(iter(baseline_concs.values())))
```

**Helper a agregar (al final del archivo, antes del retorno):**

```python
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
```

### 2. Agregar tests de regresión en `tests/test_interpretation.py`

Agregar al final del archivo:

```python
def test_ef_custom_baseline_per_site_uses_correct_site():
    """Regression: per-site custom_baseline must apply the right baseline per row."""
    df = pd.DataFrame({
        "site": ["A", "A", "B", "B", "C", "C"],
        "depth": [10, 50, 10, 50, 10, 50],
        "Al": [5.0, 6.0, 4.0, 5.0, 6.0, 7.0],
        "Pb": [30, 20, 50, 25, 100, 30],
    })
    custom = {
        "A": {"Al": 5.5, "Pb": 25.0},
        "B": {"Al": 4.5, "Pb": 30.0},
        "C": {"Al": 6.5, "Pb": 28.0},
    }
    result = compute_enrichment_factor(
        df, metals=["Pb"], reference_element="Al",
        site_col="site", depth_col="depth",
        baseline_strategy="user",
        custom_baseline=custom,
    )
    # Row 0 (site A, Al=5.0, Pb=30): EF = (30/5.0) / (25/5.5) = 6.0 / 4.5454... = 1.32
    # Row 4 (site C, Al=6.0, Pb=100): EF = (100/6.0) / (28/6.5) = 16.66... / 4.3076... = 3.87
    # If the bug is present, both rows would use baseline A and produce different (wrong) values.
    assert abs(result.ef_values.loc[0, "Pb"] - 1.32) < 0.01
    assert abs(result.ef_values.loc[4, "Pb"] - 3.87) < 0.01


def test_ef_custom_baseline_global_flat_dict_still_works():
    """Backwards compat: flat global custom_baseline must still work."""
    df = pd.DataFrame({
        "site": ["A", "A", "B", "B"],
        "depth": [10, 50, 10, 50],
        "Al": [5.0, 6.0, 4.0, 5.0],
        "Pb": [30, 20, 50, 25],
    })
    result = compute_enrichment_factor(
        df, metals=["Pb"], reference_element="Al",
        baseline_strategy="user",
        custom_baseline={"Al": 5.0, "Pb": 25.0},
    )
    assert result.ef_values["Pb"].notna().all()
    # Row 0: (30/5.0) / (25/5.0) = 6/5 = 1.2
    assert abs(result.ef_values.loc[0, "Pb"] - 1.2) < 0.01


def test_ef_custom_baseline_missing_site_raises():
    """custom_baseline must cover all sites in the dataset."""
    df = pd.DataFrame({
        "site": ["A", "B", "C"],
        "depth": [10, 10, 10],
        "Al": [5.0, 4.0, 6.0],
        "Pb": [30, 50, 100],
    })
    incomplete = {
        "A": {"Al": 5.0, "Pb": 25.0},
        "B": {"Al": 4.0, "Pb": 30.0},
        # missing "C"
    }
    with pytest.raises(ValueError, match="does not cover"):
        compute_enrichment_factor(
            df, metals=["Pb"], reference_element="Al",
            site_col="site", depth_col="depth",
            baseline_strategy="user",
            custom_baseline=incomplete,
        )


def test_ef_custom_baseline_missing_metal_raises():
    """Per-site baseline missing a required metal must raise."""
    df = pd.DataFrame({
        "site": ["A", "B"],
        "depth": [10, 10],
        "Al": [5.0, 4.0],
        "Pb": [30, 50],
    })
    bad = {
        "A": {"Al": 5.0},  # missing Pb
        "B": {"Al": 4.0, "Pb": 30.0},
    }
    with pytest.raises(ValueError, match="missing required keys"):
        compute_enrichment_factor(
            df, metals=["Pb"], reference_element="Al",
            site_col="site", depth_col="depth",
            baseline_strategy="user",
            custom_baseline=bad,
        )
```

## Verificación

Después de aplicar los cambios:

1. Ejecutar `pytest tests/ -v` y verificar que los 18 tests previos siguen pasando, más los 4 nuevos (22 en total).
2. Confirmar específicamente que los 4 nuevos tests pasan.
3. Confirmar que `test_ef_custom_baseline` (test existente con baseline global) sigue pasando.

## Commit

Mensaje sugerido:

```
fix(interpretation): per-site custom_baseline now applies the right baseline per row

Previously, when compute_enrichment_factor was called with baseline_strategy="user"
and a per-site custom_baseline, the code took only the first site's baseline and
applied it to every sample, silently ignoring the site organization.

This fix introduces _validate_user_baseline() which detects whether the user
provided a flat global dict or a per-site dict, validates structure and coverage,
and the main loop now correctly indexes per-site baselines when applicable.

Backwards compatible: flat {element: value} dicts continue to work as global baselines.

Adds 4 regression tests covering:
- per-site baselines apply the correct site's values
- flat global baselines still work
- incomplete site coverage raises ValueError
- missing required metal in a site entry raises ValueError
```
