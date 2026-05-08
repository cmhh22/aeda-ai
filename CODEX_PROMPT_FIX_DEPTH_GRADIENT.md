# CODEX_PROMPT_FIX_DEPTH_GRADIENT

**Tipo:** Bug fix (defensive coding)
**Archivos:** 1 modificado + 1 nuevo
**Tiempo estimado:** 10–15 min
**Tests esperados después:** 33 actuales + 5 nuevos = **38 tests**

---

## 1. Contexto del problema

La función `_detect_depth_gradient` en `aeda/engine/auto_selector.py` puede
crashear con un `ValueError` cuando se ejecuta el pipeline sobre datasets
del mundo real. El error es:

```
ValueError: The truth value of an array with more than one element is ambiguous.
Use a.any() or a.all()
```

Y el traceback termina en la línea actual:

```python
r, p = stats.spearmanr(valid[depth_col], valid[col])
if p < 0.05 and abs(r) > 0.3:   # <-- crashes here
```

### Causa raíz

`scipy.stats.spearmanr` devuelve **escalares** cuando recibe dos `pd.Series`
1-D, pero devuelve **matrices NxN** cuando recibe un `pd.DataFrame` (es decir,
varias columnas a la vez).

En el dataset ISOVIDA se reproduce el crash cuando:

- El Excel tiene **columnas con nombres duplicados** en alguna hoja (sucede
  cuando se carga sin `sheet_name="DATA"` explícito y se toma una hoja
  auxiliar). En ese caso `df["Profundidad"]` devuelve un `DataFrame` y no
  una `Series`.
- La función pasa ese DataFrame a `spearmanr`, que devuelve matrices `(N×N)`
  para `r` y `p`.
- La condición `p < 0.05 and abs(r) > 0.3` aplica `<` y `abs` sobre arrays
  multi-elemento → `ValueError`.

Tres escenarios secundarios que también pueden crashear o devolver NaN:

1. Columnas con varianza cero (no se puede correlacionar).
2. Columnas con muchos NaN (después de `dropna` quedan menos de 10 filas).
3. La columna de profundidad coincide con una columna de medición (auto-correlación trivial).

### Verificación previa

El bug se reproduce con este snippet:

```python
import pandas as pd, numpy as np
from scipy import stats
df = pd.DataFrame(np.random.randn(50, 3), columns=['Profundidad', 'Pb', 'Profundidad'])
r, p = stats.spearmanr(df['Profundidad'], df['Pb'])
# r y p son matrices (3, 3) → la siguiente línea crashea:
if p < 0.05 and abs(r) > 0.3: pass
```

Y se reproduce end-to-end con:

```python
from aeda.pipeline.runner import AEDAPipeline
AEDAPipeline().run('data/BD_ISOVIDA_MANGLARES2023_rectificadaYBA_230326.xlsx')
# (sin sheet_name → toma la primera hoja → falla)
```

---

## 2. Cambio en `aeda/engine/auto_selector.py`

Reemplazar **completamente** la función `_detect_depth_gradient` (actualmente
en torno a la línea 407–421) por la siguiente versión defensiva. La firma
y la semántica del retorno se mantienen idénticas — solo cambia la robustez
interna.

```python
def _detect_depth_gradient(df: pd.DataFrame, depth_col: str, measurement_cols: list[str],
                           threshold_pct: float = 0.3) -> bool:
    """Return True if a sufficient fraction of measurement columns show a
    significant monotonic relationship with depth.

    A column is counted when |Spearman r| > 0.3 and p < 0.05. The function
    returns True when more than ``threshold_pct`` of the measurement columns
    meet that criterion.

    Defensive against three real-world failure modes observed on the ISOVIDA
    dataset:

    * Duplicated column names: ``df[col]`` returns a DataFrame instead of a
      Series, which propagates into ``spearmanr`` and makes it return matrices
      instead of scalars. We collapse to the first column in that case.
    * Zero-variance columns: Spearman correlation is undefined and SciPy may
      emit a NaN or warn. We skip those columns explicitly.
    * NaN propagation: rows containing NaN in either column are dropped
      pairwise before the correlation is computed.
    """
    if depth_col not in df.columns:
        return False

    depth_series = df[depth_col]
    if isinstance(depth_series, pd.DataFrame):
        depth_series = depth_series.iloc[:, 0]

    n_significant = 0
    for col in measurement_cols:
        if col not in df.columns or col == depth_col:
            continue

        col_series = df[col]
        if isinstance(col_series, pd.DataFrame):
            col_series = col_series.iloc[:, 0]

        paired = pd.concat([depth_series, col_series], axis=1).dropna()
        if len(paired) < 10:
            continue

        x = np.asarray(paired.iloc[:, 0], dtype=float).ravel()
        y = np.asarray(paired.iloc[:, 1], dtype=float).ravel()

        if np.std(x) == 0 or np.std(y) == 0:
            continue

        try:
            result = stats.spearmanr(x, y)
            r = float(result.correlation)
            p = float(result.pvalue)
        except (ValueError, TypeError):
            continue

        if np.isnan(r) or np.isnan(p):
            continue
        if p < 0.05 and abs(r) > 0.3:
            n_significant += 1

    return n_significant / max(len(measurement_cols), 1) > threshold_pct
```

### Notas para el agente

- **No tocar nada más** del archivo `auto_selector.py`. El resto está validado
  por la suite de tests actual y se debe preservar.
- Las importaciones de `pd`, `np` y `stats` ya están presentes en el módulo —
  no agregar imports nuevos.
- La numeración de líneas puede haber cambiado ligeramente; usar un buscador
  por nombre de función (`_detect_depth_gradient`) para localizar el bloque a
  reemplazar.

---

## 3. Test nuevo a crear

Crear un archivo `tests/test_depth_gradient.py` con el siguiente contenido
exacto. Cubre los 5 escenarios que motivaron el fix (1 happy-path + 4 edge
cases que antes crasheaban o devolvían resultados ambiguos).

```python
"""Regression tests for _detect_depth_gradient.

Covers the failure modes that crashed the function on the real ISOVIDA dataset:
- Duplicate column names in the source DataFrame.
- Zero-variance measurement columns.
- NaN values in measurement columns.
- Missing depth column.
- Happy path with a real depth gradient.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aeda.engine.auto_selector import _detect_depth_gradient


def test_detect_depth_gradient_with_duplicate_column_names():
    """Duplicate column names used to make spearmanr return matrices and crash."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        rng.standard_normal((50, 3)),
        columns=["Profundidad", "Pb", "Profundidad"],
    )
    # Must not raise; with random data the gradient should not be flagged.
    result = _detect_depth_gradient(df, "Profundidad", ["Pb"])
    assert result is False


def test_detect_depth_gradient_with_zero_variance_column():
    """Zero-variance columns must be skipped, not crash."""
    depth = np.linspace(0, 10, 50)
    df = pd.DataFrame({
        "Profundidad": depth,
        "ConstantMetal": np.zeros(50),       # zero variance
        "RealMetal": depth * 1.5 + 0.1,      # genuine gradient
    })
    # With 1/2 columns showing a gradient (ConstantMetal skipped, RealMetal counted)
    # and threshold_pct=0.3, the function should still flag the gradient.
    result = _detect_depth_gradient(df, "Profundidad", ["ConstantMetal", "RealMetal"])
    assert result is True


def test_detect_depth_gradient_handles_nans():
    """Pairwise NaN rows must be dropped, not propagated into the correlation."""
    depth = np.linspace(0, 20, 50)
    df = pd.DataFrame({"Profundidad": depth, "Pb": depth * 2.0})
    df.loc[::3, "Pb"] = np.nan  # introduce ~33% NaN
    result = _detect_depth_gradient(df, "Profundidad", ["Pb"])
    assert result is True


def test_detect_depth_gradient_returns_false_when_depth_missing():
    """Missing depth column must return False, not raise."""
    df = pd.DataFrame({"Pb": np.linspace(0, 10, 50), "Zn": np.linspace(0, 5, 50)})
    assert _detect_depth_gradient(df, "Profundidad", ["Pb", "Zn"]) is False


def test_detect_depth_gradient_happy_path():
    """When most columns correlate with depth, the function must flag the gradient."""
    rng = np.random.default_rng(42)
    depth = np.linspace(0, 20, 80)
    df = pd.DataFrame({
        "Profundidad": depth,
        "Pb": depth * 2.0 + rng.standard_normal(80) * 0.5,
        "Zn": -1.5 * depth + rng.standard_normal(80) * 0.5,
        "Cu": depth * 0.8 + rng.standard_normal(80) * 0.3,
    })
    assert _detect_depth_gradient(df, "Profundidad", ["Pb", "Zn", "Cu"]) is True
```

---

## 4. Cómo correr y validar

Ejecutar exactamente estos comandos en el root del proyecto, en orden:

```bash
# 1. Ejecutar solo los tests nuevos para verificar que cubren el fix
pytest tests/test_depth_gradient.py -v

# 2. Ejecutar la suite completa para verificar que no se rompió nada
pytest tests/ -v
```

### Salida esperada

**Comando 1:**

```
tests/test_depth_gradient.py::test_detect_depth_gradient_with_duplicate_column_names PASSED
tests/test_depth_gradient.py::test_detect_depth_gradient_with_zero_variance_column PASSED
tests/test_depth_gradient.py::test_detect_depth_gradient_handles_nans PASSED
tests/test_depth_gradient.py::test_detect_depth_gradient_returns_false_when_depth_missing PASSED
tests/test_depth_gradient.py::test_detect_depth_gradient_happy_path PASSED

5 passed
```

**Comando 2:**

```
================================ 38 passed in ~25s ================================
```

(33 tests previos + 5 nuevos = 38 total)

### Validación end-to-end opcional

Para confirmar que el bug se resolvió a nivel pipeline completo, ejecutar:

```python
python -c "
from aeda.pipeline.runner import AEDAPipeline
r = AEDAPipeline().run('data/BD_ISOVIDA_MANGLARES2023_rectificadaYBA_230326.xlsx')
print('OK — plan generado:', r.plan is not None)
"
```

Esperado: `OK — plan generado: True`. Antes del fix esto crasheaba con
`ValueError: truth value of an array...`.

---

## 5. Si algo falla

- Si los 5 tests nuevos no pasan, **detenerse** y reportar el output completo.
  No intentar parchearlos — el código del fix está validado.
- Si los 33 tests existentes empiezan a fallar después del cambio, **detenerse**
  y reportar — significa que algo más se modificó por error.
- No tocar `tests/test_pipeline_interpretation.py`, `test_integration.py` ni
  ningún otro archivo de tests existente.

---

## 6. Mensaje de commit sugerido

```
fix(auto_selector): defensive _detect_depth_gradient against duplicate columns

The function used to crash with "truth value of an array is ambiguous" when
the input DataFrame had duplicate column names — a real failure mode on the
ISOVIDA Excel file when loaded without explicit sheet_name. In that case
df[col] returns a DataFrame and spearmanr returns NxN matrices instead of
scalars, breaking the comparison `p < 0.05 and abs(r) > 0.3`.

Three defensive guards added:
- Collapse DataFrame slices to a single Series when names are duplicated.
- Skip zero-variance columns (Spearman undefined).
- Drop NaN pairwise and skip NaN results from spearmanr.

Adds 5 regression tests in tests/test_depth_gradient.py covering all five
edge cases: duplicate columns, zero variance, NaN handling, missing depth
column, and the happy path with a real gradient.
```
