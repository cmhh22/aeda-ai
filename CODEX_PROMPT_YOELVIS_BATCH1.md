# TANDA 1 — Implementación del feedback de Yoelvis (9 cambios)

## Contexto

El tutor científico Yoelvis Bolaños-Alvarez revisó las 15 reglas del sistema experto y propuso ajustes. Esta tanda implementa **9 cambios simples e independientes**. Un décimo cambio (análisis espacial de fracción superficial 0-10 cm) va en una tanda posterior porque requiere desarrollo nuevo.

Antes de comenzar, ejecutar `pytest tests/ -v` para confirmar que los tests existentes pasan (debería haber 33 tests pasando si los FIX1-FIX4 anteriores se aplicaron).

Después de cada cambio, ejecutar la suite de tests. Al final, hacer un único commit con el mensaje sugerido al final de este documento.

---

## CAMBIO 1 — Trace elements: agregar los que sí están en ISOVIDA

### Archivo
`aeda/engine/auto_selector.py`

### Problema
Yoelvis revisó la lista de elementos traza y señaló que **faltan elementos que sí están en el estudio ISOVIDA**: Cr, Co, Ni, Cu, Zn, As, Mo, Pb. Estos elementos pueden pertenecer simultáneamente a TRACE_ELEMENTS y a HEAVY_METALS (un elemento puede ser traza y además ser un metal pesado regulado — son categorías ortogonales).

### Solución
En la constante `TRACE_ELEMENTS`, agregar los 8 elementos faltantes.

**Estado actual** (línea ~33):
```python
TRACE_ELEMENTS = {
    "V", "Cr", "Co", "Ni", "Cu", "Zn", "Ga", "As", "Br", "Rb",
    "Sr", "Y", "Zr", "Nb", "Mo", "Ba", "Pb", "Sc", "S", "Cl",
    "Cd", "Hg", "Ag", "Sb", "Se",
}
```

Si ya están todos (porque el FIX3 anterior los agregó), confirmar que están y avanzar al cambio siguiente. Si falta alguno de {Cr, Co, Ni, Cu, Zn, As, Mo, Pb, Cd, Hg, Ag, Sb, Se}, agregarlo.

### Test de regresión
En `tests/test_integration.py` agregar:

```python
def test_trace_elements_includes_isovida_metals():
    """Trace elements must include all metals present in ISOVIDA."""
    from aeda.engine.auto_selector import TRACE_ELEMENTS
    required = {"Cr", "Co", "Ni", "Cu", "Zn", "As", "Mo", "Pb"}
    missing = required - TRACE_ELEMENTS
    assert not missing, f"Missing trace elements in ISOVIDA: {missing}"
```

---

## CAMBIO 2 — Renombrar SEDIMENT_INDICATORS

### Archivos
`aeda/engine/auto_selector.py` (constante)
- Y todos los lugares donde se importe o use la constante.

### Problema
Yoelvis aclaró que TOC, PPI550, PPI950, HC, CaCO3 **no son "indicadores de calidad de sedimento"**: son **variables complementarias** o, en la literatura, *"ancillary data"*. Específicamente:
- PPI550 = indicador de materia orgánica (no es la MO en sí)
- PPI950 = indicador de carbonatos (no solo CaCO3, también Mg, etc.)

### Solución
Renombrar la constante `SEDIMENT_INDICATORS` a `ANCILLARY_VARIABLES`. Actualizar todas las referencias en el código y en los tests.

Agregar este docstring justo encima de la constante:

```python
# Ancillary variables (sometimes called "ancillary data" in the literature):
# These are NOT regulated quality indicators. They are complementary variables
# that contextualize the chemical analysis. For example:
#   - PPI550 (loss on ignition at 550°C) is an INDICATOR of organic matter (OM),
#     but is not OM itself. It is used to evaluate sediment composition and
#     to assess whether the transport of an element is influenced by OM,
#     which helps identify common sources.
#   - PPI950 (loss on ignition at 950°C) is an INDICATOR of carbonates,
#     not only CaCO3 — it can include Mg carbonates and others.
#   - TOC: total organic carbon, similar role to PPI550.
#   - HC: humidity content.
# Source: clarification from Dr. Yoelvis Bolaños-Alvarez (CEAC), 2026.
ANCILLARY_VARIABLES = {"TOC", "OM", "PPI550", "PPI950", "HC", "CaCO3"}
```

Asegurarse de actualizar:
- Cualquier `from aeda.engine.auto_selector import SEDIMENT_INDICATORS` en todo el repo.
- Cualquier referencia a `sediment_indicator_cols` o nombres similares en estructuras de datos (`DatasetProfile`, etc.).
- Renombrar también el atributo de perfil (si se llama `sediment_indicator_cols`) a `ancillary_cols`.
- Mensajes que el usuario ve (logs, recomendaciones) que digan "sediment indicators": cambiar a "ancillary variables".

### Test de regresión

```python
def test_ancillary_variables_renamed():
    """Sediment indicators were renamed to ancillary variables (Yoelvis feedback)."""
    from aeda.engine import auto_selector
    assert hasattr(auto_selector, "ANCILLARY_VARIABLES")
    # The old name should no longer exist (or at most as a deprecation alias)
    expected = {"TOC", "OM", "PPI550", "PPI950", "HC", "CaCO3"}
    assert auto_selector.ANCILLARY_VARIABLES == expected
```

---

## CAMBIO 3 — Eliminar regla 7 (detección de mayoritarios FRX por suma)

### Archivo
`aeda/engine/auto_selector.py`

### Problema
Yoelvis evaluó la regla y concluyó: *"No creo que este criterio tenga mucho sentido por ahora. Los mayoritarios pueden ser altos indistintamente, hay rangos para cada elemento, solo cuando uno pasa ese rango puede ser atípico"*.

La regla actual chequea si los 4 mayoritarios principales suman al total esperado con CV < 25% y, en caso afirmativo, marca el dataset como "FRX típico". Hay que eliminar esta regla.

### Solución
Localizar en `auto_selector.py` el bloque que implementa esta regla. Buscar referencias a "FRX típico" o "frx_typical" o similar. Eliminar:
1. La función o bloque que evalúa la regla.
2. El campo correspondiente en `DatasetProfile` (si existe `is_frx_typical` o similar).
3. Cualquier recomendación que dependa de ese flag.
4. Tests que referencien esa regla (si existen).

Si elimina el campo `is_frx_typical` de `DatasetProfile`, asegurarse de que no rompe nada en otros módulos (motor, runner, app). Usar grep para verificar.

### Test de regresión

```python
def test_frx_typical_rule_removed():
    """Rule 7 (FRX typical detection by sum) was removed per Yoelvis feedback."""
    from aeda.engine.auto_selector import profile_dataset
    import inspect
    src = inspect.getsource(profile_dataset)
    # The rule's typical keywords should not appear in the profiler anymore
    assert "frx_typical" not in src.lower(), "Rule 7 (FRX typical sum) should be removed"
```

---

## CAMBIO 4 — Umbral de correlación: 0.7 → 0.6

### Archivos
`aeda/engine/auto_selector.py` y/o `aeda/engine/correlations.py`

### Problema
Yoelvis prefiere un umbral más laxo: *"Creo que 0.6, eso daría un valor de fortaleza razonable, solo que lo define las pruebas de significancia estadística"*.

### Solución
Buscar todas las referencias a `0.7` que estén asociadas a umbrales de correlación para detección de bloques geoquímicos. Reemplazar por `0.6`.

Probablemente exista una constante similar a:
```python
CORRELATION_BLOCK_THRESHOLD = 0.7
```

Cambiar a:
```python
# Threshold for grouping variables into geochemical correlation blocks.
# Yoelvis (CEAC, 2026) recommended 0.6 over 0.7 as "reasonable strength,
# determined by statistical significance".
CORRELATION_BLOCK_THRESHOLD = 0.6
```

Si el valor está hardcoded en varios lugares, centralizarlo como constante.

### Test de regresión

```python
def test_correlation_block_threshold_is_06():
    """Per Yoelvis feedback, threshold lowered from 0.7 to 0.6."""
    from aeda.engine.auto_selector import CORRELATION_BLOCK_THRESHOLD
    assert CORRELATION_BLOCK_THRESHOLD == 0.6
```

---

## CAMBIO 5 — Detección de unidades: leer del diccionario + heurística como fallback

### Archivos
`aeda/io/parsers.py` (parseo del diccionario)
`aeda/engine/auto_selector.py` (detección de mixed units)

### Problema
Yoelvis aclaró que en ISOVIDA solo **Si, Al, Ca, K, Fe, Na, Mg están en %**, el resto en mg/kg. Esta información está en la **segunda hoja del Excel (Diccionario_Data)**. La heurística numérica actual (mediana de mayoritarios < 20 y mediana de traza > 20) es frágil y puede fallar. La solución es:
1. **Si el Excel tiene una hoja de diccionario con columna de unidades, usarla.**
2. **Si no, recurrir a la heurística numérica como fallback.**

### Solución

**Parte A — Extender el parser para leer la columna de unidades.**

En `aeda/io/parsers.py`, localizar la función que parsea el diccionario (probablemente `_parse_dictionary_sheet` o similar). Si ya parsea variables y descripciones, extenderlo para que también lea una columna llamada "Unidad", "Unit", "Units" o "unidades" (case-insensitive). El resultado debe ser un diccionario `{variable_name: unit_string}`.

Devolver este mapping junto con el resto de info del dataset. Si no está la hoja de diccionario o no tiene columna de unidades, devolver `{}`.

Agregar al dataclass `DatasetInfo` (si existe) un nuevo campo:
```python
units: dict[str, str] = field(default_factory=dict)
```

**Parte B — Usar el diccionario en la detección de mixed units.**

En `auto_selector.py`, donde se hace la detección de unidades mixtas:

```python
def detect_mixed_units(df, major_cols, trace_cols, units_dict=None):
    """Detect if the dataset mixes percentage and mg/kg units.

    Strategy:
    1. If units_dict is provided and non-empty, use it as the authoritative source.
       Variables whose unit string contains '%' are considered percentages;
       those containing 'mg/kg' or 'ppm' are mg/kg.
    2. Otherwise, fall back to the numeric heuristic:
       - if median(major_cols) < 20 and median(trace_cols) > 20, assume mixed units.
    """
    if units_dict:
        major_in_pct = any(
            "%" in units_dict.get(c, "") for c in major_cols if c in units_dict
        )
        trace_in_mgkg = any(
            ("mg/kg" in units_dict.get(c, "").lower()
             or "ppm" in units_dict.get(c, "").lower())
            for c in trace_cols if c in units_dict
        )
        if major_in_pct and trace_in_mgkg:
            return {
                "mixed_units_detected": True,
                "method": "dictionary",
                "evidence": f"Major elements in %, trace in mg/kg per dataset dictionary"
            }
        # If dictionary exists but units don't suggest mixed, still return based on dict
        return {
            "mixed_units_detected": False,
            "method": "dictionary",
            "evidence": "Dictionary did not indicate mixed units"
        }

    # Fallback to numeric heuristic
    return _detect_mixed_units_numeric(df, major_cols, trace_cols)
```

Renombrar la función heurística existente a `_detect_mixed_units_numeric` para distinguirla.

Asegurarse de que el `runner.py` pase el diccionario de unidades al perfilador del cerebro.

### Test de regresión

```python
def test_mixed_units_from_dictionary_preferred_over_heuristic():
    """When a units dictionary is provided, it takes precedence over the heuristic."""
    import pandas as pd
    from aeda.engine.auto_selector import detect_mixed_units

    # Build a tiny dataset where the heuristic would say "not mixed" but the dict
    # clearly indicates mixed units
    df = pd.DataFrame({
        "Al": [5.0, 6.0, 7.0],
        "Pb": [10.0, 12.0, 15.0],  # numerically similar to Al — heuristic might miss
    })
    units = {"Al": "%", "Pb": "mg/kg"}
    result = detect_mixed_units(df, major_cols=["Al"], trace_cols=["Pb"], units_dict=units)
    assert result["mixed_units_detected"] is True
    assert result["method"] == "dictionary"


def test_mixed_units_falls_back_to_heuristic_when_no_dictionary():
    """When no units dictionary is available, fall back to numeric heuristic."""
    import pandas as pd
    from aeda.engine.auto_selector import detect_mixed_units

    df = pd.DataFrame({
        "Al": [5.0, 6.0, 7.0],     # median ~6, would be %
        "Pb": [40.0, 50.0, 60.0],  # median ~50, would be mg/kg
    })
    result = detect_mixed_units(df, major_cols=["Al"], trace_cols=["Pb"], units_dict=None)
    assert result["method"] == "heuristic"


def test_isovida_units_loaded_from_dictionary():
    """Integration: ISOVIDA dataset dictionary specifies units for each variable."""
    from aeda.pipeline.runner import AEDAPipeline
    EXCLUDE = ["No", "Code", "Site_Name", "Pret_Code", "Código_muestra",
               "Sitio_muestreo", "Fecha_muestreo", "Core",
               "Latitud", "Longitud", "Profundidad"]
    p = AEDAPipeline(impute_strategy="median")
    r = p.run("data/BD_ISOVIDA_MANGLARES2023_rectificadaYBA_230326.xlsx",
              exclude_cols=EXCLUDE, sheet_name="DATA")
    # The dataset info should expose units
    assert hasattr(r.dataset_info, "units")
    units = r.dataset_info.units
    if units:  # only if the dictionary sheet was found and parsed
        # At least one major element should be in %
        major_in_pct = any("%" in units.get(c, "") for c in ["Al", "Fe", "Si", "Ca"] if c in units)
        assert major_in_pct, "Expected at least one major element to be in % per ISOVIDA dictionary"
```

---

## CAMBIO 6 — CLR manual, no automático

### Archivos
`aeda/io/preprocessor.py` (lógica)
`aeda/engine/auto_selector.py` (recomendaciones)
`aeda/pipeline/runner.py` (default del pipeline)

### Problema
Yoelvis: *"Dejalo manual"*. El sistema NO debe aplicar CLR automáticamente cuando detecta composicionalidad. Solo debe recomendarlo y, eventualmente, dejar que el usuario lo active.

### Solución

**Parte A — Cambiar default en `AEDAPipeline.__init__`:**
```python
# antes:
apply_clr: bool | str | None = False
# o si estaba como "auto":
apply_clr: bool | str | None = "auto"
```

debe quedar:
```python
# CLR is now opt-in (user must explicitly enable it). Per Yoelvis feedback,
# automatic CLR application was removed to avoid silent transformations.
apply_clr: bool = False
```

Eliminar la rama del runner que activa CLR automáticamente al detectar composicionalidad. Si CLR está en False, no aplicar nada. Si está en True, aplicar a las columnas composicionales detectadas.

**Parte B — Ajustar el auto-selector.**

La regla del cerebro que detecta composicionalidad debe **seguir detectándola** y **seguir recomendándola** en el plan, pero ya no debe disparar la transformación automática. La diferencia: la recomendación queda en `plan.recommendations` con prioridad alta y mensaje claro, pero el runner no la ejecuta a menos que el usuario haya pasado `apply_clr=True`.

En la recomendación CLR del auto-selector, ajustar el mensaje:

```python
recommendation = MethodRecommendation(
    step="preprocessing",
    method="clr_transform",
    priority=Priority.HIGH,
    rationale=(
        "Compositional data detected (sum ≈ 100%, CV < 10%). "
        "CLR transformation is recommended before multivariate analysis. "
        "Per scientific tutor decision, this transformation must be "
        "explicitly enabled by the user (apply_clr=True). "
        "It will NOT be applied automatically."
    ),
    params={"columns": compositional_cols, "user_action_required": True},
)
```

### Test de regresión

```python
def test_clr_not_applied_automatically():
    """Per Yoelvis feedback, CLR must not be applied unless the user explicitly enables it."""
    import pandas as pd
    import numpy as np
    from aeda.pipeline.runner import AEDAPipeline

    # Build a small dataset with a clear compositional triple summing to 100
    rng = np.random.default_rng(42)
    n = 50
    a = rng.uniform(20, 40, n)
    b = rng.uniform(20, 40, n)
    c = 100 - a - b
    other = rng.normal(50, 10, n)
    df = pd.DataFrame({"clay": a, "silt": b, "sand": c, "other": other})

    p = AEDAPipeline(impute_strategy="median", apply_clr=False, run_interpretation=False)
    # Save to a temp file (the pipeline reads files)
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        path = f.name
    try:
        r = p.run(path)
        # Check that processed data still has the original sum ~ 100
        proc = r.processed_data[["clay", "silt", "sand"]].sum(axis=1)
        # If CLR had been applied, the sum would not be ~100 anymore
        assert (proc > 90).all() and (proc < 110).all(), \
            "CLR should NOT have been applied automatically"
    finally:
        os.unlink(path)


def test_clr_applied_when_explicitly_enabled():
    """When apply_clr=True, the transformation IS applied."""
    import pandas as pd
    import numpy as np
    from aeda.pipeline.runner import AEDAPipeline
    import tempfile, os

    rng = np.random.default_rng(42)
    n = 50
    a = rng.uniform(20, 40, n); b = rng.uniform(20, 40, n)
    df = pd.DataFrame({"clay": a, "silt": b, "sand": 100 - a - b, "other": rng.normal(50, 10, n)})
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        path = f.name
    try:
        p = AEDAPipeline(impute_strategy="median", apply_clr=True, run_interpretation=False)
        r = p.run(path)
        # After CLR, the values are log-ratios; original sum constraint is broken
        proc = r.processed_data[["clay", "silt", "sand"]].sum(axis=1)
        assert not ((proc > 90).all() and (proc < 110).all()), \
            "CLR should have been applied"
    finally:
        os.unlink(path)
```

---

## CAMBIO 7 — Crear tabla Rudnick & Gao 2013 (corteza terrestre)

### Archivo nuevo
`aeda/interpretation/crust_reference.py`

### Problema
Yoelvis recomendó agregar la referencia de Rudnick & Gao (2013) como tabla complementaria de "valores generales de la corteza terrestre". Esto sirve como referencia auxiliar para comparar concentraciones medidas con valores típicos de la corteza, sin sustituir a TEL/PEL.

### Solución
Crear el módulo nuevo `aeda/interpretation/crust_reference.py`:

```python
"""Upper continental crust reference values from Rudnick and Gao (2013).

These values represent typical concentrations of major and trace elements
in the upper continental crust. They are used as a complementary reference
to assess whether observed concentrations are within typical natural
background ranges. They are NOT a quality threshold like TEL or PEL;
they only describe what is geochemically common in continental crust.

Reference
---------
Rudnick, R.L. and Gao, S. (2013) "Composition of the Continental Crust",
in Holland, H.D. and Turekian, K.K. (eds.) Treatise on Geochemistry,
2nd ed., vol. 4. Oxford: Elsevier, pp. 1-51.
https://doi.org/10.1016/B978-0-08-095975-7.00301-6

All values in mg/kg (ppm) for trace elements and weight percent (wt%) for majors.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class CrustReferenceValue:
    """A single element's reference value in the upper continental crust."""
    value: float
    unit: str  # either "wt%" or "mg/kg"
    notes: str = ""


# Values from Rudnick & Gao (2013), Table 3 — Upper Continental Crust composition.
UPPER_CONTINENTAL_CRUST: dict[str, CrustReferenceValue] = {
    # Major elements (wt%)
    "Si": CrustReferenceValue(value=31.10, unit="wt%", notes="SiO2 66.62 wt%"),
    "Al": CrustReferenceValue(value=8.15,  unit="wt%", notes="Al2O3 15.40 wt%"),
    "Fe": CrustReferenceValue(value=3.92,  unit="wt%", notes="Fe2O3T 5.04 wt%"),
    "Ca": CrustReferenceValue(value=2.57,  unit="wt%", notes="CaO 3.59 wt%"),
    "Na": CrustReferenceValue(value=2.43,  unit="wt%", notes="Na2O 3.27 wt%"),
    "K":  CrustReferenceValue(value=2.32,  unit="wt%", notes="K2O 2.80 wt%"),
    "Mg": CrustReferenceValue(value=1.50,  unit="wt%", notes="MgO 2.48 wt%"),
    "Ti": CrustReferenceValue(value=0.38,  unit="wt%", notes="TiO2 0.64 wt%"),
    "P":  CrustReferenceValue(value=0.066, unit="wt%", notes="P2O5 0.15 wt%"),
    "Mn": CrustReferenceValue(value=0.0775, unit="wt%", notes="MnO 0.10 wt%"),
    # Trace elements (mg/kg)
    "As": CrustReferenceValue(value=4.8,   unit="mg/kg"),
    "Ba": CrustReferenceValue(value=628.0, unit="mg/kg"),
    "Cd": CrustReferenceValue(value=0.09,  unit="mg/kg"),
    "Co": CrustReferenceValue(value=17.3,  unit="mg/kg"),
    "Cr": CrustReferenceValue(value=92.0,  unit="mg/kg"),
    "Cs": CrustReferenceValue(value=4.9,   unit="mg/kg"),
    "Cu": CrustReferenceValue(value=28.0,  unit="mg/kg"),
    "Ga": CrustReferenceValue(value=17.5,  unit="mg/kg"),
    "Hg": CrustReferenceValue(value=0.05,  unit="mg/kg"),
    "Li": CrustReferenceValue(value=24.0,  unit="mg/kg"),
    "Mo": CrustReferenceValue(value=1.1,   unit="mg/kg"),
    "Nb": CrustReferenceValue(value=12.0,  unit="mg/kg"),
    "Ni": CrustReferenceValue(value=47.0,  unit="mg/kg"),
    "Pb": CrustReferenceValue(value=17.0,  unit="mg/kg"),
    "Rb": CrustReferenceValue(value=84.0,  unit="mg/kg"),
    "Sb": CrustReferenceValue(value=0.4,   unit="mg/kg"),
    "Sc": CrustReferenceValue(value=14.0,  unit="mg/kg"),
    "Sn": CrustReferenceValue(value=2.1,   unit="mg/kg"),
    "Sr": CrustReferenceValue(value=320.0, unit="mg/kg"),
    "Th": CrustReferenceValue(value=10.5,  unit="mg/kg"),
    "U":  CrustReferenceValue(value=2.7,   unit="mg/kg"),
    "V":  CrustReferenceValue(value=97.0,  unit="mg/kg"),
    "Y":  CrustReferenceValue(value=21.0,  unit="mg/kg"),
    "Zn": CrustReferenceValue(value=67.0,  unit="mg/kg"),
    "Zr": CrustReferenceValue(value=193.0, unit="mg/kg"),
}


def get_crust_reference(element: str) -> CrustReferenceValue:
    """Retrieve the upper continental crust reference value for an element.

    Parameters
    ----------
    element : str
        Element symbol (e.g., 'Pb', 'Hg'). Case-sensitive.

    Returns
    -------
    CrustReferenceValue

    Raises
    ------
    KeyError
        If the element is not in the Rudnick & Gao (2013) table.
    """
    if element not in UPPER_CONTINENTAL_CRUST:
        raise KeyError(
            f"Element '{element}' not found in Rudnick & Gao (2013) table. "
            f"Available: {sorted(UPPER_CONTINENTAL_CRUST.keys())}"
        )
    return UPPER_CONTINENTAL_CRUST[element]


def compare_to_crust(concentration: float, element: str, sample_unit: str = "mg/kg") -> dict:
    """Compare a measured concentration to the upper continental crust value.

    Returns a dict with the ratio sample/crust and a qualitative label:
    - 'below_crust' if ratio < 0.5
    - 'similar_to_crust' if 0.5 ≤ ratio ≤ 2
    - 'enriched' if 2 < ratio ≤ 10
    - 'highly_enriched' if ratio > 10

    These bands are reference orientations only — they do not substitute
    TEL/PEL or Birch (2003) Enrichment Factor classifications.
    """
    ref = get_crust_reference(element)
    # Unit conversion: if crust value is in wt%, convert to mg/kg for comparison
    crust_value = ref.value
    if ref.unit == "wt%" and sample_unit == "mg/kg":
        crust_value = ref.value * 10000  # wt% to mg/kg
    elif ref.unit == "mg/kg" and sample_unit == "wt%":
        crust_value = ref.value / 10000

    if crust_value == 0:
        return {"ratio": float("nan"), "label": "undefined"}

    ratio = concentration / crust_value
    if ratio < 0.5:
        label = "below_crust"
    elif ratio <= 2:
        label = "similar_to_crust"
    elif ratio <= 10:
        label = "enriched"
    else:
        label = "highly_enriched"

    return {"ratio": ratio, "label": label, "crust_value": crust_value, "crust_unit": ref.unit}
```

Exportar el módulo en `aeda/interpretation/__init__.py`:

```python
from aeda.interpretation.crust_reference import (
    UPPER_CONTINENTAL_CRUST,
    CrustReferenceValue,
    get_crust_reference,
    compare_to_crust,
)
```

Y actualizar `__all__` con esos 4 nombres.

### Tests de regresión
Crear `tests/test_crust_reference.py`:

```python
"""Tests for the upper continental crust reference module."""

import pytest


def test_crust_table_includes_main_metals():
    """Table must include the metals relevant to environmental sediment studies."""
    from aeda.interpretation.crust_reference import UPPER_CONTINENTAL_CRUST
    required = {"Al", "Fe", "Cr", "Cu", "Ni", "Pb", "Zn", "As", "Cd", "Hg", "Mn"}
    missing = required - set(UPPER_CONTINENTAL_CRUST.keys())
    assert not missing, f"Missing main metals in Rudnick & Gao table: {missing}"


def test_get_crust_reference_returns_dataclass():
    from aeda.interpretation.crust_reference import get_crust_reference, CrustReferenceValue
    ref = get_crust_reference("Pb")
    assert isinstance(ref, CrustReferenceValue)
    assert ref.unit == "mg/kg"
    assert ref.value == 17.0


def test_get_crust_reference_unknown_raises():
    from aeda.interpretation.crust_reference import get_crust_reference
    with pytest.raises(KeyError):
        get_crust_reference("Unobtanium")


def test_compare_to_crust_labels():
    from aeda.interpretation.crust_reference import compare_to_crust
    # Pb crust = 17 mg/kg
    assert compare_to_crust(5.0, "Pb")["label"] == "below_crust"
    assert compare_to_crust(20.0, "Pb")["label"] == "similar_to_crust"
    assert compare_to_crust(100.0, "Pb")["label"] == "enriched"
    assert compare_to_crust(500.0, "Pb")["label"] == "highly_enriched"


def test_compare_unit_conversion_wtpct_to_mgkg():
    from aeda.interpretation.crust_reference import compare_to_crust
    # Al crust = 8.15 wt% = 81500 mg/kg
    # A sample with 4.0 wt% = 40000 mg/kg → ratio = 40000 / 81500 ≈ 0.49
    result = compare_to_crust(40000.0, "Al", sample_unit="mg/kg")
    assert 0.4 < result["ratio"] < 0.6


def test_module_is_exported_from_interpretation():
    from aeda.interpretation import (
        UPPER_CONTINENTAL_CRUST,
        get_crust_reference,
        compare_to_crust,
    )
    assert UPPER_CONTINENTAL_CRUST is not None
    assert callable(get_crust_reference)
    assert callable(compare_to_crust)
```

---

## CAMBIO 8 — Documentar HEAVY_METALS (no es regulación cubana)

### Archivo
`aeda/engine/auto_selector.py`

### Problema
Yoelvis cuestionó el nombre "regulados". Aclaró que **no existe normativa cubana específica para metales en sedimentos**; lo que se usa es la **Guía de Calidad** de NOAA (Buchman 2008), que no es regulación.

### Solución
Mantener el nombre `HEAVY_METALS` (para no romper la API), pero **mejorar el docstring** para que quede claro:

```python
# Heavy metals analyzed for environmental contamination assessment in sediments.
# This list is aligned with the keys of NOAA Buchman (2008) Screening Quick
# Reference Tables (SQuiRTs) for marine sediment.
#
# IMPORTANT TERMINOLOGY:
# These metals are NOT "regulated" in the strict legal sense — there is no
# Cuban regulation specific to metals in sediments. NOAA Buchman (2008) is a
# QUALITY GUIDELINE (Sediment Quality Guidelines, SQG), not a regulatory norm.
# The list reflects metals of environmental interest historically associated
# with anthropogenic impact in coastal and estuarine sediments.
#
# Source: Buchman, M.F. (2008). NOAA Screening Quick Reference Tables.
# NOAA OR&R Report 08-1. Seattle, WA.
# Confirmed with scientific tutor (Dr. Yoelvis Bolaños-Alvarez, CEAC, 2026).
HEAVY_METALS = {"As", "Cd", "Cr", "Cu", "Hg", "Ni", "Pb", "Zn", "Ag", "Sb"}
```

No requiere test adicional — es solo documentación.

---

## CAMBIO 9 — Clarificar "gradiente vertical" en docs y logs

### Archivos
`aeda/engine/auto_selector.py` y donde aparezcan mensajes al usuario

### Problema
Yoelvis preguntó: *"'gradiente vertical significativo', eso es que estás comparando las variables en la vertical??"*. Hay que clarificar la terminología.

### Solución

**Parte A — Docstrings**

En la función que detecta el gradiente vertical (en auto_selector), agregar/expandir docstring:

```python
def detect_vertical_gradient(df, depth_col, variables, alpha=0.05, r_threshold=0.3):
    """Detect variables with a significant vertical gradient in the dataset.

    A variable is considered to have a significant vertical gradient if its
    concentration varies systematically with depth — that is, if the Spearman
    rank correlation between the variable's values and the sample depth meets
    BOTH conditions:

    - p-value < alpha (default 0.05): the correlation is statistically significant.
    - |r| > r_threshold (default 0.3): the correlation has meaningful magnitude.

    A positive correlation means concentration increases with depth (greater
    values in older sediment); a negative correlation means concentration
    decreases with depth (greater values near the surface, typical of recent
    anthropogenic input).

    This analysis does NOT compare variables against each other; it characterizes
    each variable's relationship with depth independently.

    Parameters
    ----------
    df : DataFrame
    depth_col : str
        Name of the depth column.
    variables : list[str]
        Columns to evaluate.
    alpha : float
        Significance threshold for the p-value. Default 0.05.
    r_threshold : float
        Minimum |r| to consider the correlation meaningful. Default 0.3.

    Returns
    -------
    dict
        For each variable, its Spearman r, p-value, and whether it qualifies
        as having a significant vertical gradient.
    """
```

**Parte B — Mensajes al usuario**

Cualquier mensaje en `plan.recommendations` o en el reporte (`summary`) que diga "gradient detected" o "vertical gradient", cambiar a un mensaje más explícito. Por ejemplo:

Antes:
```
"Vertical gradient detected in 12 variables"
```

Después:
```
"12 variables show a significant correlation with sampling depth "
"(Spearman p<0.05, |r|>0.3): these variables likely reflect either "
"recent anthropogenic input (decreasing with depth) or diagenetic "
"processes (increasing with depth). Vertical profile analysis recommended."
```

No requiere test específico, pero ajustar tests existentes si fallan por cambios en strings.

---

## Ejecución y verificación

Después de aplicar los 9 cambios:

1. Ejecutar `pytest tests/ -v` y verificar que todos los tests pasan, incluidos los nuevos (deberían sumar aproximadamente 12-15 tests más respecto al estado previo).
2. Ejecutar el script de verificación contra ISOVIDA:

```python
from aeda.pipeline.runner import AEDAPipeline

EXCLUDE = ["No", "Code", "Site_Name", "Pret_Code", "Código_muestra",
           "Sitio_muestreo", "Fecha_muestreo", "Core",
           "Latitud", "Longitud", "Profundidad"]

p = AEDAPipeline(impute_strategy="median")
r = p.run("data/BD_ISOVIDA_MANGLARES2023_rectificadaYBA_230326.xlsx",
          exclude_cols=EXCLUDE, sheet_name="DATA")

print(r.summary())
print()
print(f"Units from dictionary: {r.dataset_info.units}")
print(f"Ancillary cols: {r.plan.profile.ancillary_cols}")
```

Verificar visualmente que:
- Las unidades aparecen leídas del diccionario (si la hoja existe en el Excel).
- Las variables ancillary están detectadas correctamente.
- No hay errores ni warnings inesperados.

## Commit único

Mensaje sugerido:

```
feat: integrate scientific tutor feedback (9 changes across the expert system)

This commit applies Dr. Yoelvis Bolaños-Alvarez's review of the 15 domain
rules, batch 1 (simple changes that do not require new functionality).

Changes:
1. TRACE_ELEMENTS: confirm Cr, Co, Ni, Cu, Zn, As, Mo, Pb are included.
2. Rename SEDIMENT_INDICATORS -> ANCILLARY_VARIABLES with docstring clarifying
   that PPI550 is OM indicator (not OM itself), PPI950 is carbonates indicator
   (not only CaCO3), and these are complementary variables ("ancillary data").
3. Remove rule 7 (FRX typical sum detection) — no scientific merit per tutor.
4. Lower correlation block threshold from 0.7 to 0.6 (reasonable strength per tutor).
5. Read units from Excel dictionary sheet (Diccionario_Data) when available;
   numeric heuristic kept as fallback.
6. CLR is now opt-in. The brain still recommends it on compositional data
   but the runner no longer applies it automatically. User must pass apply_clr=True.
7. Add Rudnick & Gao (2013) upper continental crust reference table as a new
   module aeda/interpretation/crust_reference.py, with helper compare_to_crust().
8. Document HEAVY_METALS list to clarify that NOAA Buchman is a Quality Guideline
   (not Cuban regulation; no Cuban regulation exists for metals in sediments).
9. Expand docstrings and user-facing messages around "vertical gradient" to
   clarify that the analysis evaluates each variable's correlation with depth,
   not variable-to-variable comparison.

Adds approximately 12-15 regression tests covering each change.
A separate batch (#10) will add spatial surface-layer analysis (0-10 cm) for
multi-site comparison, as a new feature with its own commit.
```
