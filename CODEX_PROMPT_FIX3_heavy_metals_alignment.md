# BUG IMPORTANTE 3 — HEAVY_METALS desalineada con tabla NOAA

## Contexto

En `aeda/engine/auto_selector.py` la constante `HEAVY_METALS` no coincide con la tabla NOAA Buchman 2008 implementada en `aeda/interpretation/thresholds.py`. Esto produce una inconsistencia visible: el cerebro detecta como "heavy metals" elementos que después no pueden compararse con TEL/PEL, y omite metales que sí tienen umbrales NOAA.

Además, `Mn` está duplicado: aparece tanto en `MAJOR_ELEMENTS` como en `HEAVY_METALS`.

## Diagnóstico verificado

```python
# Estado actual:
HEAVY_METALS = {"Cr", "Mn", "Co", "Ni", "Cu", "Zn", "As", "Pb", "Mo"}

# Tabla NOAA (en thresholds.py):
TEL_PEL_MARINE_SEDIMENT keys = {"As", "Cd", "Cr", "Cu", "Hg", "Ni", "Pb", "Zn", "Ag", "Sb"}

# Inconsistencia con ISOVIDA:
# Detectados como HEAVY pero sin TEL/PEL: {Mn, Co, Mo}
# Con TEL/PEL pero NO detectados: {Cd, Hg, Ag, Sb}
```

Esto también es coherente con la lista que figura en el documento que mandaste a Yoelvis ("Bloque A: regla 3 — As, Cd, Cr, Cu, Hg, Ni, Pb, Zn, Ag, Sb").

## Cambios requeridos

### 1. Modificar `aeda/engine/auto_selector.py`

**Reemplazar** (líneas ~32-37):

```python
MAJOR_ELEMENTS = {"Na", "Mg", "Al", "Si", "K", "Ca", "Fe", "Ti", "Mn", "P"}
TRACE_ELEMENTS = {
    "V", "Cr", "Co", "Ni", "Cu", "Zn", "Ga", "As", "Br", "Rb",
    "Sr", "Y", "Zr", "Nb", "Mo", "Ba", "Pb", "Sc", "S", "Cl",
}
HEAVY_METALS = {"Cr", "Mn", "Co", "Ni", "Cu", "Zn", "As", "Pb", "Mo"}
```

**Por:**

```python
# Geochemical element classification (aligned with NOAA Buchman 2008 thresholds
# and reviewed with the project's scientific tutor).
# These constants are also exposed for external alignment with the
# interpretation module thresholds table.
MAJOR_ELEMENTS = {"Na", "Mg", "Al", "Si", "K", "Ca", "Fe", "Ti", "Mn", "P"}
TRACE_ELEMENTS = {
    "V", "Cr", "Co", "Ni", "Cu", "Zn", "Ga", "As", "Br", "Rb",
    "Sr", "Y", "Zr", "Nb", "Mo", "Ba", "Pb", "Sc", "S", "Cl",
    "Cd", "Hg", "Ag", "Sb", "Se",
}
# Regulated heavy metals with TEL/PEL thresholds in NOAA Buchman (2008).
# This list MUST stay in sync with aeda.interpretation.thresholds.TEL_PEL_MARINE_SEDIMENT.
HEAVY_METALS = {"As", "Cd", "Cr", "Cu", "Hg", "Ni", "Pb", "Zn", "Ag", "Sb"}
```

Notas sobre el cambio:

- `MAJOR_ELEMENTS`: sin cambios.
- `TRACE_ELEMENTS`: se agregan `Cd`, `Hg`, `Ag`, `Sb`, `Se` para que también puedan ser detectados como traza si aparecen.
- `HEAVY_METALS`: ahora coincide exactamente con las claves de `TEL_PEL_MARINE_SEDIMENT`. Se eliminan `Mn`, `Co`, `Mo` (no tienen TEL/PEL en la tabla NOAA del módulo) y se agregan `Cd`, `Hg`, `Ag`, `Sb`.
- `Mn` queda solo en `MAJOR_ELEMENTS` (donde corresponde por su composición típica > 1%).

### 2. Asegurar la sincronización con la tabla NOAA

Al final del archivo `aeda/engine/auto_selector.py`, después de las constantes pero antes de los dataclasses, agregar una verificación en tiempo de import:

```python
# Sanity check: HEAVY_METALS must stay aligned with the NOAA thresholds table
# in the interpretation module. Drift between these two lists creates the bug
# where the brain "detects" metals that the interpretation module cannot classify.
def _check_heavy_metals_alignment() -> None:
    try:
        from aeda.interpretation.thresholds import TEL_PEL_MARINE_SEDIMENT
    except ImportError:  # pragma: no cover
        # interpretation module not available at import time — skip silently.
        return
    noaa_metals = set(TEL_PEL_MARINE_SEDIMENT.keys())
    if HEAVY_METALS != noaa_metals:
        # Use a lazy assertion so the import does not fail in production but
        # surfaces the issue clearly during development.
        import warnings
        warnings.warn(
            f"HEAVY_METALS in auto_selector ({sorted(HEAVY_METALS)}) does not match "
            f"NOAA TEL/PEL table ({sorted(noaa_metals)}). "
            f"This will cause inconsistencies between the brain's recommendations "
            f"and the interpretation module.",
            UserWarning,
        )


_check_heavy_metals_alignment()
```

### 3. Agregar tests de regresión en `tests/test_integration.py`

```python
def test_heavy_metals_aligned_with_noaa_table():
    """Regression: HEAVY_METALS in auto_selector must equal the NOAA thresholds table."""
    from aeda.engine.auto_selector import HEAVY_METALS
    from aeda.interpretation.thresholds import TEL_PEL_MARINE_SEDIMENT

    assert HEAVY_METALS == set(TEL_PEL_MARINE_SEDIMENT.keys()), (
        f"HEAVY_METALS={sorted(HEAVY_METALS)} does not match "
        f"NOAA={sorted(TEL_PEL_MARINE_SEDIMENT.keys())}"
    )


def test_mn_only_in_major_elements_not_heavy():
    """Mn is a major (lithogenic) element, not a regulated heavy metal in NOAA."""
    from aeda.engine.auto_selector import MAJOR_ELEMENTS, HEAVY_METALS
    assert "Mn" in MAJOR_ELEMENTS
    assert "Mn" not in HEAVY_METALS


def test_cd_hg_ag_sb_now_recognized_as_heavy_metals():
    """These metals have NOAA thresholds and must be detected by the brain."""
    from aeda.engine.auto_selector import HEAVY_METALS
    for metal in ("Cd", "Hg", "Ag", "Sb"):
        assert metal in HEAVY_METALS, f"{metal} should be in HEAVY_METALS (has NOAA TEL/PEL)"


def test_isovida_heavy_metal_detection_includes_relevant_metals():
    """Validation against the real ISOVIDA dataset."""
    from aeda.pipeline.runner import AEDAPipeline
    EXCLUDE = ["No", "Code", "Site_Name", "Pret_Code", "Código_muestra",
               "Sitio_muestreo", "Fecha_muestreo", "Core",
               "Latitud", "Longitud", "Profundidad"]
    p = AEDAPipeline(impute_strategy="median")
    r = p.run("data/BD_ISOVIDA_MANGLARES2023_rectificadaYBA_230326.xlsx",
              exclude_cols=EXCLUDE, sheet_name="DATA")

    detected = set(r.plan.profile.heavy_metal_cols)
    # ISOVIDA's FRX measurements include at least these heavy metals
    expected_detected = {"As", "Cr", "Cu", "Ni", "Pb", "Zn"}
    assert expected_detected.issubset(detected), (
        f"Expected at least {expected_detected} to be detected as heavy metals, got {detected}"
    )

    # Mn must NOT appear here (it's in MAJOR)
    assert "Mn" not in detected
```

## Impacto esperado en ISOVIDA

Antes del fix: `heavy_metal_cols = ['As', 'Co', 'Cr', 'Cu', 'Mn', 'Mo', 'Ni', 'Pb', 'Zn']`
Después del fix: `heavy_metal_cols = ['As', 'Cr', 'Cu', 'Ni', 'Pb', 'Zn']` (no aparecen Cd, Hg, Ag, Sb porque ISOVIDA no los mide).

Esta lista más corta es la **correcta** desde el punto de vista de Yoelvis: solo aparecen metales que el módulo de interpretación puede clasificar contra TEL/PEL.

## Verificación

1. `pytest tests/ -v` → todos pasan, incluidos los 4 nuevos.
2. Correr el script de verificación contra ISOVIDA y revisar que `heavy_metal_cols` ahora solo contiene metales con TEL/PEL.
3. Verificar que la página Plan de Streamlit ya no muestra Mn, Co, Mo en "Heavy metals".

## Commit

Mensaje sugerido:

```
fix(auto_selector): align HEAVY_METALS with NOAA Buchman 2008 thresholds table

The HEAVY_METALS constant used by the brain was inconsistent with the
TEL_PEL_MARINE_SEDIMENT table in the interpretation module:
- The brain detected Mn, Co, Mo as heavy metals (no TEL/PEL → unclassifiable)
- The brain ignored Cd, Hg, Ag, Sb (which DO have TEL/PEL)

This drift created a confusing UX: the Plan page showed metals that the
interpretation module could not classify, and missed metals that it could.

Changes:
- HEAVY_METALS now equals the keys of NOAA's TEL_PEL_MARINE_SEDIMENT.
- Mn moved out (it's already in MAJOR_ELEMENTS where it belongs).
- Cd, Hg, Ag, Sb, Se added to TRACE_ELEMENTS so they can still be detected
  if a dataset provides them.
- Added _check_heavy_metals_alignment() at module import time that warns if
  the two lists ever drift apart again.

Adds 4 regression tests including one validating against the real ISOVIDA dataset.
```
