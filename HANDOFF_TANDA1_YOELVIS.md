# Handoff de contexto — Tanda 1 de Yoelvis aplicada

> **Para:** Chat de programación de AEDA-AI
> **De:** Chat de tesis
> **Propósito:** Sincronizar el contexto del trabajo realizado mientras estabas esperando la respuesta de Yoelvis, y dejar listo el próximo paso (Tanda 2).

---

## 1. Qué pasó (cronología)

1. **Pausa del desarrollo** del software hasta recibir la respuesta del tutor científico Yoelvis Bolaños-Alvarez sobre las 15 reglas químico-ambientales del sistema experto.
2. **Yoelvis respondió** las 15 preguntas (vía documento Word con track changes).
3. **Se interpretaron las respuestas** y se identificaron 10 cambios al código.
4. **Se decidió** dividir los cambios en dos tandas:
   - **Tanda 1** (9 cambios simples e independientes) — APLICADA por Codex en este otro chat
   - **Tanda 2** (análisis espacial de fracción superficial 0-10 cm) — pendiente
5. Mientras tanto, en el chat de programación, se trabajó en la página Auditoría y Configuración Avanzada (Fase 2 de Streamlit), con refinamiento de errores adicionales.

---

## 2. Estado actual del código

### Último commit relevante de Tanda 1
- **Hash:** `9329074`
- **Tests:** 53/53 pasando (subió desde 33 antes de la tanda)
- **Branch:** trabajando en main (asumido)

### Archivos modificados en Tanda 1
- `aeda/engine/auto_selector.py` — múltiples cambios
- `aeda/interpretation/__init__.py` — exports añadidos
- `aeda/interpretation/crust_reference.py` — **NUEVO**
- `aeda/io/parsers.py` — leer unidades del diccionario
- `aeda/pipeline/runner.py` — wiring de unidades, CLR opt-in
- `app/views/advanced.py` — toggle CLR
- `app/views/depth.py` — ajustes menores
- `app/views/upload.py` — ajustes menores
- `tests/test_integration.py` — tests nuevos
- `tests/test_crust_reference.py` — **NUEVO**

### Otras cosas modificadas en paralelo (chat de programación)
- Página de Auditoría implementada
- Página de Configuración Avanzada implementada
- Refinamiento de errores y bugs encontrados durante la implementación

---

## 3. Detalle de los 9 cambios de Tanda 1

### Cambio 1 — TRACE_ELEMENTS amplía cobertura

**Razón:** Yoelvis identificó que faltaban elementos que sí están en ISOVIDA.

**Estado:** Verificado en código. La lista debe contener:
```
{"V", "Cr", "Co", "Ni", "Cu", "Zn", "Ga", "As", "Br", "Rb",
 "Sr", "Y", "Zr", "Nb", "Mo", "Ba", "Pb", "Sc", "S", "Cl",
 "Cd", "Hg", "Ag", "Sb", "Se"}
```

**Test asociado:** `test_trace_elements_includes_isovida_metals`

---

### Cambio 2 — Renombrar SEDIMENT_INDICATORS → ANCILLARY_VARIABLES

**Razón:** Yoelvis aclaró que TOC, PPI550, PPI950, HC, CaCO3 no son "indicadores de calidad", sino **variables complementarias** ("ancillary data" en literatura).

**Específicamente:**
- PPI550 = indicador de MO (no es la MO en sí)
- PPI950 = indicador de carbonatos (no solo CaCO3)

**Constante:**
```python
ANCILLARY_VARIABLES = {"TOC", "OM", "PPI550", "PPI950", "HC", "CaCO3"}
```

**Renombrado también:** el atributo `sediment_indicator_cols` del DatasetProfile a `ancillary_cols`.

**Test asociado:** `test_ancillary_variables_renamed`

---

### Cambio 3 — Eliminada regla 7 (FRX detection por suma)

**Razón:** Yoelvis: *"No creo que este criterio tenga mucho sentido por ahora. Los mayoritarios pueden ser altos indistintamente, hay rangos para cada elemento."*

**Acción:** Bloque de código y campo `is_frx_typical` eliminados de `auto_selector.py` y `DatasetProfile`.

**Test asociado:** `test_frx_typical_rule_removed`

---

### Cambio 4 — Umbral correlación 0.7 → 0.6

**Razón:** Yoelvis: *"Creo que 0.6, eso daría un valor de fortaleza razonable, solo que lo define las pruebas de significancia estadística."*

**Constante:**
```python
CORRELATION_BLOCK_THRESHOLD = 0.6
```

**Test asociado:** `test_correlation_block_threshold_is_06`

---

### Cambio 5 — Unidades desde diccionario del Excel (con fallback)

**Razón:** Yoelvis aclaró que en ISOVIDA solo Si, Al, Ca, K, Fe, Na, Mg están en %, el resto en mg/kg. Esta info está en la hoja **Diccionario_Data** del Excel.

**Arquitectura:**
1. El parser (`aeda/io/parsers.py`) ahora extrae la columna de unidades de la hoja de diccionario (busca "Unidad", "Unit", "Units", "unidades" case-insensitive).
2. `DatasetInfo` ahora tiene un campo `units: dict[str, str]`.
3. La función `detect_mixed_units()` ahora acepta un `units_dict` opcional:
   - **Si se pasa**: prioriza el diccionario.
   - **Si no**: usa la heurística numérica anterior (renombrada a `_detect_mixed_units_numeric`).
4. El `runner.py` pasa el diccionario al perfilador del cerebro.

**Tests asociados:**
- `test_mixed_units_from_dictionary_preferred_over_heuristic`
- `test_mixed_units_falls_back_to_heuristic_when_no_dictionary`
- `test_isovida_units_loaded_from_dictionary`

---

### Cambio 6 — CLR opt-in (no automático)

**Razón:** Yoelvis: *"Dejalo manual"*. El sistema no debe aplicar CLR automáticamente al detectar composicionalidad.

**Cambios:**
- `AEDAPipeline.__init__` ahora tiene `apply_clr: bool = False` (antes era "auto" o similar).
- El runner ya no activa CLR automáticamente al detectar composicionalidad.
- El cerebro **sigue detectando** composicionalidad y **sigue recomendándola** en el plan con mensaje explícito: *"CLR is recommended... must be explicitly enabled by the user (apply_clr=True). It will NOT be applied automatically."*
- En el UI (`upload.py`, `advanced.py`) se ajustó el toggle CLR.

**Tests asociados (versión robusta basada en preprocessing_log.steps):**
- `test_clr_not_applied_automatically`
- `test_clr_applied_when_explicitly_enabled`

---

### Cambio 7 — Tabla Rudnick & Gao 2013 (corteza terrestre)

**Razón:** Yoelvis: *"Puedes mantener a Rudnick and Gao 2013 como referencia general de valores de la corteza terrestre para elementos."*

**Archivo nuevo:** `aeda/interpretation/crust_reference.py`

**Contenido:**
- Dataclass `CrustReferenceValue(value, unit, notes)`
- Diccionario `UPPER_CONTINENTAL_CRUST` con valores de Rudnick & Gao (2013) Tabla 3.
- Cubre majors (en wt%) y traza (en mg/kg) para todos los elementos relevantes a estudios ambientales:
  - Majors: Si, Al, Fe, Ca, Na, K, Mg, Ti, P, Mn
  - Traza: As, Ba, Cd, Co, Cr, Cs, Cu, Ga, Hg, Li, Mo, Nb, Ni, Pb, Rb, Sb, Sc, Sn, Sr, Th, U, V, Y, Zn, Zr
- Función `get_crust_reference(element)`
- Función `compare_to_crust(concentration, element, sample_unit)` que devuelve `{ratio, label}` con bandas:
  - `below_crust` (< 0.5)
  - `similar_to_crust` (0.5 – 2)
  - `enriched` (2 – 10)
  - `highly_enriched` (> 10)

**Exportado en** `aeda/interpretation/__init__.py`.

**Tests asociados (en archivo nuevo `tests/test_crust_reference.py`):**
- `test_crust_table_includes_main_metals`
- `test_get_crust_reference_returns_dataclass`
- `test_get_crust_reference_unknown_raises`
- `test_compare_to_crust_labels`
- `test_compare_unit_conversion_wtpct_to_mgkg`
- `test_module_is_exported_from_interpretation`

---

### Cambio 8 — Documentación de HEAVY_METALS

**Razón:** Yoelvis cuestionó el término "regulados". Aclaró que **no hay normativa cubana** para metales en sedimentos. NOAA Buchman 2008 es una **Guía de Calidad de Sedimentos** (SQG), no normativa.

**Acción:** Docstring expandido encima de la constante. La lista (`{As, Cd, Cr, Cu, Hg, Ni, Pb, Zn, Ag, Sb}`) no cambia, pero se aclara que:
- No es una lista regulada en sentido legal.
- No existe regulación cubana para metales en sedimentos.
- NOAA Buchman 2008 es una guía de calidad (SQG), no norma.
- La lista coincide con las keys de TEL_PEL_MARINE_SEDIMENT.

Sin test específico (solo documentación).

---

### Cambio 9 — Clarificar "gradiente vertical" en docs y logs

**Razón:** Yoelvis preguntó: *"'gradiente vertical significativo', eso es que estás comparando las variables en la vertical?"*. Confusión terminológica.

**Acción:**
- Docstring de `detect_vertical_gradient()` expandido para aclarar que el análisis evalúa la **correlación de cada variable con la profundidad de manera independiente**, no compara variables entre sí.
- Mensajes al usuario actualizados (más explícitos, incluyen los umbrales y la interpretación).

Sin test específico (solo documentación).

---

## 4. Cambio 10 — PENDIENTE (Tanda 2)

### Análisis espacial de fracción superficial 0-10 cm

**Origen:** Pregunta 11 a Yoelvis sobre "priorización vertical".

**Aclaración de Yoelvis:**
> *"En ese encuentro te dije que dejamos los tres, pero hay análisis que se hacen verticales, y otros horizontales… entre múltiples sitios se prioriza el análisis espacial, pero generalmente de la fracción superficial que está entre 0-10 cm (para algunos autores es 0-5 cm, otros 0-20 cm)."*

**Lo que falta implementar:**
1. **Filtrado por capa superficial** configurable (default 10 cm, parametrizable a 5 cm o 20 cm).
2. **Clustering espacial sobre la capa superficial** entre sitios (no sobre todo el dataset).
3. **Visualizaciones específicas** para la fracción superficial (mapa de calor por sitio, scatter geográfico si hay coordenadas, etc.).
4. **Reportar ambos análisis en paralelo:** vertical (perfiles por sitio) y espacial (capa superficial entre sitios).
5. **Recomendación del cerebro** que active el análisis espacial cuando detecta múltiples sitios + profundidad.

**Alcance estimado:** 2-3 horas de Codex.

---

## 5. Verificación contra ISOVIDA (próximo paso recomendado)

Antes de avanzar con Tanda 2 o seguir refinando la UI, conviene correr un script de verificación que confirme que la Tanda 1 funciona correctamente sobre el dataset real.

**Script propuesto** (`verify_batch1.py`):

```python
"""Verification script: confirms batch 1 changes work on the real ISOVIDA dataset."""

from aeda.pipeline.runner import AEDAPipeline
from aeda.engine.auto_selector import (
    HEAVY_METALS, TRACE_ELEMENTS, ANCILLARY_VARIABLES,
    CORRELATION_BLOCK_THRESHOLD,
)

EXCLUDE = ["No", "Code", "Site_Name", "Pret_Code", "Código_muestra",
           "Sitio_muestreo", "Fecha_muestreo", "Core",
           "Latitud", "Longitud", "Profundidad"]

print("=" * 72)
print("BATCH 1 VERIFICATION ON ISOVIDA")
print("=" * 72)

print(f"\nConstants check:")
print(f"  CORRELATION_BLOCK_THRESHOLD = {CORRELATION_BLOCK_THRESHOLD}  (expected 0.6)")
print(f"  HEAVY_METALS = {sorted(HEAVY_METALS)}")
print(f"  ANCILLARY_VARIABLES = {sorted(ANCILLARY_VARIABLES)}")
print(f"  TRACE_ELEMENTS has {len(TRACE_ELEMENTS)} elements")

# Run pipeline with default settings (apply_clr should be False)
p = AEDAPipeline(impute_strategy="median")
r = p.run("data/BD_ISOVIDA_MANGLARES2023_rectificadaYBA_230326.xlsx",
          exclude_cols=EXCLUDE, sheet_name="DATA")

print(f"\n--- Dataset info ---")
print(f"  Total samples: {len(r.raw_data)}")
print(f"  Units from dictionary: {len(r.dataset_info.units)} entries")
if r.dataset_info.units:
    # Show a few examples
    sample_units = {k: r.dataset_info.units[k] for k in list(r.dataset_info.units)[:6]}
    print(f"  Sample units: {sample_units}")

print(f"\n--- Profile ---")
print(f"  Heavy metal cols detected: {r.plan.profile.heavy_metal_cols}")
print(f"  Ancillary cols detected: {r.plan.profile.ancillary_cols}")
print(f"  Major elements detected: {r.plan.profile.major_element_cols}")

print(f"\n--- Preprocessing log ---")
clr_steps = [s for s in r.preprocessing_log.steps if "clr" in str(s).lower()]
print(f"  CLR steps in log: {len(clr_steps)}  (expected 0 by default)")

print(f"\n--- Interpretation ---")
if r.interpretation:
    print(f"  Reference element: {r.interpretation.ef_result.reference_element}")
    print(f"  Metals analyzed: {r.interpretation.metals_analyzed}")
    print(f"  Baseline strategy: {r.interpretation.ef_result.baseline_strategy}")

print(f"\n--- Verification ---")
checks = []
checks.append(("Units loaded from dictionary", len(r.dataset_info.units) > 0))
checks.append(("Correlation threshold is 0.6", CORRELATION_BLOCK_THRESHOLD == 0.6))
checks.append(("HEAVY_METALS aligned with NOAA", HEAVY_METALS == {"As", "Cd", "Cr", "Cu", "Hg", "Ni", "Pb", "Zn", "Ag", "Sb"}))
checks.append(("CLR not applied automatically", len(clr_steps) == 0))
checks.append(("ANCILLARY_VARIABLES exists", "ANCILLARY_VARIABLES" in dir()))
checks.append(("Interpretation ran on ISOVIDA", r.interpretation is not None))

for label, ok in checks:
    mark = "✓" if ok else "✗"
    print(f"  [{mark}] {label}")

if all(ok for _, ok in checks):
    print("\n  All batch 1 checks PASSED on ISOVIDA.")
else:
    print("\n  Some checks FAILED. Review above.")
```

Guardarlo en la raíz del proyecto y correrlo con `python verify_batch1.py > verify_batch1_output.txt` después de aplicar Tanda 1.

---

## 6. Próximos pasos sugeridos (orden recomendado)

### Inmediato
1. **Correr el script de verificación de Tanda 1** sobre ISOVIDA y guardar la salida.
2. **Compartir salida** con el chat de tesis (sirve como evidencia para el Capítulo 3 y para mostrar a Yoelvis).

### Corto plazo (esta semana o próxima)
3. **Tanda 2 — Análisis espacial 0-10 cm** (la única feature pendiente del feedback de Yoelvis).
4. **Refinar páginas Streamlit** (Auditoría y Configuración Avanzada) si quedaron pendientes detalles.

### Mediano plazo
5. **Deploy en Streamlit Cloud** (1 día).
6. **Manual de usuario** para el anexo CENDA (2-3 días).
7. **Demo en vivo a Yoelvis** con todo integrado.

---

## 7. Información de contacto entre chats

**Chat de tesis** (este): se sigue usando para
- Redacción de capítulos
- Documentos para tutores
- Gestión de Mendeley/bibliografía
- Interpretación de feedback de tutores

**Chat de programación** (al que vas): se sigue usando para
- Desarrollo y bugs del código
- Tests
- Auditorías y refactors
- Implementación de páginas Streamlit
- Despliegue

Cuando necesites cruce de información (ej: pegar resultados de un análisis científico en la tesis, o discutir cómo redactar un cambio metodológico), pasás el archivo o el resultado entre chats con un mensaje breve de contexto.

---

## 8. TL;DR para el chat de programación

> Tanda 1 (9 cambios del feedback de Yoelvis) aplicada y commiteada (hash `9329074`, 53/53 tests). Mientras tanto avancé en Auditoría y Configuración Avanzada en este mismo chat. Próximo paso: correr el script de verificación de Tanda 1 contra ISOVIDA, después arrancar la Tanda 2 (análisis espacial de fracción superficial 0-10 cm). Adjunto este documento de contexto y el RAR actualizado para referencia.
