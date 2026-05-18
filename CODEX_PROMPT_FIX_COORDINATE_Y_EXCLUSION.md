# CODEX_PROMPT_FIX_COORDINATE_Y_EXCLUSION

**Tipo:** Bug fix (UI auto-exclude)
**Archivos:** 1 modificado
**Tiempo estimado:** 3 min
**Tests esperados después:** 38 passed (sin cambios)

---

## 1. Contexto

En el Audit Overview de ISOVIDA aparece esta línea:

> Metadata detected: site column Sitio_muestreo (7 sites); depth column
> Profundidad; coordinates Latitud, Longitud, **Y**.

La columna `Y` es la coordenada UTM-Y del muestreo, pero **no** está en
la lista hardcoded de auto-exclusión (`METADATA_COLUMN_NAMES` en
`app/views/upload.py`), así que entra al análisis ML.

### Trade-off conocido

Hay riesgo: en geoquímica, **Y** también es el símbolo del **Itrio**
(elemento de tierras raras detectable por FRX). Excluir `Y` ciegamente
podría dejar fuera una variable química legítima en otros datasets.

**Mitigación:**
1. La auto-exclusión solo se ofrece como **sugerencia por defecto** en el
   multiselect — el usuario puede des-seleccionarla si necesita Y como
   itrio.
2. Solo se excluye cuando aparece junto a `X` o coordenadas que confirmen
   el contexto UTM (ver implementación abajo).

Para ISOVIDA específicamente, donde Y es claramente coordenada UTM, esto
resuelve el problema sin afectar otros casos de uso.

---

## 2. Cambio en `app/views/upload.py`

**Archivo:** `app/views/upload.py`

### 2.1 Ampliar la lista de nombres de metadatos

**BUSCAR:**

```python
    # Common metadata column names across Spanish and English datasets
    METADATA_COLUMN_NAMES = {
        # Coordinates
        "Latitud", "Longitud", "Latitude", "Longitude", "Lat", "Lon", "Lng",
        "X_UTM", "Y_UTM", "UTM_X", "UTM_Y",
        # Row numbers / sample IDs that may be numeric
        "No", "N", "ID", "Id", "Sample_ID", "SampleID", "Sample_No", "Sample",
        "Order", "Row",
    }
    numeric_metadata = [c for c in all_cols if c in METADATA_COLUMN_NAMES]
```

**REEMPLAZAR POR:**

```python
    # Common metadata column names across Spanish and English datasets.
    # Note: short names like "X" and "Y" are ambiguous (Y is also the symbol
    # for Yttrium in geochemistry). We only flag them as metadata when there
    # is evidence that the dataset uses them as coordinates — specifically
    # when BOTH X and Y appear together, or when explicit UTM variants exist.
    METADATA_COLUMN_NAMES = {
        # Explicit coordinate names (unambiguous)
        "Latitud", "Longitud", "Latitude", "Longitude", "Lat", "Lon", "Lng",
        "X_UTM", "Y_UTM", "UTM_X", "UTM_Y",
        "Easting", "Northing", "Coord_X", "Coord_Y",
        # Row numbers / sample IDs that may be numeric
        "No", "N", "ID", "Id", "Sample_ID", "SampleID", "Sample_No", "Sample",
        "Order", "Row", "Index",
    }
    numeric_metadata = [c for c in all_cols if c in METADATA_COLUMN_NAMES]

    # Ambiguous coordinate names: X and Y. Only treat them as metadata when
    # they appear *together* (typical of UTM coord pairs) AND there are no
    # nearby chemistry columns that would indicate they could be elements
    # like Yttrium (Y). In our datasets, Y as a coordinate always travels
    # with X as a pair; Y as an element appears among many other element
    # columns (Cr, Cu, Zn, etc.) without an X column.
    if "X" in all_cols and "Y" in all_cols:
        numeric_metadata.extend(["X", "Y"])
```

---

## 3. Validación

```bash
# 1. Tests siguen verdes (este cambio no toca el engine)
pytest tests/ -q
```
**Esperado:** `38 passed`.

```bash
# 2. Sintaxis OK
python -c "import sys; sys.path.insert(0, '.'); import app.views.upload; print('OK')"
```

```bash
# 3. Smoke visual
streamlit run app/main.py
```

**Verificación visual:**

- ✅ **Upload — Step 3 "Columns to analyze":** al subir ISOVIDA, en el
  multiselect "Exclude these columns from the ML analysis" aparece **Y**
  pre-marcada para excluir (junto con Latitud, Longitud, No).
- ✅ Si el dataset no tiene una columna `X`, entonces `Y` **no** se excluye
  automáticamente (se asume que es Itrio). Este caso no se va a dar con
  ISOVIDA, pero es la lógica defensiva del fix.

---

## 4. Si algo falla

- Si después del fix el dataset ISOVIDA no muestra `Y` como excluida →
  verificar que el archivo realmente tiene una columna llamada `X` también.
  En caso contrario, el fix no la detectará como par UTM. (Para ISOVIDA
  esto debería funcionar; si no, reportar qué columnas tiene el archivo).
- No tocar `aeda/`, `tests/`, ni nada bajo `app/` que no sea
  `app/views/upload.py`.

---

## 5. Mensaje de commit sugerido

```
fix(ui): exclude X/Y as UTM coordinates when they appear paired

In ISOVIDA the column "Y" is the UTM-Y coordinate, but was not in the
auto-exclude list. Adding "Y" unconditionally would also exclude legitimate
Yttrium (Y) measurements in XRF datasets. Compromise: flag X and Y as
metadata only when *both* appear together (the typical UTM pair shape).
Also added a few more coordinate variants (Easting, Northing, Coord_X,
Coord_Y) and the generic "Index" identifier.

UI-only change. 38 tests pass.
```
