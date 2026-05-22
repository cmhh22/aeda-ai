# CODEX_PROMPT_FIX_SURFACE_EXCLUDE_COLS

**Tipo:** Bug fix (visual QA del PDF de surface tab)
**Archivos:** 2 modificados
**Tiempo estimado:** 5 min
**Tests esperados después:** 71 passed (sin cambios)

---

## 1. Contexto

En el smoke visual de la pestaña Surface (spatial) sobre ISOVIDA, el
heatmap muestra como primera columna `No` — el índice de fila del Excel,
que es numérico pero no es una variable química.

**Causa:** `info.measurement_cols` lo lista como medición (es numérico y
el parser no tiene una regla por nombre para excluirlo). El usuario sí
lo marcó para excluir en Upload, pero esa exclusión se aplica solo a
PCA / clustering / correlations, no al surface analysis.

El runner pasa `info.measurement_cols` íntegro al `surface_spatial_analysis`
sin respetar `exclude_cols`. Y la UI hace lo mismo cuando recalcula al
cambiar el dropdown de profundidad.

Validado el fix contra ISOVIDA: tras aplicarlo, el surface analysis usa
35 columnas (en vez de 36) y la columna `No` desaparece del heatmap y
del clustering.

---

## 2. Cambio en `aeda/pipeline/runner.py`

**BUSCAR:**

```python
        # 11. SURFACE-LAYER SPATIAL ANALYSIS
        # Only runs when both a site column and a depth column are available.
        # The analysis filters to the surface layer (default 10 cm), averages
        # by site, and clusters sites. See aeda/engine/spatial_surface.py.
        if info.site_col is not None and info.depth_col is not None:
            try:
                results.surface_analysis = surface_spatial_analysis(
                    df,
                    depth_col=info.depth_col,
                    site_col=info.site_col,
                    measurement_cols=info.measurement_cols,
                    max_depth_cm=self.surface_depth_cm,
                    coordinate_cols=info.coordinate_cols or None,
                )
            except Exception as e:
                logger.warning(
                    f"Surface spatial analysis failed: {type(e).__name__}: {e}"
                )
                results.surface_analysis = None
```

**REEMPLAZAR POR:**

```python
        # 11. SURFACE-LAYER SPATIAL ANALYSIS
        # Only runs when both a site column and a depth column are available.
        # The analysis filters to the surface layer (default 10 cm), averages
        # by site, and clusters sites. See aeda/engine/spatial_surface.py.
        if info.site_col is not None and info.depth_col is not None:
            # Respect the user's exclude_cols: info.measurement_cols is the
            # parser's view of "what looks like a measurement", but the user
            # may have flagged columns like "No" (row index) as metadata in
            # the Upload page. We honor that here so the surface heatmap and
            # site clustering only use real chemistry.
            surface_measurement_cols = [
                c for c in info.measurement_cols
                if not exclude_cols or c not in exclude_cols
            ]
            try:
                results.surface_analysis = surface_spatial_analysis(
                    df,
                    depth_col=info.depth_col,
                    site_col=info.site_col,
                    measurement_cols=surface_measurement_cols,
                    max_depth_cm=self.surface_depth_cm,
                    coordinate_cols=info.coordinate_cols or None,
                )
            except Exception as e:
                logger.warning(
                    f"Surface spatial analysis failed: {type(e).__name__}: {e}"
                )
                results.surface_analysis = None
```

---

## 3. Cambio en `app/views/_surface_tab.py`

La UI replica la misma lógica cuando el usuario cambia el dropdown de
profundidad (cada cambio dispara un recálculo directo del módulo, sin
pasar por el runner).

**BUSCAR:**

```python
    # Recompute only if the user picked a different depth than the cached
    # pipeline result. The surface module is cheap; the rest of the
    # pipeline output is preserved.
    if float(selected_depth) != float(initial.max_depth):
        try:
            current = surface_spatial_analysis(
                raw_df,
                depth_col=info.depth_col,
                site_col=info.site_col,
                measurement_cols=info.measurement_cols,
                max_depth_cm=float(selected_depth),
                coordinate_cols=info.coordinate_cols or None,
            )
        except Exception as e:
            st.error(
                f"Could not recompute the surface analysis at {selected_depth} cm: "
                f"{type(e).__name__}: {e}"
            )
            return
    else:
        current = initial
```

**REEMPLAZAR POR:**

```python
    # Recompute only if the user picked a different depth than the cached
    # pipeline result. The surface module is cheap; the rest of the
    # pipeline output is preserved.
    if float(selected_depth) != float(initial.max_depth):
        # Replicate the same measurement_cols filter the pipeline applies
        # (drop user-excluded columns) so the recomputation matches the
        # initial result's column scope.
        run_ctx = st.session_state.get("run_context") or {}
        excluded = run_ctx.get("exclude_cols") or []
        surface_cols = [c for c in info.measurement_cols if c not in excluded]
        try:
            current = surface_spatial_analysis(
                raw_df,
                depth_col=info.depth_col,
                site_col=info.site_col,
                measurement_cols=surface_cols,
                max_depth_cm=float(selected_depth),
                coordinate_cols=info.coordinate_cols or None,
            )
        except Exception as e:
            st.error(
                f"Could not recompute the surface analysis at {selected_depth} cm: "
                f"{type(e).__name__}: {e}"
            )
            return
    else:
        current = initial
```

---

## 4. Validación

```bash
# 1. Tests siguen verdes
pytest tests/ -q
```
**Esperado:** `71 passed`.

```bash
# 2. Smoke contra ISOVIDA: verificar que "No" desaparece
python -c "
from aeda.pipeline.runner import AEDAPipeline
EXCLUDE = ['No','Code','Site_Name','Pret_Code','Código_muestra',
           'Sitio_muestreo','Fecha_muestreo','Core','Latitud','Longitud']
r = AEDAPipeline(impute_strategy='median').run(
    'data/BD_ISOVIDA_MANGLARES2023_rectificadaYBA_230326.xlsx',
    exclude_cols=EXCLUDE, sheet_name='DATA',
)
cols = list(r.surface_analysis.site_means.columns)
print(f'Total cols: {len(cols)}')
print(f'No incluida: {chr(34)}No{chr(34) in cols}')
"
```
**Esperado:**
```
Total cols: 35
No incluida: False
```

```bash
# 3. Smoke visual
streamlit run app/main.py
```

**Verificación visual:**

- ✅ Subir ISOVIDA, ir a Results → Surface (spatial).
- ✅ El heatmap **ya no incluye la columna `No`** como primera variable.
  Ahora empieza con `< 2 µm` (granulometría).
- ✅ Cambiar el dropdown de profundidad a 5 cm o 20 cm: el heatmap se
  redibuja también sin `No`.

---

## 5. Si algo falla

- Si los tests fallan → revisar que el cambio en `runner.py` no rompió
  ningún test que llame al pipeline (no debería: el filtro solo afecta
  qué columnas pasa al surface module, no toca el resto del flujo).
- Si el heatmap sigue mostrando `No` → verificar que `exclude_cols`
  está llegando al runner correctamente desde Upload (no debería
  cambiar; este flujo ya estaba probado).
- No tocar `aeda/engine/spatial_surface.py` (el módulo del backend está
  bien — el bug era de wiring, no de lógica).

---

## 6. Mensaje de commit sugerido

```
fix(surface): respect exclude_cols when running surface spatial analysis

Visual QA on ISOVIDA showed that the "No" column (row index) appeared
as the first column of the surface heatmap. Root cause: the pipeline
passed info.measurement_cols whole to surface_spatial_analysis without
filtering by the user's exclude_cols.

Fix applied in two places:
- aeda/pipeline/runner.py: filters info.measurement_cols by exclude_cols
  before calling the surface module.
- app/views/_surface_tab.py: replicates the same filter when the depth
  selector triggers a UI-side recompute, reading exclude_cols from the
  run_context cached in session state.

Validated: surface analysis now uses 35 chemistry columns on ISOVIDA
(previously 36, where the extra one was "No"). 71 tests still pass.
```
