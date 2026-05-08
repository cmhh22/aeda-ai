# CODEX_PROMPT_FIX_RESULTS_AND_AUDIT_ERRORS

**Tipo:** Bug fixes (2 errores reportados al correr la app)
**Archivos:** 2 modificados (`pyproject.toml` y `app/views/audit.py`)
**Tiempo estimado:** 3 minutos
**Tests esperados después:** 38 (sin cambios)

---

## 1. Contexto

Tras aplicar los prompts UI_FOUNDATION + UI_PAGES_POLISH y correr la app
sobre el dataset ISOVIDA, aparecen dos errores:

### Error A — `Results` page

```
ImportError: `Import matplotlib` failed. Styler.background_gradient requires
matplotlib. Use pip or conda to install the matplotlib package.
```

**Causa:** `app/views/results.py` usa `loadings.style.background_gradient(...)`
para colorear la tabla de loadings de PCA. Pandas delega esa coloración
en matplotlib, que no está declarado como dependencia del proyecto.

### Error B — `Audit` page

```
NameError: name '_render_overview' is not defined
File "app/views/audit.py", line 40, in render
    _render_overview(results)
```

**Causa:** El código actual de `audit.py` quedó en un estado mixto: el
prompt anterior reorganizaba la página en 4 tabs llamando a las funciones
ya existentes (`_render_run_summary`, `_render_validation`, etc.), pero
en la versión que hay en el repo Codex cambió las llamadas a nombres
nuevos (`_render_overview`, etc.) sin definir esas funciones. Hay que
restaurar las llamadas correctas a las funciones que sí existen.

---

## 2. Fix A — agregar `matplotlib` como dependencia

### 2.1 Editar `pyproject.toml`

Localizar la sección `[project]` con la lista `dependencies = [...]` y
agregar `"matplotlib>=3.7"` al final de la lista (manteniendo el resto
intacto).

**Ejemplo de cómo debe quedar la lista (los nombres exactos de las otras
dependencias pueden variar — solo añadir la nueva línea, no tocar las
demás):**

```toml
dependencies = [
    "pandas>=2.0",
    "numpy>=1.24",
    "scipy>=1.10",
    "scikit-learn>=1.3",
    "openpyxl>=3.1",
    "plotly>=5.17",
    "streamlit>=1.28",
    "matplotlib>=3.7",
]
```

### 2.2 Reinstalar el proyecto

Desde la raíz del repo, con el venv activado:

```bash
pip install -e . --upgrade
```

Eso instalará matplotlib y deja todo lo demás como estaba.

---

## 3. Fix B — `app/views/audit.py`: restaurar las llamadas correctas

### 3.1 Localizar la función `render()` al inicio del archivo

Aproximadamente entre las líneas 22 y 60. El bloque que crea los 4 tabs.

### 3.2 BUSCAR el bloque de tabs que está roto

Buscar la sección dentro de `render()` que llama a funciones inexistentes
(probablemente `_render_overview`, `_render_decisions`, `_render_interp`,
`_render_tech` o similar). Reemplazar **todo** ese bloque desde
`tab_overview, tab_decisions, ...` hasta el final del último `with tab_tech:`.

### 3.3 REEMPLAZAR POR el siguiente bloque exacto

```python
    tab_overview, tab_decisions, tab_interp, tab_tech = st.tabs([
        "Overview",
        "Decisions",
        "Interpretation",
        "Technical",
    ])

    with tab_overview:
        _render_run_summary(results)
        st.divider()
        _render_validation(results)

    with tab_decisions:
        _render_brain_decisions(results)
        st.divider()
        _render_preprocessing_trace(results)

    with tab_interp:
        _render_interpretation_audit(results)

    with tab_tech:
        _render_failures(results)
        _render_ml_metrics(results)
```

### 3.4 Verificar que las funciones existen en el archivo

Las siguientes funciones **deben estar definidas** más abajo en el mismo
archivo (no son nuevas; ya estaban antes del polish):

- `_render_run_summary(results)`
- `_render_validation(results)`
- `_render_brain_decisions(results)`
- `_render_preprocessing_trace(results)`
- `_render_interpretation_audit(results)`
- `_render_failures(results)`
- `_render_ml_metrics(results)`

Si alguna de estas funciones **no existe** en el archivo (porque Codex la
renombró por error), reportarlo y detenerse — habría que recuperar la
versión completa del archivo desde el prompt UI_PAGES_POLISH. Si todas
existen, el fix está completo con solo el cambio del bloque de `render()`.

---

## 4. Validación

```bash
# 1. La suite de tests sigue verde
pytest tests/ -q
```

**Esperado:** `38 passed`.

```bash
# 2. La app arranca sin errores
streamlit run app/main.py
```

**Verificación visual:**

- ✅ Subir el dataset ISOVIDA y correr el análisis.
- ✅ Ir a **Results → PCA**: la tabla de loadings se muestra con un gradiente
  de color (rojo → azul) sin el `ImportError`. Si la tabla aparece sin
  color en algunas celdas pero sin error, está bien.
- ✅ Ir a **Audit**: la página renderiza con 4 tabs (Overview, Decisions,
  Interpretation, Technical) sin el `NameError`. Cada tab muestra contenido
  al hacer clic.

---

## 5. Si algo falla

- Si después del fix de matplotlib aparece `ModuleNotFoundError: No module
  named 'matplotlib'` → la reinstalación de `pip install -e .` no se
  completó. Correr `pip install matplotlib` directamente y reintentar.
- Si Audit sigue tirando `NameError` pero por una función distinta a
  `_render_overview` → significa que más llamadas dentro de las funciones
  privadas también fueron renombradas. Reportar el nuevo NameError y se
  resuelve con un solo replace adicional.
- No tocar `aeda/`, `tests/` ni nada bajo `app/` que no sean los dos
  archivos mencionados.

---

## 6. Mensaje de commit sugerido

```
fix(ui): missing matplotlib dependency and broken audit tab callbacks

Two runtime errors reported on the Streamlit app:

- Results > PCA crashed with ImportError because pandas Styler's
  background_gradient delegates color rendering to matplotlib, which was
  not declared as a project dependency. Added matplotlib>=3.7 to
  pyproject.toml.

- Audit page crashed with NameError because the tab dispatch in render()
  was calling non-existent functions (_render_overview, etc.) instead of
  the real ones already defined in the file (_render_run_summary,
  _render_validation, _render_brain_decisions, _render_preprocessing_trace,
  _render_interpretation_audit, _render_failures, _render_ml_metrics).
  Restored the correct dispatch.

No engine changes; 38 tests pass.
```
