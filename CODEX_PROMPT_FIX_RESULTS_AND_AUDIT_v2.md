# CODEX_PROMPT_FIX_RESULTS_AND_AUDIT_v2

**Tipo:** Bug fixes (re-aplicación con instrucciones más explícitas)
**Archivos:** 2 modificados (`pyproject.toml`, `app/views/audit.py`)
**Tiempo estimado:** 5 minutos

---

## 1. Contexto

Los dos errores reportados anteriormente persisten:

- **Results page:** `ImportError: Styler.background_gradient requires matplotlib`
- **Audit page:** `NameError: name '_render_overview' is not defined`

Para el primero, la solución anterior agregaba `matplotlib` al `pyproject.toml`
pero **eso no instala el paquete** — solo lo declara como dependencia. Hay que
instalarlo explícitamente. Y conviene también agregar `seaborn`, que es la
otra biblioteca de visualización que pandas/scikit suelen usar para gradientes.

Para el segundo, el error es **literalmente el mismo**, lo cual significa que
o bien el cambio no llegó al archivo, o el servidor Streamlit está sirviendo
una versión cacheada. Hay que verificar el estado actual del archivo y
reaplicar el fix con más cuidado.

---

## 2. Fix A — Instalar matplotlib y seaborn

### 2.1 Agregar a `pyproject.toml`

Localizar la sección `[project]` con la lista `dependencies = [...]`. Agregar
estas dos líneas (si ya existe `matplotlib`, dejarla; si ya existe `seaborn`,
dejarla):

```toml
    "matplotlib>=3.7",
    "seaborn>=0.12",
```

### 2.2 INSTALAR los paquetes (este es el paso que faltó la vez pasada)

Desde la terminal de VS Code, en la raíz del proyecto, con el venv activado:

```bash
pip install matplotlib seaborn
```

**Verificar que se instaló correctamente** corriendo:

```bash
python -c "import matplotlib; import seaborn; print('matplotlib', matplotlib.__version__); print('seaborn', seaborn.__version__)"
```

Esperado: imprime las dos versiones sin error.

### 2.3 Reiniciar el servidor Streamlit

**Esto es importante.** Si Streamlit está corriendo, los nuevos paquetes no
se cargan hasta reiniciar:

1. En la terminal donde corre `streamlit run`, presionar `Ctrl+C`.
2. Volver a correr `streamlit run app/main.py`.

---

## 3. Fix B — Verificar y reaplicar el fix de Audit

El error `NameError: name '_render_overview' is not defined` está en la
línea 40 de `app/views/audit.py`. Eso indica que la función `render()`
todavía contiene una llamada a `_render_overview` que no existe.

### 3.1 PRIMER PASO: leer el archivo actual

Abrir `app/views/audit.py` en VS Code y mirar las primeras 60 líneas. Buscar
la función `render()` y ver qué llamadas hace dentro de los `with tab_X:`.

Si dentro del bloque de tabs aparecen llamadas a funciones con nombres como
`_render_overview`, `_render_decisions`, `_render_interp`, `_render_tech`,
`_render_summary`, etc. → **esas son las llamadas rotas**.

### 3.2 SEGUNDO PASO: reemplazar todo el bloque

Localizar la línea que comienza con:

```python
    tab_overview, tab_decisions, tab_interp, tab_tech = st.tabs([
```

Esa línea está dentro de la función `render()`. Desde esa línea, hacia abajo,
hay un bloque de tabs que termina en algo como:

```python
    with tab_tech:
        _render_tech(results)        # o cualquier otro nombre roto
```

**Seleccionar todo el bloque desde `tab_overview, tab_decisions, ...` hasta
el cierre del último `with tab_tech:` (incluido)** y reemplazarlo por el
siguiente bloque exacto:

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

Las **siete funciones** que aparecen en este bloque deben existir definidas
más abajo en el mismo archivo:

- `_render_run_summary`
- `_render_validation`
- `_render_brain_decisions`
- `_render_preprocessing_trace`
- `_render_interpretation_audit`
- `_render_failures`
- `_render_ml_metrics`

### 3.3 TERCER PASO: verificar que las funciones existen

Después de reemplazar el bloque, hacer `Ctrl+F` en VS Code y buscar cada uno
de esos siete nombres dentro del archivo. Cada uno debe tener al menos **dos
matches**: uno en la llamada (dentro del bloque que acabamos de reemplazar)
y otro en la definición (`def _render_X(results):` más abajo).

**Si alguna función NO aparece definida** (solo aparece la llamada, sin un
`def` correspondiente), reportarlo de inmediato. Eso significa que el polish
anterior no se aplicó completo y hay que recuperar el contenido completo del
archivo del prompt UI_PAGES_POLISH.md.

### 3.4 CUARTO PASO: reiniciar Streamlit

Igual que en Fix A: `Ctrl+C` en la terminal donde corre Streamlit y volver
a lanzarlo con `streamlit run app/main.py`.

---

## 4. Validación

```bash
# 1. Tests siguen verdes
pytest tests/ -q
```
**Esperado:** `38 passed`.

```bash
# 2. Imports limpios
python -c "
import sys; sys.path.insert(0, '.')
import importlib
for m in ['app.main', 'app.views.audit', 'app.views.results']:
    importlib.import_module(m)
    print(f'OK  {m}')
"
```
**Esperado:** las 3 líneas con `OK`.

```bash
# 3. Smoke visual
streamlit run app/main.py
```

**Verificación en el navegador (después de subir ISOVIDA y correr análisis):**

- ✅ **Results → PCA tab:** la tabla de loadings se renderiza con gradiente
  de color rojo→azul, sin ImportError.
- ✅ **Audit:** la página renderiza con 4 tabs (Overview, Decisions,
  Interpretation, Technical) sin NameError. Cada tab muestra contenido
  real al hacer clic.

---

## 5. Si Audit sigue fallando con NameError

Si después de aplicar los 4 pasos del Fix B, la página Audit **sigue tirando
el mismo NameError**, eso indica que el archivo `audit.py` está incompleto
(le faltan las definiciones de algunas de las 7 funciones privadas). En ese
caso:

1. **Detenerse y reportar** cuáles funciones de la lista NO aparecen
   definidas en el archivo (es decir, no tienen un `def _render_X(results):`
   correspondiente).
2. Dado ese reporte, se generará un prompt nuevo con el archivo `audit.py`
   completo para reemplazarlo entero.

---

## 6. Mensaje de commit sugerido

```
fix(ui): install matplotlib/seaborn and restore audit tab callbacks

- Added matplotlib>=3.7 and seaborn>=0.12 to pyproject.toml AND installed
  them in the venv. Pandas Styler.background_gradient (used in the PCA
  loadings table) requires matplotlib at runtime.
- Restored the correct function names in app/views/audit.py render()
  tab dispatch. Previously calling _render_overview which does not exist;
  now calling the actual functions (_render_run_summary, _render_validation,
  _render_brain_decisions, _render_preprocessing_trace,
  _render_interpretation_audit, _render_failures, _render_ml_metrics).

No engine changes; 38 tests pass.
```
