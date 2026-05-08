# CODEX_PROMPT_FIX_DUPLICATED_SIDEBAR_MENU

**Tipo:** Bug fix (Streamlit convention conflict)
**Archivos:** 1 carpeta renombrada + 1 archivo modificado
**Tiempo estimado:** 2 minutos
**Tests esperados después:** 38 (sin cambios — esto solo renombra un módulo)

---

## 1. Contexto del problema

Tras correr `streamlit run app/main.py`, el sidebar muestra **dos menús de
navegación**: uno arriba con enlaces minúsculos sin estilo (`main`, `advanced`,
`audit`, `depth`, `plan`, `results`, `upload`) que no responden al click,
y debajo el menú propio de la app (con título 🔬 AEDA-AI y la lista correcta
de páginas).

### Causa

Streamlit tiene una convención de "Multipage Apps": **cualquier archivo
`.py` dentro de una carpeta llamada `pages/` ubicada al lado del script
principal se convierte automáticamente en una página de navegación**, y
Streamlit añade enlaces a cada una en el sidebar.

Como nuestras "páginas" no son scripts Streamlit independientes sino módulos
con una función `render()` que llamamos manualmente desde `app/main.py`, los
enlaces auto-generados apuntan a archivos que **no son ejecutables como
páginas**, lo cual hace que el clic no haga nada útil (y duplica visualmente
la navegación).

### Solución

Renombrar la carpeta `app/pages/` a `app/views/`. Streamlit solo reconoce
el nombre exacto `pages/`; cualquier otro nombre desactiva el descubrimiento
automático y el sidebar nativo desaparece. Nuestro menú propio (con la radio
del `_render_sidebar()` en `main.py`) queda como única navegación funcional.

---

## 2. Cambios

### 2.1 Renombrar la carpeta

En la raíz del repo:

```bash
git mv app/pages app/views
```

(O en VS Code: clic derecho sobre `app/pages` → Rename → `views`. Git lo
detecta como rename automáticamente.)

Si hay un directorio `__pycache__` dentro, se puede borrar — se regenera
solo en el próximo arranque.

### 2.2 Actualizar el registro de páginas en `app/main.py`

**BUSCAR:**

```python
PAGES = [
    ("Upload & Configure", "app.pages.upload"),
    ("Analysis Plan", "app.pages.plan"),
    ("Results", "app.pages.results"),
    ("Depth Profiles", "app.pages.depth"),
    ("Audit", "app.pages.audit"),
    ("Advanced Configuration", "app.pages.advanced"),
]
```

**REEMPLAZAR POR:**

```python
PAGES = [
    ("Upload & Configure", "app.views.upload"),
    ("Analysis Plan", "app.views.plan"),
    ("Results", "app.views.results"),
    ("Depth Profiles", "app.views.depth"),
    ("Audit", "app.views.audit"),
    ("Advanced Configuration", "app.views.advanced"),
]
```

**Nota importante:** este es el **único** sitio en todo el código donde se
referencian las páginas por path importable. Los archivos dentro de `app/views/*.py`
no se importan entre sí. No hace falta tocar nada más.

---

## 3. Validación

```bash
# 1. Imports limpios con el nuevo nombre
python -c "
import sys
sys.path.insert(0, '.')
import importlib
for m in ['app.main','app.views.upload','app.views.plan','app.views.results',
         'app.views.depth','app.views.audit','app.views.advanced']:
    importlib.import_module(m)
    print(f'OK  {m}')
"
```

**Esperado:** las 7 líneas con `OK`.

```bash
# 2. Tests siguen verdes (no debería haber cambios — esto es solo el shell)
pytest tests/ -q
```

**Esperado:** `38 passed`.

```bash
# 3. Verificación visual
streamlit run app/main.py
```

**Verificación visual en el navegador:**

- ✅ El sidebar muestra **un único menú**, el de la app (🔬 AEDA-AI con la
  radio de 6 páginas).
- ✅ El menú nativo de Streamlit con enlaces minúsculos (`main`, `advanced`,
  `audit`, etc.) **ya no aparece** arriba del logo.
- ✅ Hacer clic en cada opción de la radio cambia la página correctamente.

---

## 4. Si algo falla

- Si al arrancar Streamlit tira `ModuleNotFoundError: app.views.X` → la
  carpeta no se renombró. Verificar con `ls app/` que existe `views/` y no
  `pages/`.
- Si el menú nativo **sigue apareciendo** → revisar que no haya quedado un
  `app/pages/` vacío. Si quedó (por ejemplo porque el rename solo movió el
  contenido), borrarlo.
- No tocar nada bajo `aeda/` ni `tests/`. Esto es solo `app/`.

---

## 5. Mensaje de commit sugerido

```
fix(ui): rename app/pages -> app/views to disable Streamlit auto-routing

Streamlit treats any folder named `pages/` next to the main script as a
Multipage App and adds an auto-generated link for each .py file in the
sidebar. Our pages are not standalone Streamlit scripts; they are modules
with a render() function called manually from main.py, so the auto-links
were dead clicks duplicating the real (radio-based) navigation.

Renaming to `app/views/` opts us out of the convention. The custom sidebar
in main.py is now the single navigation surface.

Single import path updated in app/main.py. No engine changes; 38 tests pass.
```
