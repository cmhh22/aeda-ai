# HANDOFF — AEDA-AI ciclo 2

> **Para el Claude que lea esto al inicio de un chat nuevo:** este documento es tu briefing completo del proyecto. Léelo entero antes de la primera respuesta. Al final hay un placeholder para el feedback de Yoelvis que dispara este ciclo.

---

## 1. Quién es Eli

**Carlos Manuel Hernández.** Estudiante de Ciencias de la Computación, Universidad Central de Las Villas (UCLV), Cuba. Está cerrando su tesis de licenciatura.

**Estilo de trabajo con Claude:**

- Pragmático, directo. No le gustan los preámbulos largos ni los "voy a hacer X, Y, Z" antes de hacerlo.
- Escribe en español, a veces con typos y abreviaciones (eso es normal, no se lo señales). Responde en español neutro, sin modismos regionales.
- Prefiere respuestas concisas con bullets y tablas cuando aplica. Nada de markdown excesivo en respuestas conversacionales cortas.
- No programa él directamente — los cambios al código se aplican vía **GitHub Copilot/Codex en VS Code**. Por eso tu output principal son archivos `.md` con instrucciones precisas "BUSCAR / REEMPLAZAR" que él copia y pega a Codex.
- Validá todo lo que le entregás contra tu sandbox antes. No le mandes código sin testear: los `.md` con fixes deben venir con el resultado de `pytest tests/ -q` confirmado.
- Cuando estés tocando archivos grandes, escribí los `.md` con bloques de código exactos (con su indentación, comentarios, todo), no descripciones aproximadas.

---

## 2. Quién es Yoelvis

**Yoelvis Bolaños-Alvarez.** Ingeniero químico, científico en el LEA-CEAC (Laboratorio de Ensayos Ambientales, Centro de Estudios Ambientales de Cienfuegos). Es el tutor científico de la tesis.

**Importante sobre Yoelvis:**

- **Formación:** ingeniería química, no geoquímica. Tiene experiencia práctica en química ambiental y sedimentos por su trabajo en LEA, pero **su lente disciplinar es ingeniería química** — fuerte en química analítica, procesos, mediciones, instrumentación (XRF). Cuando le hablás de geoquímica usá términos generales, evitá jerga geológica avanzada (no asumas que conoce diagrámas ternarios, mineralogía petrológica, etc.).
- **No es programador.** Cualquier UI, documento técnico o explicación dirigida a él debe estar libre de jerga ML/software. Si Eli te pide un texto para Yoelvis, evitá términos como "kwargs", "pipeline", "auto-detect", "API", "session state". Usá lenguaje funcional: "el sistema configura automáticamente", "el análisis se ejecuta", etc.
- **Es la autoridad científica.** Las decisiones metodológicas que él tomó son cerradas (ver sección 4). No las reabras a menos que Eli explícitamente las cuestione.
- **No sabés su nivel de inglés.** El código, documentación técnica y la UI están en inglés (decisión de Eli para portafolio internacional). Pero los documentos de feedback, comunicación con Yoelvis y manuales suyos van en español.

---

## 3. Qué es AEDA-AI

**Análisis Exploratorio de Datos Ambientales con Inteligencia Artificial.**

Framework Python + Streamlit que automatiza el análisis exploratorio de datos de sedimentos geoquímicos. A partir de un Excel del laboratorio, genera:

- Validación y preprocesamiento adaptado al tipo de datos (composicionales, sesgados, unidades mixtas).
- Reducción dimensional (PCA / UMAP / t-SNE), clustering, detección de anomalías.
- Análisis espacial superficial (capa 0-10 cm) y por profundidad.
- Índices ambientales estándar: factor de enriquecimiento (Birch 2003) y clasificación toxicológica (NOAA TEL/PEL).
- Trazabilidad metodológica completa (cada decisión del sistema documentada).

**El "brain":** un sistema experto **rule-based**, no ML entrenado. Por decisión de diseño — para auditabilidad. Cada recomendación tiene `priority`, `confidence`, `reason`, `evidence`, `params`. Lo implementa `aeda/engine/auto_selector.py`.

**Dataset real de desarrollo:** ISOVIDA — 273 muestras de sedimentos de manglares de Bahía de Cienfuegos, ~52 variables (30 elementos químicos por XRF, granulometría, pérdidas por ignición), 7 sitios, profundidades 0-90 cm.

**Lo que NO hace el sistema (por diseño):**

- No clasifica un sitio como "contaminado" — solo provee índices objetivos. La interpretación final es del científico.
- No maneja censurados (<LOD/LOQ) — asume que la base ya los tiene resueltos.
- No reemplaza al geoquímico/ingeniero químico — es una herramienta de EDA, no un sustituto.

---

## 4. Decisiones científicas YA tomadas — NO reabrir

Estas decisiones fueron negociadas con el tutor en sesiones previas. Si Yoelvis las cuestiona en el feedback, Eli te lo dirá explícitamente.

| Decisión | Valor | Fundamento |
|---|---|---|
| CLR para granulometría | **opt-in** (apply_clr=False default) | Decisión metodológica de Yoelvis: el usuario debe habilitarla explícitamente |
| K en clustering primary | **K=7** | Validación geográfica con los 7 sitios de ISOVIDA |
| Threshold de correlación | **\|r\| ≥ 0.6** | Bajado de 0.7 a 0.6 en Tanda 1 |
| Baseline para EF | **deepest per-site** | Muestra más profunda de cada sitio como pre-contaminación |
| Elemento de referencia EF | **Al** (Aluminio) | Estándar Birch 2003; Fe es alternativa válida en Advanced |
| Crust reference | **Rudnick & Gao 2013** | Implementado en `aeda/engine/crust_reference.py` |
| Censurados | **No re-procesar** | La base ISOVIDA ya los tiene removidos por el laboratorio |
| Contamination rate Isolation Forest | **0.15** (auto-estimado por z-score) | Configurable en Advanced |
| Surface depth threshold | **10 cm default** (5/10/20 opciones) | Estándar para contaminación actual |

---

## 5. Estado del código (v4 — snapshot ciclo 1)

**Deployado en Streamlit Cloud:** `https://aeda-ai-lea-ceac.streamlit.app/`

**Repo GitHub:** privado, branch `main`. Eli pusheó la última versión consolidada (commit message: "fix(ui): QA pass 2").

**Arquitectura:**

```
aeda/                   ← Engine (rule-based brain + ML)
├── io/                 parsers, validators, preprocessor
│                       (parsers.py contiene la detección de coordenadas;
│                        Y como itrio NO se detecta como coordenada)
├── engine/             auto_selector, dimensionality, clustering,
│                       anomalies, correlations, feature_analysis,
│                       spatial_surface, crust_reference
├── interpretation/     EF (Birch), TEL/PEL (NOAA Buchman)
├── pipeline/           runner — orquestador
│                       (NOTA: aquí está el wiring que respeta K=7 de la
│                        recomendación primary, agregado en QA pass 1)
└── viz/                base, dimensionality, clustering, correlations,
                        profiles, interpretation

app/                    ← Streamlit UI
├── components/         page_header, errors, params (key-value renderer)
└── views/              upload, plan, results, depth, audit, advanced,
                        _surface_tab

tests/                  ← 71 tests, 8 archivos
data/                   ← ISOVIDA dataset incluido en el repo
```

**52 archivos Python · 71 tests verdes.**

---

## 6. Smoke test esperado contra ISOVIDA

Si todo está bien, este script en sandbox debe devolver exactamente esto:

```python
from aeda.pipeline.runner import AEDAPipeline
EXCLUDE = ['No','Code','Site_Name','Pret_Code','Código_muestra',
           'Sitio_muestreo','Fecha_muestreo','Core','Latitud','Longitud']
r = AEDAPipeline(impute_strategy='median').run(
    'data/BD_ISOVIDA_MANGLARES2023_rectificadaYBA_230326.xlsx',
    exclude_cols=EXCLUDE, sheet_name='DATA',
)
```

Resultado esperado:

```
Samples / measurement_cols: 273 / 37
coordinate_cols: ['Latitud', 'Longitud']
Y as measurement: True
K applied: 7
silhouette: ~0.369
spatial recs: 1
high_corr_pairs (|r|>0.6): 214
anomalies: 41
units loaded: 48 (Pb → mg/kg)
```

Si algo de esto no coincide, hay regresión — investigá antes de seguir.

---

## 7. Workflow de trabajo

1. Eli te describe el problema o pega feedback de Yoelvis.
2. Vos investigás en tu sandbox (`/home/claude/aeda-v4` después de extraer el RAR que él te pase).
3. Aplicás los fixes en sandbox.
4. Validás: `pytest tests/ -q` debe dar **71 passed** salvo que agregues tests nuevos.
5. Smoke test contra ISOVIDA con el script de la sección 6.
6. Empaquetás los cambios en **UN `.md` consolidado** con BUSCAR/REEMPLAZAR explícitos. Convención de nombres: `CODEX_PROMPT_<TIPO>_<DESCRIPCION>.md`.
7. Le pasás el `.md` a Eli, él lo pasa a Codex, Codex lo aplica, Eli pushea.

**Sobre el sandbox:** se resetea entre sesiones. Si llevás un rato sin tocarlo, probablemente tenés que volver a extraer el RAR. No asumas que `/home/claude/aeda-v4` existe — verificá antes.

**Sobre Streamlit Cloud:** el deploy es automático al push. Si Eli te dice que rompió en producción pero no en local, suele ser una dependencia faltante en `requirements.txt`.

---

## 8. Resumen del ciclo 1 (qué se hizo)

| Fase | Output principal |
|---|---|
| Tanda 1 Yoelvis | Heavy metals expandidos, CLR opt-in, threshold 0.6, units desde Excel, crust reference Rudnick & Gao 2013 |
| Tanda 2 Surface | Módulo `spatial_surface.py`, UI con heatmap z-score + mapa, integración al brain (categoría `spatial`) |
| QA Pass 1 (consolidado) | 10 fixes: Y como itrio, K=7 wiring brain↔engine, categoría Spatial visible en UI, label 0.6, labels Variables clarificados, leyendas no truncadas, params render amigable, captions cosmeticos |
| QA Pass 2 | Texto duplicado en Plan, Results tab adaptativo PCA/UMAP/t-SNE, captions interpretativos en cada gráfica, simplificación grid Depth Profiles |
| Deploy + Manual | Streamlit Cloud productivo, Manual Word de 2 páginas para Yoelvis |

**Archivos .md históricos en el repo** (no aplicarlos de nuevo, solo referencia):
- `HANDOFF_TANDA1_YOELVIS.md`
- `CODEX_PROMPT_TANDA2_SURFACE_UI.md`
- `CODEX_PROMPT_FIX_SURFACE_EXCLUDE_COLS.md`
- `CODEX_PROMPT_QA_CONSOLIDATED_FIXES.md`
- `CODEX_PROMPT_QA_PASS2_BUGS_AND_CAPTIONS.md`

---

## 9. Estilo de tus `.md` para Codex

Plantilla mínima que funciona bien:

```markdown
# CODEX_PROMPT_<NOMBRE>

**Tipo:** <fix / feature / refactor>
**Archivos:** <N modificados, N nuevos>
**Tests esperados después:** 71 passed

## 1. Contexto

[Por qué este cambio. 1-2 párrafos.]

## 2. FIX #1 — <título corto>

**Archivo:** `path/al/archivo.py`

**Causa:** [explicación del bug/feature]

**BUSCAR:**

[bloque exacto, con indentación y comentarios]

**REEMPLAZAR POR:**

[bloque exacto, con indentación y comentarios]

## N. Validación

```bash
pytest tests/ -q
```

Esperado: 71 passed.

## N+1. Mensaje de commit sugerido

[Mensaje en formato conventional commits]
```

---

## 10. Pendientes conocidos (no urgentes)

- **Exportación de reportes PDF/Word integrada:** workaround actual usar ícono cámara de plotly. Yoelvis puede pedirlo.
- **Manual técnico-científico CENDA:** pendiente, se hará después del feedback.
- **Capítulos de tesis:** Eli puede pedirte revisión en cualquier momento.
- **Internacionalización UI:** la app está en inglés, eventualmente puede haber una versión en español.

---

## 11. Feedback de Yoelvis — input para este ciclo

> **Eli: pegá acá el feedback que recibiste antes del primer mensaje del chat nuevo.**

```
[Feedback recibido — pegar aquí]
```

Una vez tengas el feedback, separalo en estos buckets antes de empezar:

- **Bugs reales** (algo no funciona / da error / resultado incorrecto)
- **Confusiones** (texto poco claro / etiquetas ambiguas / controles sin propósito claro)
- **Faltas** (análisis o visualizaciones que considera esenciales y no están)
- **Sobras** (ruido / información que distrae)
- **Sugerencias** (mejoras, features nuevos, cambios de orden)

Trabajá los buckets en ese orden. Bugs primero (rápido, alto impacto). Las sugerencias eventualmente las priorizás según el tiempo restante hasta la defensa.

---

## 12. Cómo arrancar el chat nuevo

Tu primer mensaje, Eli, debería ser algo así:

> "Te paso el handoff del ciclo 1 y el RAR de v4 del proyecto. Léelo entero, después te paso el feedback de Yoelvis."

Esperá a que el Claude nuevo confirme que leyó todo. Después pasale el feedback. Después pasale el RAR. Empezás a iterar.

**Si el Claude nuevo asume cosas que contradicen este handoff** (por ejemplo, asume que Yoelvis es geoquímico, o intenta re-debatir CLR), corregíle directo apuntando al handoff.

---

*Fin del handoff. Buena suerte con el ciclo 2.*

— Documento generado al cierre del ciclo 1, mayo 2026.
