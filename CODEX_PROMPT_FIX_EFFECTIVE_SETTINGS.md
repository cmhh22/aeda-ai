# CODEX_PROMPT_FIX_EFFECTIVE_SETTINGS

**Tipo:** Bug fixes en el engine (effective settings + contamination del plan)
**Archivos:** 3 modificados
**Tiempo estimado:** ~20 min
**Tests esperados después:** 38 passed (sin cambios en número)

---

## 1. Contexto

Dos bugs relacionados detectados durante el QA visual:

### Bug A — CLR inconsistente en Advanced Configuration
**Síntoma:** La página Advanced muestra "CLR transform: off" aunque el
pipeline efectivamente aplicó CLR (porque el plan lo recomendó como
Primary con high confidence).

**Causa:** El `run_context` que persiste Upload contiene el dict `settings`
construido a partir de los inputs del formulario, no los valores que el
auto-selector resolvió realmente (`effective_*` en `runner.py`). El usuario
ve lo que escribió, no lo que pasó.

### Bug B — Contamination del plan no se aplica
**Síntoma:** El plan recomienda `contamination=0.15` (15% basado en
z-score outliers), pero `detect_anomalies` corre con `0.05` (default
hardcoded).

**Causa:** El constructor `AEDAPipeline(contamination=0.05)` usa un default
fijo que no consulta el plan. La recomendación del auto-selector queda
inerte: se muestra al usuario pero nunca se aplica.

### Solución unificada

Ambos bugs comparten la misma raíz: los parámetros "efectivos" (los que
el sistema resolvió desde el plan) no estaban expuestos. Este prompt:

1. Agrega un campo `effective_settings: dict` al dataclass `AEDAResults`
   que persiste **todos** los parámetros resueltos.
2. Cambia `contamination: float = 0.05` a `Optional[float] = None`, de
   modo que `None` significa "delegar al plan" y un valor explícito
   significa "el usuario sobreescribe".
3. Las páginas Upload y Advanced leen `results.effective_settings` en
   lugar de los inputs del formulario al armar el `run_context`.
4. Upload pasa `apply_clr="auto"` explícitamente al constructor para
   delegar la decisión al plan (el default sigue siendo `False` para
   no romper código existente que crea `AEDAPipeline()` directo).

**Validado contra ISOVIDA:** después del fix, anomaly detection produce
41 anomalías (15% del dataset, como recomienda el plan), y `apply_clr`
efectivo es `True`.

---

## 2. Cambios en `aeda/pipeline/runner.py`

Tres cambios localizados.

### 2.1 Agregar `effective_settings` al dataclass `AEDAResults`

**BUSCAR:**

```python
    # Environmental interpretation (EF, TEL/PEL, Birch)
    interpretation: Optional[InterpretationReport] = None

    def summary(self) -> str:
```

**REEMPLAZAR POR:**

```python
    # Environmental interpretation (EF, TEL/PEL, Birch)
    interpretation: Optional[InterpretationReport] = None

    # Resolved parameters actually used in this run (after the auto-selector
    # has resolved any "auto" values from the plan). This is what the UI
    # reads to display the *effective* configuration in the Advanced page,
    # not the user input.
    effective_settings: Optional[dict] = None

    def summary(self) -> str:
```

### 2.2 Cambiar `contamination` default a `Optional[float] = None`

**BUSCAR:**

```python
        apply_clr: bool | str | None = False,
        contamination: float = 0.05,
```

**REEMPLAZAR POR:**

```python
        apply_clr: bool | str | None = False,
        contamination: Optional[float] = None,
```

### 2.3 Resolver `contamination` desde el plan + persistir effective_settings

**BUSCAR:**

```python
        # Final fallbacks for any remaining "auto" value
        if effective_scale == "auto":
            effective_scale = "standard"
        if effective_impute == "auto":
            effective_impute = "median"
        if effective_clr == "auto":
            effective_clr = False

        # 4. PREPROCESSING
```

**REEMPLAZAR POR:**

```python
        # Final fallbacks for any remaining "auto" value
        if effective_scale == "auto":
            effective_scale = "standard"
        if effective_impute == "auto":
            effective_impute = "median"
        if effective_clr == "auto":
            effective_clr = False

        # Resolve contamination: if the user did not set it explicitly, use
        # the value recommended by the auto-selector (primary anomaly recom-
        # mendation). Falls back to 0.05 if no recommendation is available.
        effective_contamination = self.contamination
        if effective_contamination is None:
            for rec in plan.recommendations:
                if rec.category == "anomaly" and rec.priority == 1:
                    candidate = rec.params.get("contamination")
                    if candidate is not None:
                        effective_contamination = float(candidate)
                    break
            if effective_contamination is None:
                effective_contamination = 0.05

        # Persist the resolved configuration so the UI (Advanced page) can
        # display what was actually applied, not what the user typed.
        results.effective_settings = {
            "scale_method": effective_scale,
            "impute_strategy": effective_impute,
            "apply_clr": effective_clr,
            "contamination": effective_contamination,
            "dim_method": self.dim_method,
            "clustering_method": self.clustering_method,
            "anomaly_method": self.anomaly_method,
            "correlation_method": self.correlation_method,
            "run_interpretation": self.run_interpretation,
            "reference_element": self.reference_element,
            "baseline_strategy": self.baseline_strategy,
            "custom_baseline": self.custom_baseline,
            "dim_kwargs": dict(self.dim_kwargs),
            "clustering_kwargs": dict(self.clustering_kwargs),
            "anomaly_kwargs": dict(self.anomaly_kwargs),
        }

        # 4. PREPROCESSING
```

### 2.4 Usar `effective_contamination` en `detect_anomalies`

**BUSCAR:**

```python
        # 7. ANOMALY DETECTION
        try:
            results.anomalies = detect_anomalies(
                processed,
                method=self.anomaly_method,
                contamination=self.contamination,
                **self.anomaly_kwargs,
            )
```

**REEMPLAZAR POR:**

```python
        # 7. ANOMALY DETECTION
        try:
            results.anomalies = detect_anomalies(
                processed,
                method=self.anomaly_method,
                contamination=effective_contamination,
                **self.anomaly_kwargs,
            )
```

---

## 3. Cambios en `app/views/upload.py`

### 3.1 Pasar `apply_clr="auto"` al pipeline

Esto delega la decisión de aplicar CLR al plan del auto-selector, que es
exactamente lo que el usuario espera al subir desde Streamlit (no se le
pregunta sobre CLR en el formulario).

**BUSCAR:**

```python
        pipeline = AEDAPipeline(
            impute_strategy=impute,
            dim_method=dim_method,
            clustering_method=cluster_method,
        )
```

**REEMPLAZAR POR:**

```python
        pipeline = AEDAPipeline(
            impute_strategy=impute,
            dim_method=dim_method,
            clustering_method=cluster_method,
            apply_clr="auto",
        )
```

### 3.2 Persistir `effective_settings` en el `run_context`

**BUSCAR:**

```python
        # Store in session state
        st.session_state.results = results
        st.session_state.raw_df = results.raw_data
        st.session_state.filename = filename
        st.session_state.run_context = {
            "tmp_path": filepath,
            "sheet_name": sheet_name,
            "exclude_cols": exclude_cols,
            "settings": settings,
        }
```

**REEMPLAZAR POR:**

```python
        # Store in session state
        st.session_state.results = results
        st.session_state.raw_df = results.raw_data
        st.session_state.filename = filename
        st.session_state.run_context = {
            "tmp_path": filepath,
            "sheet_name": sheet_name,
            "exclude_cols": exclude_cols,
            # Persist the *effective* settings (what the auto-selector actually
            # resolved and applied), not the raw form inputs. This is what the
            # Advanced page reads to pre-fill its controls.
            "settings": results.effective_settings or settings,
        }
```

---

## 4. Cambios en `app/views/advanced.py`

Que Advanced también persista los effective_settings al re-correr (porque
la re-corrida también pasa por el auto-selector y puede resolver "auto" a
valores concretos).

**BUSCAR:**

```python
        # Update session state with new results and the settings used for this run.
        st.session_state.results = results
        st.session_state.raw_df = results.raw_data
        st.session_state.run_context = {
            "tmp_path": ctx["tmp_path"],
            "sheet_name": ctx.get("sheet_name"),
            "exclude_cols": ctx.get("exclude_cols"),
            "settings": settings,
        }
```

**REEMPLAZAR POR:**

```python
        # Update session state with new results and the settings used for this run.
        st.session_state.results = results
        st.session_state.raw_df = results.raw_data
        st.session_state.run_context = {
            "tmp_path": ctx["tmp_path"],
            "sheet_name": ctx.get("sheet_name"),
            "exclude_cols": ctx.get("exclude_cols"),
            # Persist the effective settings actually resolved by the pipeline,
            # not what the user typed — that way the next page load reflects
            # what really ran (e.g. apply_clr=True when the plan recommended it).
            "settings": results.effective_settings or settings,
        }
```

---

## 5. Validación

```bash
# 1. Tests siguen verdes
pytest tests/ -q
```
**Esperado:** `38 passed`. (Si alguno falla, **detenerse y reportar** — algo
del cambio no se aplicó como esperaba.)

```bash
# 2. Smoke test funcional con ISOVIDA
python -c "
from aeda.pipeline.runner import AEDAPipeline
EXCLUDE = ['No','Code','Site_Name','Pret_Code','Código_muestra','Sitio_muestreo',
           'Fecha_muestreo','Core','Latitud','Longitud']
r = AEDAPipeline(apply_clr='auto').run(
    'data/BD_ISOVIDA_MANGLARES2023_rectificadaYBA_230326.xlsx',
    exclude_cols=EXCLUDE, sheet_name='DATA',
)
print(f'apply_clr efectivo:     {r.effective_settings[\"apply_clr\"]}')
print(f'contamination efectiva: {r.effective_settings[\"contamination\"]}')
print(f'anomalias:              {r.anomalies.n_anomalies}/{r.raw_data.shape[0]}')
"
```

**Esperado:**

```
apply_clr efectivo:     True
contamination efectiva: 0.15
anomalias:              41/273
```

```bash
# 3. Validación visual: re-subir ISOVIDA en Streamlit
streamlit run app/main.py
```

**Verificación visual:**

- ✅ Tras correr el análisis, en **Audit → Decisions**, la categoría "Sample
  grouping" sigue mostrando lo del plan. **Anomaly detection** ahora reporta
  realmente lo que el plan recomendó (no el default 0.05).
- ✅ En **Results → KPI bar** debería verse algo cercano a 41 anomalías en lugar
  de 14.
- ✅ En **Advanced Configuration**:
  - "CLR transform (compositional)" muestra "on" (ya no "off").
  - "Anomaly contamination rate" muestra 0.15 (ya no 0.05).
  - El resto de los controles refleja lo que efectivamente corrió.

---

## 6. Si algo falla

- Si los tests existentes empiezan a fallar → revisar que `apply_clr` default
  en el constructor sigue siendo `False`, no `"auto"`. Solo Upload debe
  pasar `"auto"` explícitamente. El default tiene que quedarse en `False`
  para no romper tests que usan datasets sintéticos con NaN en granulometría.
- Si Advanced muestra `None` o un dict vacío como settings → el fallback
  `results.effective_settings or settings` no encontró ni uno ni otro;
  revisar que el cambio 2.3 (la asignación de `results.effective_settings`)
  esté antes del bloque de preprocessing en `run()`.
- Si Streamlit crashea con `AttributeError: 'AEDAResults' object has no
  attribute 'effective_settings'` → el cambio 2.1 (agregar el campo al
  dataclass) no se aplicó.
- No tocar `tests/`, `aeda/engine/`, `aeda/interpretation/`. Solo
  `aeda/pipeline/runner.py`, `app/views/upload.py`, `app/views/advanced.py`.

---

## 7. Mensaje de commit sugerido

```
fix(engine): expose effective_settings and respect plan's contamination

Two related bugs surfaced during the ISOVIDA QA pass:

- Advanced Configuration page showed "CLR transform: off" even when the
  pipeline had effectively applied CLR (because the plan recommended it).
  Root cause: run_context persisted the form inputs, not the resolved
  parameters.

- The plan's recommended contamination (0.15 for ISOVIDA, based on z-score
  outlier rate) was never applied — detect_anomalies always ran with the
  constructor default 0.05.

Fix:
- AEDAResults now exposes an effective_settings dict containing every
  parameter the auto-selector actually resolved (apply_clr, contamination,
  scale_method, impute_strategy, ...). The UI reads this instead of the
  raw form inputs.
- AEDAPipeline contamination default changed from 0.05 to Optional[None].
  None means "use the plan's recommendation"; an explicit float overrides
  the plan. Backwards-compatible for callers that passed contamination
  explicitly.
- app/views/upload.py now passes apply_clr="auto" to the constructor so
  the plan decides whether to apply CLR.

Validated against ISOVIDA: 41 anomalies (15.0%) detected, matching the
plan's recommendation. apply_clr effective = True. 38 tests pass.
```
