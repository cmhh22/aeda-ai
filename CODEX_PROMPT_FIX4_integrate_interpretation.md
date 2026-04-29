# GAP 1 — Integrar el módulo de interpretación en el pipeline

## Contexto

El módulo `aeda/interpretation/` (EF, TEL/PEL, Birch, LOD) está implementado y testeado, pero el orquestador `AEDAPipeline.run()` no lo invoca. El dataclass `AEDAResults` no tiene un campo `interpretation`, así que aunque el código existe, la interfaz web nunca lo expone.

Este fix integra el módulo de interpretación como una etapa más del pipeline, ejecutada automáticamente cuando se cumplen los pre-requisitos: hay metales pesados detectados, hay un elemento de referencia disponible, y hay columna de profundidad.

**Pre-requisito:** este prompt debe aplicarse DESPUÉS del fix del Bug 3 (alineación de HEAVY_METALS), porque la lista de metales que el cerebro detecta es ahora la correcta.

## Cambios requeridos

### 1. Modificar `aeda/pipeline/runner.py`

**1.a — Agregar import** al principio del archivo, junto a los otros imports del paquete `aeda`:

```python
from aeda.interpretation import (
    InterpretationReport,
    build_interpretation_report,
)
```

**1.b — Agregar campo `interpretation` al dataclass AEDAResults:**

En la sección de campos del dataclass `AEDAResults`, después del campo `feature_importance`, agregar:

```python
    # Environmental interpretation (EF, TEL/PEL, Birch)
    interpretation: Optional[InterpretationReport] = None
```

**1.c — Extender el método `summary` de AEDAResults** para incluir la interpretación. Antes de la línea final `lines.append("=" * 60)`, agregar:

```python
        if self.interpretation:
            lines.append(f"\nEnvironmental interpretation:")
            lines.append(f"  Metals analyzed: {self.interpretation.metals_analyzed}")
            if self.interpretation.ef_result is not None:
                lines.append(f"  Reference element: {self.interpretation.ef_result.reference_element}")
                lines.append(f"  Baseline strategy: {self.interpretation.ef_result.baseline_strategy}")
            tel_pel_total = sum(
                self.interpretation.tel_pel_classifications[m].notna().sum()
                for m in self.interpretation.metals_analyzed
                if m in self.interpretation.tel_pel_classifications.columns
            )
            lines.append(f"  TEL/PEL classifications computed: {int(tel_pel_total)}")
        else:
            lines.append("\nEnvironmental interpretation: not run (no eligible metals or reference element)")
```

**1.d — Extender la firma del constructor de `AEDAPipeline`** para aceptar parámetros del módulo de interpretación. Reemplazar la firma actual:

```python
    def __init__(
        self,
        scale_method: str = "auto",
        impute_strategy: str = "auto",
        dim_method: str = "auto",
        clustering_method: str = "auto",
        anomaly_method: str = "auto",
        correlation_method: str = "compare",
        apply_clr: bool | str | None = False,
        contamination: float = 0.05,
    ):
```

por:

```python
    def __init__(
        self,
        scale_method: str = "auto",
        impute_strategy: str = "auto",
        dim_method: str = "auto",
        clustering_method: str = "auto",
        anomaly_method: str = "auto",
        correlation_method: str = "compare",
        apply_clr: bool | str | None = False,
        contamination: float = 0.05,
        # Environmental interpretation parameters
        run_interpretation: bool = True,
        reference_element: str = "Al",
        baseline_strategy: str = "deepest",
        custom_baseline: Optional[dict] = None,
    ):
```

Y dentro del cuerpo del `__init__`, después de las asignaciones existentes, agregar:

```python
        self.run_interpretation = run_interpretation
        self.reference_element = reference_element
        self.baseline_strategy = baseline_strategy
        self.custom_baseline = custom_baseline
```

**1.e — Agregar la etapa 10 al pipeline** (interpretación ambiental). Después del bloque "9. FEATURE IMPORTANCE" y antes del `return results`, insertar:

```python
        # 10. ENVIRONMENTAL INTERPRETATION (EF, TEL/PEL, Birch)
        if self.run_interpretation:
            try:
                results.interpretation = self._run_interpretation(
                    raw_df=df,
                    info=info,
                    plan=plan,
                )
            except Exception as e:
                logger.warning(
                    f"Environmental interpretation failed: {type(e).__name__}: {e}"
                )
                results.interpretation = None
```

**1.f — Agregar el método helper `_run_interpretation`** al final de la clase `AEDAPipeline` (después del método `run`, dentro de la misma clase):

```python
    def _run_interpretation(
        self,
        raw_df: pd.DataFrame,
        info: DatasetInfo,
        plan: AnalysisPlan,
    ) -> Optional[InterpretationReport]:
        """Run the environmental interpretation layer if pre-requisites are met.

        Pre-requisites:
        - At least one heavy metal detected by the brain.
        - The reference element is present in the raw dataset.
        - A depth column is available (required for the deepest baseline strategy).

        If any pre-requisite is missing, this returns None and logs a debug message.
        """
        metals = [m for m in plan.profile.heavy_metal_cols if m in raw_df.columns]

        if not metals:
            logger.debug("Skipping interpretation: no heavy metals detected.")
            return None

        if self.reference_element not in raw_df.columns:
            logger.debug(
                f"Skipping interpretation: reference element "
                f"'{self.reference_element}' not in dataset."
            )
            return None

        # The reference element cannot also be one of the metals being analyzed.
        metals = [m for m in metals if m != self.reference_element]
        if not metals:
            return None

        depth_col = info.depth_col
        site_col = info.site_col

        # If we picked the deepest strategy but there is no depth column, fall back
        # to global_min_depth (only requires depth) or skip with a clear warning.
        effective_strategy = self.baseline_strategy
        if effective_strategy == "deepest" and depth_col is None:
            logger.warning(
                "baseline_strategy='deepest' requested but no depth column found; "
                "skipping interpretation."
            )
            return None

        return build_interpretation_report(
            raw_df,
            metals=metals,
            reference_element=self.reference_element,
            site_col=site_col,
            depth_col=depth_col,
            baseline_strategy=effective_strategy,
            custom_baseline=self.custom_baseline,
        )
```

### 2. Agregar tests de regresión en `tests/test_integration.py`

```python
def test_pipeline_runs_interpretation_on_isovida():
    """Regression: pipeline must produce an interpretation report when prerequisites are met."""
    from aeda.pipeline.runner import AEDAPipeline

    EXCLUDE = ["No", "Code", "Site_Name", "Pret_Code", "Código_muestra",
               "Sitio_muestreo", "Fecha_muestreo", "Core",
               "Latitud", "Longitud", "Profundidad"]

    p = AEDAPipeline(impute_strategy="median")
    r = p.run("data/BD_ISOVIDA_MANGLARES2023_rectificadaYBA_230326.xlsx",
              exclude_cols=EXCLUDE, sheet_name="DATA")

    assert r.interpretation is not None, "Interpretation must run on ISOVIDA"
    assert r.interpretation.ef_result is not None
    assert r.interpretation.ef_result.reference_element == "Al"
    assert len(r.interpretation.metals_analyzed) > 0
    # TEL/PEL should be computed for at least Pb
    assert "Pb" in r.interpretation.tel_pel_classifications.columns


def test_pipeline_skips_interpretation_without_reference_element():
    """If the reference element is not in the dataset, interpretation must be skipped, not crash."""
    from aeda.pipeline.runner import AEDAPipeline

    EXCLUDE = ["No", "Code", "Site_Name", "Pret_Code", "Código_muestra",
               "Sitio_muestreo", "Fecha_muestreo", "Core",
               "Latitud", "Longitud", "Profundidad"]

    # Use a non-existent reference element
    p = AEDAPipeline(impute_strategy="median", reference_element="Unobtanium")
    r = p.run("data/BD_ISOVIDA_MANGLARES2023_rectificadaYBA_230326.xlsx",
              exclude_cols=EXCLUDE, sheet_name="DATA")

    # Pipeline must still run successfully but with no interpretation
    assert r.dim_reduction is not None  # other steps still work
    assert r.interpretation is None


def test_pipeline_with_custom_reference_element():
    """The user can override the reference element via the constructor."""
    from aeda.pipeline.runner import AEDAPipeline

    EXCLUDE = ["No", "Code", "Site_Name", "Pret_Code", "Código_muestra",
               "Sitio_muestreo", "Fecha_muestreo", "Core",
               "Latitud", "Longitud", "Profundidad"]

    p = AEDAPipeline(impute_strategy="median", reference_element="Fe")
    r = p.run("data/BD_ISOVIDA_MANGLARES2023_rectificadaYBA_230326.xlsx",
              exclude_cols=EXCLUDE, sheet_name="DATA")

    assert r.interpretation is not None
    assert r.interpretation.ef_result.reference_element == "Fe"
    # Fe must not appear among the metals analyzed (it is the reference)
    assert "Fe" not in r.interpretation.metals_analyzed


def test_run_interpretation_flag_disables_step():
    """run_interpretation=False must skip the step entirely."""
    from aeda.pipeline.runner import AEDAPipeline

    EXCLUDE = ["No", "Code", "Site_Name", "Pret_Code", "Código_muestra",
               "Sitio_muestreo", "Fecha_muestreo", "Core",
               "Latitud", "Longitud", "Profundidad"]

    p = AEDAPipeline(impute_strategy="median", run_interpretation=False)
    r = p.run("data/BD_ISOVIDA_MANGLARES2023_rectificadaYBA_230326.xlsx",
              exclude_cols=EXCLUDE, sheet_name="DATA")

    assert r.interpretation is None
```

## Verificación

1. Ejecutar `pytest tests/ -v` y verificar que todos los tests pasan, incluidos los 4 nuevos.
2. Ejecutar el siguiente script y verificar que `r.interpretation` ya no es `None`:

```python
from aeda.pipeline.runner import AEDAPipeline

EXCLUDE = ["No", "Code", "Site_Name", "Pret_Code", "Código_muestra",
           "Sitio_muestreo", "Fecha_muestreo", "Core",
           "Latitud", "Longitud", "Profundidad"]

p = AEDAPipeline(impute_strategy="median")
r = p.run("data/BD_ISOVIDA_MANGLARES2023_rectificadaYBA_230326.xlsx",
          exclude_cols=EXCLUDE, sheet_name="DATA")
print(r.summary())
assert r.interpretation is not None
print(r.interpretation.summary())
```

Debe imprimir, entre otras cosas:
- `Environmental interpretation: Metals analyzed: ['As', 'Cr', 'Cu', 'Ni', 'Pb', 'Zn']`
- `Reference element: Al`
- `TEL/PEL classifications computed: <número grande>`

## Commit

Mensaje sugerido:

```
feat(pipeline): integrate environmental interpretation as automatic stage 10

The interpretation module (EF, TEL/PEL, Birch, LOD imputation) was implemented
but never invoked by AEDAPipeline.run(). The Streamlit UI therefore could not
display contamination classifications.

This change adds:
- AEDAResults.interpretation field (Optional[InterpretationReport])
- AEDAPipeline constructor params: run_interpretation, reference_element,
  baseline_strategy, custom_baseline
- New stage in run(): _run_interpretation, with pre-requisite checks
  (heavy metals present, reference element in dataset, depth column available)
- Failure-tolerant: if any pre-requisite is missing, interpretation is skipped
  with a logger.debug message; the pipeline continues and produces all other
  results normally.
- Updated AEDAResults.summary() to display interpretation status

Adds 4 regression tests covering the integration on the real ISOVIDA dataset
and edge cases (missing reference element, custom reference element, disabled flag).

This is the prerequisite for the upcoming Audit and Advanced Configuration UI pages.
```
