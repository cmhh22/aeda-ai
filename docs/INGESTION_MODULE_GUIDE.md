# Universal Ingestion Module Guide

This guide explains how to use the AEDA universal ingestion module for environmental datasets.

## Purpose

The universal ingestor transforms environmental laboratory data into analysis-ready tabular data with traceable metadata.

It is designed for heterogeneous matrix types:

- Sediment
- Soil
- Water
- Air
- Biota

## Main Capabilities

- Matrix type auto-detection
- Parsing of analytical notation (`< LOD`, `> AQL`, `± uncertainty`)
- Censored value handling with configurable strategies
- Automatic quality report generation

## Minimal Usage (Rectified ISOVIDA)

```python
from src.ingestion.universal_data_ingestor import UniversalDataIngestor

ingestor = UniversalDataIngestor(
    analyte_schema={
        "V": "ppm",
        "Cr": "ppm",
        "Pb": "ppm",
    },
    apply_censored_handling=False,
)

result = ingestor.run("data/raw/BD_ISOVIDA_MANGLARES2023_rectificadaYBA_230326.xlsx")
clean_data = result["data"]
```

## Advanced Usage (Rectified ISOVIDA)

```python
from src.ingestion.universal_data_ingestor import UniversalDataIngestor

elements = {
    "Na": "%",
    "Mg": "%",
    "Al": "%",
    "V": "ppm",
    "Cr": "ppm",
    "Pb": "ppm",
}

metadata_columns = {
    "Code", "Código_muestra", "Sitio_muestreo", "Fecha_muestreo",
    "Core", "Latitud", "Longitud", "Profundidad", "PPI550", "U_PPI550"
}

ingestor = UniversalDataIngestor(
    analyte_schema=elements,
    metadata_columns=metadata_columns,
    target_unit="ppm",
    strict_schema=False,
    censored_value_strategy="lod_half",
    apply_censored_handling=False,
    generate_quality_report=True,
)

result = ingestor.run(
    "data/raw/BD_ISOVIDA_MANGLARES2023_rectificadaYBA_230326.xlsx",
    matrix_type_hint="sediment",  # optional
)
```

## Output Structure

The ingestor returns a standardized dictionary-like output with:

- `data`: processed DataFrame
- `metadata`: ingestion lineage and detection metadata
- optional keys (when enabled):
  - `quality_report`
  - `censored_summary`
  - `raw_data`
  - `parsed_data`

## Censored Value Strategies

For the rectified ISOVIDA dataset, tutor guidance sets LOD/LOQ handling out of Module 1 scope.
Use `apply_censored_handling=False` for operational runs.
The strategies below remain available for other datasets or investigator-led sensitivity analyses.

- `lod_half` (default): robust and transparent baseline
- `ros`: useful when censoring is moderate and sample size is sufficient
- `qmle`: for larger datasets and model-based imputation workflows
- `percentile`: configurable placement between conservative and observed bounds

## Matrix Detection

Detection combines:

- Metadata keyword signals
- Typical elemental signatures
- Numeric concentration range heuristics
- Physical property indicators

You can bypass uncertainty by passing `matrix_type_hint`.

## Recommended Workflow

1. Define your analyte schema.
2. Run ingestion on the rectified `DATA` input.
3. Review `quality_report` and `censored_summary`.
4. Save `result["data"]` to `data/processed/`.
5. Validate methodological decisions with the tutor (censoring, units, QA/QC, objective).
6. Continue with Module 2 exploration.

## Validation Scripts

- `python tests/validation/test_ingestion_simple.py`
- `python tests/validation/validate_ingestion.py`
- `python main_ingestion.py`

## Troubleshooting

- If columns are missing, verify exact names in `analyte_schema`.
- If confidence is low, set `matrix_type_hint` explicitly.
- If censoring is high, compare `lod_half` vs `ros` for sensitivity analysis.

## Tutor-Aligned Checklist (ISOVIDA)

- Confirm LOD/LOQ remains out-of-scope for Module 1 in the rectified dataset.
- Canonical units per analyte and allowed conversions.
- QA/QC acceptance thresholds (missingness, outliers, duplicates).
- QA/QC rule for low-variability/repeated-value variables.
- Official data dictionary lock (`Diccionario_DATA`).
- Priority scientific objective for this phase.

## Policy

Active development and documentation are maintained in English.
