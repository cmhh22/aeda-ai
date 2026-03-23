# Universal Ingestion Module Guide

This guide explains how to use the AEDA universal ingestion module for environmental datasets.

## Purpose

The universal ingestor transforms raw environmental laboratory data into analysis-ready tabular data with traceable metadata.

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

## Minimal Usage

```python
from src.ingestion.universal_data_ingestor import UniversalDataIngestor

ingestor = UniversalDataIngestor(
    analyte_schema={
        "V_(ppm)": "ppm",
        "Cr_(ppm)": "ppm",
        "Pb_(ppm)": "ppm",
    }
)

result = ingestor.run("data/raw/my_dataset.xlsx")
clean_data = result["data"]
```

## Advanced Usage

```python
from src.ingestion.universal_data_ingestor import UniversalDataIngestor

elements = {
    "Na_(%)": "%",
    "Mg_(%)": "%",
    "Al_(%)": "%",
    "V_(ppm)": "ppm",
    "Cr_(ppm)": "ppm",
    "Pb_(ppm)": "ppm",
}

metadata_columns = {
    "Code", "Site_Name", "Sampling_Date", "Site_Code", "Core",
    "Coord_Latitud", "Coord.Longitud", "pH", "PPI_550ºC (%)"
}

ingestor = UniversalDataIngestor(
    analyte_schema=elements,
    metadata_columns=metadata_columns,
    target_unit="ppm",
    strict_schema=False,
    censored_value_strategy="lod_half",
    generate_quality_report=True,
)

result = ingestor.run(
    "data/raw/BD_ISOVIDA_MANGLARES2023_version250226. Entregarxlsx.xlsx",
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
2. Run ingestion on raw input.
3. Review `quality_report` and `censored_summary`.
4. Save `result["data"]` to `data/processed/`.
5. Continue with Module 2 exploration.

## Validation Scripts

- `python tests/validation/test_ingestion_simple.py`
- `python tests/validation/validate_ingestion.py`
- `python main_ingestion.py`

## Troubleshooting

- If columns are missing, verify exact names in `analyte_schema`.
- If confidence is low, set `matrix_type_hint` explicitly.
- If censoring is high, compare `lod_half` vs `ros` for sensitivity analysis.

## Policy

Active development and documentation are maintained in English.
