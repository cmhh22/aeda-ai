# AEDA Framework

AI-assisted framework for **Advanced Environmental Data Analysis (AEDA)**.

## Current Status

- Module 1 (Universal Ingestion): **Implemented and validated**
- Module 2 (Exploration): In progress
- Module 3 (Explainable ML): Planned
- Module 4 (Reporting and Dashboard): Planned

## Module 1: Universal Ingestion

Module 1 ingests heterogeneous environmental datasets (sediment, water, air, soil, biota) and produces analysis-ready tables.

### Core Features

- Automatic matrix type detection
- Parsing of analytical notation (`< LOD`, `> AQL`, `± uncertainty`)
- Censored value handling with multiple strategies (`lod_half`, `ros`, `qmle`, `percentile`)
- Structured quality reporting and traceable metadata

### Core Components

- `src/ingestion/universal_data_ingestor.py`
- `src/ingestion/raw_data_ingestor.py`
- `src/ingestion/censored_value_handler.py`
- `src/ingestion/matrix_type_detector.py`
- `src/ingestion/data_quality_reporter.py`

## Quick Start

### 1) Create environment

```bash
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

### 2) Run ingestion demo

```bash
python main_ingestion.py
```

### 3) Run validation scripts

```bash
python tests/validation/test_ingestion_simple.py
python tests/validation/validate_ingestion.py
```

## Typical Programmatic Usage

```python
from src.ingestion.universal_data_ingestor import UniversalDataIngestor

schema = {
    "V_(ppm)": "ppm",
    "Cr_(ppm)": "ppm",
    "Mn_(ppm)": "ppm",
    "Fe_(%)": "%",
    "Pb_(ppm)": "ppm",
}

ingestor = UniversalDataIngestor(
    analyte_schema=schema,
    censored_value_strategy="lod_half",
    generate_quality_report=True,
)

result = ingestor.run("data/raw/your_dataset.xlsx")
clean_data = result["data"]
metadata = result["metadata"]
```

## Project Layout

```
AEDA - Framework/
├── config/
├── data/
│   ├── raw/
│   └── processed/
├── docs/
├── logs/
├── notebooks/
├── tests/
│   ├── unit/
│   └── validation/
├── src/
│   ├── ingestion/
│   ├── preprocessing/
│   ├── pipeline/
│   ├── exporting/
│   └── utils/
├── main.py
├── main_ingestion.py
├── ingestion_examples.py
└── setup_env.ps1
```

## Additional Documentation

- `docs/INGESTION_MODULE_GUIDE.md` for researcher-facing usage
- `docs/IMPLEMENTATION_SUMMARY.md` for technical implementation details

## Language Policy

All new development, comments, and documentation are maintained in **English**.
