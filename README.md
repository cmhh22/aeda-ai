# AEDA-AI

AEDA-AI (Automated Exploratory Data Analysis for Environmental Data) is a Python framework for environmental datasets such as FRX chemistry, granulometry, and heavy-metal sediment studies.

## Core architecture

- `aeda.io`: loading, validation, preprocessing
- `aeda.engine`: auto selector, dimensionality reduction, clustering, anomaly detection, correlations, feature analysis
- `aeda.pipeline`: end-to-end orchestration via `AEDAPipeline`

## Quick start

```python
from aeda.pipeline.runner import AEDAPipeline

pipeline = AEDAPipeline()
results = pipeline.run(
    "BD_ISOVIDA_MANGLARES2023.xlsx",
    exclude_cols=[
        "No",
        "Code",
        "Site_Name",
        "Pret_Code",
        "Codigo_muestra",
        "Sitio_muestreo",
        "Fecha_muestreo",
        "Core",
        "Latitud",
        "Longitud",
        "Profundidad",
    ],
    sheet_name="DATA",
)

print(results.plan.summary())
print(results.summary())
```

## Installation

```bash
pip install -e .
```

## Reproducibilidad

El entorno exacto de ejecución puede recrearse con:

```bash
# Opción conda
conda env create -f environment.yml

# Opción pip
pip install -r requirements-lock.txt
```

Todos los componentes estocásticos (K-Means, Isolation Forest, t-SNE, UMAP,
Random Forest) usan semilla fija, por lo que ejecuciones repetidas sobre el
mismo conjunto de datos producen resultados idénticos.
