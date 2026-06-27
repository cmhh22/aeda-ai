# AEDA-AI

> **Automated Exploratory Data Analysis for Environmental Data** — a rule-based expert system + unsupervised machine learning + geochemical interpretation, with a bilingual web interface and full methodological traceability.

**Live app:** https://aeda-ai-lea-ceac.streamlit.app · **Repository:** https://github.com/cmhh22/aeda-ai

🇬🇧 [English](#english) · 🇪🇸 [Español](#español)

---

## English

### Overview
AEDA-AI automates the exploratory analysis of environmental datasets (e.g., X-ray fluorescence chemistry, grain size, heavy-metal sediment studies). It profiles a dataset, selects the appropriate methods through a **rule-based expert system**, runs the analysis (dimensionality reduction, clustering, anomaly detection, correlations), and produces a **geochemical interpretation** (Enrichment Factor, Birch severity bands, NOAA TEL/PEL thresholds, below-detection-limit imputation). Every automatic decision is recorded with its rule, reason and evidence, giving the user full methodological traceability.

It was developed for the Environmental Testing Laboratory of the Cienfuegos Environmental Studies Center (LEA-CEAC, Cuba) and validated with the **ISOVIDA** mangrove sediments dataset (Cienfuegos Bay).

### Key features
- **Rule-based expert system** that profiles the dataset and recommends methods automatically.
- **Unsupervised ML:** PCA / UMAP / t-SNE, K-Means / Ward / DBSCAN, Isolation Forest / LOF, Pearson / Spearman.
- **Environmental interpretation layer:** Enrichment Factor, Birch classification, TEL/PEL thresholds, LOD imputation (Succop et al., 2004).
- **Full traceability:** every decision is logged with its rule, reason and evidence (audit page).
- **Bilingual web interface** (Spanish / English), PDF report and Excel export.
- **Reproducible:** fixed random seeds + pinned environment.

### Architecture
| Layer | Subpackage |
| --- | --- |
| Ingestion & validation | `aeda.io` |
| Analysis engine (expert system + algorithms) | `aeda.engine` |
| Environmental interpretation | `aeda.interpretation` |
| Orchestration (`AEDAPipeline`) | `aeda.pipeline` |
| Visualization | `aeda.viz` |
| Web interface (Streamlit) | `app` |

The web interface is kept separate from the engine, so the analysis pipeline can be used programmatically.

### Quick start (programmatic)
```python
from aeda.pipeline.runner import AEDAPipeline

pipeline = AEDAPipeline()
results = pipeline.run(
    "data/BD_ISOVIDA_MANGLARES2023_rectificadaYBA_230326.xlsx",
    sheet_name="DATA",
    exclude_cols=[
        "No", "Code", "Site_Name", "Pret_Code", "Codigo_muestra",
        "Sitio_muestreo", "Fecha_muestreo", "Core",
        "Latitud", "Longitud", "Profundidad",
    ],
)

print(results.plan.summary())
print(results.summary())
```

### Installation
```bash
pip install -e .
```

### Reproducibility
Recreate the exact execution environment:
```bash
conda env create -f environment.yml      # conda
# or
pip install -r requirements-lock.txt     # pip
```
All stochastic components (K-Means, Isolation Forest, t-SNE, UMAP, Random Forest) use a fixed seed, so repeated runs on the same dataset with the same parameters produce identical results.

### Tests
```bash
pytest tests/
```

### Citation
If you use this software, please cite it using the metadata in [`CITATION.cff`](CITATION.cff).

### License
Released under the MIT License — see [`LICENSE`](LICENSE).

---

## Español

### Descripción
AEDA-AI automatiza el análisis exploratorio de datos ambientales (por ejemplo, química por fluorescencia de rayos X, granulometría y estudios de metales pesados en sedimentos). Perfila el conjunto de datos, selecciona los métodos apropiados mediante un **sistema experto basado en reglas**, ejecuta el análisis (reducción dimensional, agrupamiento, detección de anomalías, correlaciones) y produce una **interpretación geoquímica** (Factor de Enriquecimiento, bandas de severidad de Birch, umbrales TEL/PEL de la NOAA, imputación de valores por debajo del límite de detección). Cada decisión automática queda registrada con su regla, su razón y su evidencia, lo que da al usuario una trazabilidad metodológica completa.

Fue desarrollado para el Laboratorio de Ensayos Ambientales del Centro de Estudios Ambientales de Cienfuegos (LEA-CEAC, Cuba) y validado con el conjunto de datos **ISOVIDA** de sedimentos de manglar de la Bahía de Cienfuegos.

### Características principales
- **Sistema experto basado en reglas** que perfila el conjunto de datos y recomienda métodos automáticamente.
- **Aprendizaje automático no supervisado:** PCA / UMAP / t-SNE, K-Means / Ward / DBSCAN, Isolation Forest / LOF, Pearson / Spearman.
- **Capa de interpretación ambiental:** Factor de Enriquecimiento, clasificación de Birch, umbrales TEL/PEL, imputación bajo LDM (Succop et al., 2004).
- **Trazabilidad completa:** cada decisión se registra con su regla, razón y evidencia (página de auditoría).
- **Interfaz web bilingüe** (español / inglés), informe en PDF y exportación a Excel.
- **Reproducible:** semillas fijas + entorno bloqueado.

### Arquitectura
| Capa | Subpaquete |
| --- | --- |
| Ingesta y validación | `aeda.io` |
| Motor de análisis (sistema experto + algoritmos) | `aeda.engine` |
| Interpretación ambiental | `aeda.interpretation` |
| Orquestación (`AEDAPipeline`) | `aeda.pipeline` |
| Visualización | `aeda.viz` |
| Interfaz web (Streamlit) | `app` |

La interfaz web está separada del motor, de modo que el flujo de análisis puede usarse de forma programática.

### Uso rápido (programático)
```python
from aeda.pipeline.runner import AEDAPipeline

pipeline = AEDAPipeline()
results = pipeline.run(
    "data/BD_ISOVIDA_MANGLARES2023_rectificadaYBA_230326.xlsx",
    sheet_name="DATA",
    exclude_cols=[
        "No", "Code", "Site_Name", "Pret_Code", "Codigo_muestra",
        "Sitio_muestreo", "Fecha_muestreo", "Core",
        "Latitud", "Longitud", "Profundidad",
    ],
)

print(results.plan.summary())
print(results.summary())
```

### Instalación
```bash
pip install -e .
```

### Reproducibilidad
Recrear el entorno exacto de ejecución:
```bash
conda env create -f environment.yml      # conda
# o
pip install -r requirements-lock.txt     # pip
```
Todos los componentes estocásticos (K-Means, Isolation Forest, t-SNE, UMAP, Random Forest) usan semilla fija, por lo que ejecuciones repetidas sobre el mismo conjunto de datos con los mismos parámetros producen resultados idénticos.

### Pruebas
```bash
pytest tests/
```

### Cómo citar
Si usa este software, cítelo con los metadatos del archivo [`CITATION.cff`](CITATION.cff).

### Licencia
Distribuido bajo la Licencia MIT — ver [`LICENSE`](LICENSE).