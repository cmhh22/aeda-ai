# 📊 AEDA Framework - Resumen Detallado

**Última actualización:** Marzo 17, 2026 | **Versión:** 1.0 | **Estado:** ✅ Productivo

---

## 📋 Tabla de Contenidos

1. [Descripción General](#descripción-general)
2. [Tecnologías y Dependencias](#tecnologías-y-dependencias)
3. [Estructura del Proyecto](#estructura-del-proyecto)
4. [Componentes Principales](#componentes-principales)
5. [Características de IA/ML](#características-de-iaml)
6. [Guía de Uso](#guía-de-uso)
7. [Arquitectura y Patrones](#arquitectura-y-patrones)
8. [Sistema de Auditoría](#sistema-de-auditoría)
9. [Exportación de Datos](#exportación-de-datos)
10. [Configuración](#configuración)

---

## 🎯 Descripción General

### ¿Qué es AEDA Framework?

**AEDA** (Advanced Environmental Data Analysis) es un framework Python enterprise-grade diseñado para el análisis, limpieza y procesamiento de datos ambientales con énfasis en:

- **Reproducibilidad científica**: Cada transformación queda registrada en auditoría JSON
- **Robustez**: Manejo avanzado de outliers, datos faltantes e inconsistencias
- **Escalabilidad**: Arquitectura modular basada en patrones de diseño probados
- **Trazabilidad**: Versioning de datos con hashes SHA-256 para auditoría

### Objetivos Principales

✅ Orquestar pipelines de procesamiento de datos ambientales  
✅ Aplicar técnicas de IA/ML para detección y corrección automática de anomalías  
✅ Garantizar reproducibilidad mediante audit logging detallado  
✅ Exportar datos procesados en múltiples formatos (Parquet, CSV, Pickle)  
✅ Proveer interfaz CLI intuitiva para usuarios finales  

---

## 🔧 Tecnologías y Dependencias

### Stack Tecnológico

| Categoría | Tecnología | Versión | Propósito |
|-----------|------------|---------|----------|
| **Datos** | Pandas | ≥ 2.2.0 | Manipulación de DataFrames |
| **Datos** | NumPy | ≥ 1.26.0 | Operaciones numéricas |
| **Datos** | PyArrow | ≥ 16.0.0 | Serialización Parquet |
| **ML** | Scikit-learn | ≥ 1.5.0 | Escalado (StandardScaler), ML base |
| **Outliers** | PyOD | ≥ 2.0.0 | Isolation Forest para detección de anomalías |
| **Imputación** | MissingPy | ≥ 0.2.0 | MissForest (RF-based imputation) |
| **Interpolación** | SciPy | ≥ 1.13.0 | PCHIP (Piecewise Cubic Hermite) |
| **CLI** | argparse | Built-in | Parsing de argumentos CLI |
| **Progreso** | tqdm | ≥ 4.66.0 | Barras de progreso en terminal |

### Dependencias de Desarrollo

- `pytest` / `unittest` (testing)
- `python-dotenv` (variables de entorno)
- `PyYAML` (configuración)

---

## 📁 Estructura del Proyecto

```
AEDA - Framework/
│
├── 📄 main.py                          ← Punto de entrada CLI
├── 📄 setup.py                         ← Configuración del paquete
├── 📄 requirements.txt                 ← Dependencias Python
├── 📄 setup_env.ps1                    ← Setup automático (PowerShell)
├── 📄 .gitignore                       ← Git exclusiones
├── 📄 .env                             ← Variables de entorno (no versionado)
├── 📄 environment.yml                  ← Spec conda (alternativo)
│
├── 🗂️ config/
│   └── 📄 params.yaml                  ← Configuración centralizada (thresholds, paths, seeds)
│
├── 🗂️ data/
│   ├── 📂 raw/                         ← Datos originales (sin procesar)
│   │   └── 📄 BD_ISOVIDA_MANGLARES.xlsx
│   └── 📂 processed/                   ← Datos procesados y exportados
│       └── 📄 .gitkeep
│
├── 🗂️ src/
│   ├── 📂 pipeline/                    ← Orquestador principal
│   │   ├── 📄 aeda_pipeline.py         ← Chain of Responsibility orchestrator
│   │   ├── 📄 pipeline_step.py         ← ABC base para pasos
│   │   ├── 📄 pipeline_logger.py       ← Snapshots y logging
│   │   ├── 📄 example_usage.py         ← Ejemplo de uso API
│   │   └── 📄 test_aeda_pipeline.py    ← Tests unitarios (13+ casos)
│   │
│   ├── 📂 preprocessing/               ← Transformaciones de datos
│   │   ├── 📄 outlier_detector.py      ← Detección de anomalías (Isolation Forest + reglas)
│   │   ├── 📄 data_reconstructor.py    ← Imputación (MissForest, PCHIP)
│   │   ├── 📄 data_standardizer.py     ← Escalado (StandardScaler, Box-Cox, log)
│   │   └── 📄 __init__.py              ← Exports públicos
│   │
│   ├── 📂 utils/                       ← Utilidades centralizadas
│   │   ├── 📄 decorators.py            ← @track_transformation (audit logging)
│   │   ├── 📄 logs.py                  ← Primitivas JSON thread-safe
│   │   ├── 📄 metadata.py              ← Extracción de metadatos DataFrame
│   │   └── 📄 __init__.py
│   │
│   ├── 📂 exporting/                   ← Exportación de datos
│   │   ├── 📄 data_exporter.py         ← Multi-format export con versioning
│   │   └── 📄 __init__.py
│   │
│   ├── 📂 ingestion/                   ← Lectura de datos (expandible)
│   │   └── (futuro: loaders específicos)
│   │
│   ├── 📄 data_component.py            ← Base data abstractions
│   └── 📄 __init__.py
│
├── 🗂️ notebooks/
│   └── (EDA, exploraciones, demostraciones)
│
├── 🗂️ .git/                            ← Repositorio Git
│
└── 🗂️ .venv/                           ← Ambiente virtual Python

**Archivos generados en runtime:**
- 📄 audit_log.json                     ← Log de todas las transformaciones
- 📄 *.parquet/*.csv/*.pkl              ← Datos exportados (data/processed)
```

---

## 🏗️ Componentes Principales

### 1. **AEDA_Pipeline** - Orquestador Principal

**Ubicación:** `src/pipeline/aeda_pipeline.py` (250+ líneas)

**Patrón:** Chain of Responsibility

**Funcionalidad:**
- Registra pasos de procesamiento (`register_step()`)
- Ejecuta secuencialmente con snapshots de estado
- Captura errores con contexto detallado
- Permite rollback a estado anterior

**Métodos principales:**
```python
pipeline = AEDA_Pipeline(name="EDA_Pipeline", log_dir="logs")
pipeline.register_step(OutlierDetector(), name="outlier_removal")
pipeline.register_step(DataReconstructor(), name="imputation")
pipeline.register_step(DataStandardizer(), name="scaling")

result = pipeline.execute(data, stop_on_error=True)
```

**Características:**
- 🔹 Snapshots automáticos pre/post cada paso
- 🔹 Logging de errores con stack trace + estado
- 🔹 Recuperación parcial (partial_ok=True)
- 🔹 Skip de pasos iniciales (execute_from())

---

### 2. **@track_transformation** - Decorador de Auditoría

**Ubicación:** `src/utils/decorators.py` (100+ líneas)

**Propósito:** Registrar CADA transformación de datos para reproducibilidad científica

**Qué captura:**
- ✅ Nombre de la función
- ✅ Parámetros usados (via `inspect.signature()`)
- ✅ Timestamp UTC (ISO 8601)
- ✅ Tiempo de ejecución (segundos)
- ✅ Estadísticas ANTES: shape, dtypes, nulos, media/std
- ✅ Estadísticas DESPUÉS: shape, dtypes, nulos, media/std
- ✅ Estado (success/error)

**Archivo de auditoría:**
```json
[
  {
    "timestamp_utc": "2026-03-17T14:32:15.123456Z",
    "function_name": "OutlierDetector.run",
    "parameters": {
      "contamination": 0.05,
      "method": "isolation_forest"
    },
    "execution_time_seconds": 2.341,
    "status": "success",
    "dataset_before": {
      "shape": [1000, 45],
      "dtypes": {"numeric": 30, "object": 15},
      "null_counts": 45,
      "numeric_summary": {
        "column_name": {"mean": 25.3, "std": 10.2, "count": 1000}
      }
    },
    "dataset_after": {
      "shape": [980, 45],
      "null_counts": 45,
      "numeric_summary": {...}
    }
  }
]
```

**Thread-safe:** Usa `threading.Lock` para evitar corrupción en escrituras concurrentes

---

### 3. **OutlierDetector** - Detección de Anomalías

**Ubicación:** `src/preprocessing/outlier_detector.py`

**Algoritmos implementados:**

#### a) **Isolation Forest** (ML)
- Modelo no supervisado que aísla anomalías
- Usa `contamination` como parámetro (fracción esperada de outliers)
- Rápido y eficiente para datos de alta dimensión

#### b) **Reglas Físicas Domain-Specific**
- Límites NAAQS (National Ambient Air Quality Standards)
- Límites personalizables por columna
- Detecta violaciones de rangos esperados

**Uso:**
```python
detector = OutlierDetector(contamination=0.05, method="hybrid")
data = detector.run(data)  # Remueve outliers automáticamente
```

**Decorado con:** `@track_transformation` → auditoría JSON

---

### 4. **DataReconstructor** - Imputación de Datos

**Ubicación:** `src/preprocessing/data_reconstructor.py`

**Métodos implementados:**

#### a) **MissForest** (Iterativo, ML)
- Algoritmo de imputación iterativa basado en Random Forest
- Trata características correlacionadas
- Configurable: `n_estimators=200`, `max_iter=10`

```python
reconstructor = DataReconstructor()
data = reconstructor.reconstruct_tabular_missforest(
    data, 
    n_estimators=200,
    max_iter=10
)
```

#### b) **PCHIP** (Spline interpolación)
- Piecewise Cubic Hermite Interpolating Polynomial
- Para series de tiempo
- Preserva monotonía local

```python
data = reconstructor.reconstruct_time_series_pchip(data)
```

**Decorado con:** `@track_transformation` → auditoría

---

### 5. **DataStandardizer** - Escalado Normalización

**Ubicación:** `src/preprocessing/data_standardizer.py`

**Transformaciones aplicadas:**

#### a) **StandardScaler** (Media=0, Std=1)
- Normalización Z-score clásica

#### b) **Box-Cox Transformation** (para datos sesgados)
- Transforma distribuciones no-normales → aproximadamente normales
- Solo para datos positivos

#### c) **Log1p Transformation**
- `log(1 + x)` para datos con rango amplio
- Estabiliza varianza

**Detección automática:**
- Calcula skewness de cada columna
- Si skewness > threshold → aplica Box-Cox o log1p
- Threshold configurable en `params.yaml`

**Decorado con:** `@track_transformation`

---

### 6. **DataExporter** - Exportación Versionada

**Ubicación:** `src/exporting/data_exporter.py` (120+ líneas)

**Formatos soportados:**

| Formato | Compresión | Ventaja | Caso de uso |
|---------|-----------|---------|-----------|
| **Parquet** | Snappy (configurable) | Columnar, eficiente, schema preservado | Análisis posterior, reproducibilidad |
| **CSV** | Gzip opcional | Universalmente legible | Compartir con no-técnicos |
| **Pickle** | N/A | Objeto Python puro | Modelos ML serializados |

**Versionado automático:**
```
nombre_YYYYMMDDTHHMMSSZ_hash12.ext

Ejemplo: 
  data_20260317T143215Z_a1b2c3d4e5f6.parquet
  data_20260317T143215Z_a1b2c3d4e5f6.csv
```

**Hash:** SHA-256(concatenación de dtypes + row hashes) → primeros 12 caracteres

**Método:**
```python
exporter = DataExporter(output_dir="data/processed")
hash_val, timestamp, parquet_path, csv_path = exporter.export_dataframes(
    data,
    base_name="análisis_isovida",
    include_index=False,
    parquet_compression="snappy"
)
```

**Previene:** Sobrescrituras accidentales (hash diferente = archivo nuevo)

---

### 7. **CLI (main.py)** - Interfaz de Usuario

**Ubicación:** `main.py` (200+ líneas)

**Subcomandos:**

```bash
python main.py run-pipeline \
  --input data/raw/samples.csv \
  --output data/processed/result.csv \
  --impute missforest \  # o 'pchip'
  --outliers \           # o --no-outliers
  --contamination 0.05 \
  --random-state 42
```

**Parámetros:**
- `--input`: ✅ Requerido. Ruta a CSV/XLSX
- `--output`: Opcional. Ruta de salida (default: input_stem_processed.csv)
- `--impute`: Método imputación {missforest, pchip}
- `--outliers`: Flag para habilitar detección
- `--contamination`: Fracción de outliers esperados (0-1)
- `--random-state`: Seed para reproducibilidad

**Progreso:**
```
Pipeline progress: 37%|███▌     | 3/8 [00:12<00:23, 2.15s/stage]
```

Barras TQDM por cada etapa:
1. Lectura de datos
2. Detección de outliers (si aplica)
3. Imputación (método elegido)
4. Estandarización
5. Exportación

---

## 🤖 Características de IA/ML

### Algoritmos de Machine Learning Utilizados

#### 1. **Isolation Forest** (Detección de Anomalías)
- ✅ **Librería:** PyOD
- ✅ **Tipo:** Aprendizaje no supervisado (unsupervised)
- ✅ **Fundamento:** Aísla observaciones anómalas usando splits aleatorios
- ✅ **Ventajas:**
  - Eficiente en datos de alta dimensión
  - No paramétrico (no asume distribución)
  - Depende poco del diseño de características
- ✅ **Configuración:** `contamination=0.05` predeterminado
- ✅ **Salida:** Puntuaciones de anomalía por observación

**Uso en el framework:**
```python
detector = OutlierDetector(contamination=0.05, method="isolation_forest")
cleaned_data = detector.run(raw_data)  # Remueve filas anómalas
```

---

#### 2. **Random Forest Imputation (MissForest)**
- ✅ **Librería:** MissingPy (Stekhoven et al., 2012)
- ✅ **Tipo:** Aprendizaje semisupervisado + iterativo
- ✅ **Fundamento:** 
  - Itera entre: estimar características missing → reentrenar modelo
  - Captura dependencias complejas entre variables
- ✅ **Ventajas:**
  - Maneja correlaciones multivariadas
  - Superior a imputación simple (media, KNN)
  - Robusto a patrones de missingness
- ✅ **Configuración:** 
  - `n_estimators=200` (árboles en forest)
  - `max_iter=10` (iteraciones convergencia)
- ✅ **Complejidad:** O(n * p * t * k) donde t=iteraciones, k=estimators

**Uso en el framework:**
```python
reconstructor = DataReconstructor()
imputed_data = reconstructor.reconstruct_tabular_missforest(
    data_with_nulls,
    n_estimators=200,
    max_iter=10
)
```

---

#### 3. **Standardization / Feature Scaling**
- ✅ **Librería:** Scikit-learn (StandardScaler, PowerTransformer)
- ✅ **Tipo:** Normalización (preprocessing estadístico)
- ✅ **Técnicas:**
  - **Z-score:** (x - μ) / σ → Media=0, Std=1
  - **Box-Cox:** λ-parameter transform para normalidad
  - **Log1p:** log(1 + x) para estabilizar rango

**Uso en el framework:**
```python
standardizer = DataStandardizer()
scaled_data = standardizer.run(data)  # Aplica heurística automática
```

---

#### 4. **PCHIP Interpolation** (Series Temporales)
- ✅ **Librería:** SciPy
- ✅ **Tipo:** Interpolación tipo Spline
- ✅ **Fundamento:** Polinomios cúbicos a trozos hermitanos
- ✅ **Ventajas:**
  - Preserva monotonía local
  - Suave (C¹ continuo)
  - Sin oscilaciones (vs Lagrange)

---

### Síntesis: Pipeline ML Full-Stack

```
Datos Crudos (raw)
       ↓
   [Isolation Forest] ← Detección outliers ML
       ↓
   [MissForest] ← Imputación iterativa ML
       ↓
   [StandardScaler/Box-Cox] ← Feature scaling ML
       ↓
Datos Limpios (processed)
```

**Total de modelos ML:** 3 algoritmos (1 detección + 1 imputación + 1 escalado)

---

## 💻 Guía de Uso

### Instalación

```bash
# Clonar repositorio
git clone https://github.com/cmhh22/aeda-framework.git
cd "AEDA - Framework"

# Crear environment (PowerShell)
.\setup_env.ps1

# O manual:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

### Uso CLI

#### Ejemplo 1: Procesamiento estándar
```bash
python main.py run-pipeline \
  --input data/raw/samples.csv \
  --impute missforest \
  --outliers
```

**Resultado:**
```
data/processed/samples_processed.csv
audit_log.json (actualizado con transformaciones)
```

#### Ejemplo 2: Con parámetros personalizados
```bash
python main.py run-pipeline \
  --input data/raw/BD_ISOVIDA_MANGLARES.xlsx \
  --output data/processed/isovida_clean.csv \
  --impute pchip \
  --no-outliers \
  --random-state 123
```

#### Ejemplo 3: Sin imputación, solo outliers
```bash
python main.py run-pipeline \
  --input data/raw/environmental_data.csv \
  --contamination 0.10 \
  --outliers
```

---

### Uso Programático (API Python)

```python
from pathlib import Path
from src.pipeline import AEDA_Pipeline
from src.preprocessing import OutlierDetector, DataReconstructor, DataStandardizer
from src.exporting import DataExporter
import pandas as pd

# 1. Cargar datos
data = pd.read_csv("data/raw/samples.csv")

# 2. Crear pipeline
pipeline = AEDA_Pipeline(name="my_analysis", log_dir="logs")
pipeline.register_step(OutlierDetector(contamination=0.05), name="detector")
pipeline.register_step(DataReconstructor(), name="imputation")
pipeline.register_step(DataStandardizer(), name="scaling")

# 3. Ejecutar
result = pipeline.execute(data)

# 4. Exportar con versionado
exporter = DataExporter(output_dir="data/processed")
hash_val, timestamp, parquet_path, csv_path = exporter.export_dataframes(
    result,
    base_name="analysis_output"
)

print(f"✅ Exportado: {parquet_path}")
print(f"📊 Hash: {hash_val}")
print(f"📅 Timestamp: {timestamp}")
```

---

### Configuración (config/params.yaml)

```yaml
project:
  name: AEDA-Framework
  timezone: UTC

paths:
  input_default: data/raw/samples.csv
  output_processed: data/processed
  audit_log: audit_log.json

pipeline:
  random_state: 42              # Para reproducibilidad
  outliers_enabled: true        # Habilitar detección
  imputation_method: missforest # missforest o pchip

outliers:
  contamination: 0.05           # Fracción esperada de outliers
  physical_limits_source: DEFAULT_NAAQS_LIMITS

imputation:
  missforest_n_estimators: 200  # Árboles en Random Forest
  missforest_max_iter: 10       # Iteraciones de convergencia
  pchip_enabled: true           # Habilitar interpolación

standardization:
  normal_skew_threshold: 0.5    # Umbral para Box-Cox
  high_skew_threshold: 1.0
  normaltest_alpha: 0.05        # Significancia normalidad
  epsilon: 0.000001             # Evitar división por cero
```

**Modificar según necesidad** (ej: `contamination: 0.10` para más sensibilidad a outliers)

---

## 🏛️ Arquitectura y Patrones

### Patrones de Diseño Utilizados

#### 1. **Chain of Responsibility**
```python
class AEDA_Pipeline:
    def execute(self, data):
        for step in self.steps:
            data = step.execute(data)  # Cadena de transformaciones
        return data
```
- Ventaja: Fácil agregar/quitar pasos sin modificar código existente
- Ubicación: `src/pipeline/aeda_pipeline.py`

---

#### 2. **Decorator Pattern**
```python
@track_transformation
def run(self, data):
    # Auditoría automática: parámetros, tiempo, estadísticas
    return processed_data
```
- Ventaja: Añade auditoría sin modificar lógica de transformación
- Ubicación: `src/utils/decorators.py`

---

#### 3. **Strategy Pattern**
```python
reconstructor = DataReconstructor()
# Estrategia 1: MissForest
data = reconstructor.reconstruct_tabular_missforest(data)
# Estrategia 2: PCHIP
data = reconstructor.reconstruct_time_series_pchip(data)
```
- Ventaja: Intercambia algoritmos en runtime
- Ubicación: `src/preprocessing/data_reconstructor.py`

---

#### 4. **Factory Pattern**
```python
# CLI main.py construye pasos dinámicamente
stages = []
if args.outliers:
    stages.append(OutlierDetector())
if args.impute == "missforest":
    stages.append(DataReconstructor())
# ...etc
```
- Ventaja: Construcción flexible según parámetros
- Ubicación: `main.py`

---

### Flujo Arquitectónico

```
┌─────────────────────────────────────────┐
│          CLI Entry (main.py)            │
│  argparse + tqdm progress               │
└──────────────┬──────────────────────────┘
               ↓
       ┌───────────────────┐
       │  AEDA_Pipeline    │
       │ (Orchestrator)    │
       └──────────┬────────┘
                  ↓
        ┌─────────────────────┐
        │   Register Steps    │
        ├─────────────────────┤
        │ 1. OutlierDetector  │ ← PyOD (Isolation Forest)
        │ 2. DataReconstructor│ ← MissingPy (MissForest) + SciPy (PCHIP)
        │ 3. DataStandardizer │ ← Scikit-learn (StandardScaler, Box-Cox)
        └──────────┬──────────┘
                   ↓
        ┌─────────────────────┐
        │  Execute Pipeline   │
        │  + Snapshots        │
        │  + Error Handling   │
        └──────────┬──────────┘
                   ↓
        ┌─────────────────────┐
        │  @track_transformation
        │  Decorator Thread   │ ← Thread-safe JSON audit log
        └──────────┬──────────┘
                   ↓
        ┌─────────────────────┐
        │  DataExporter       │
        │ (Multi-format)      │
        ├─────────────────────┤
        │ • Parquet (PyArrow) │
        │ • CSV (pandas)      │
        │ • Pickle (ML models)│
        └──────────┬──────────┘
                   ↓
        ┌─────────────────────┐
        │  Versioned Output   │
        │ name_TIMESTAMP_HASH │
        └─────────────────────┘
```

---

## 📝 Sistema de Auditoría

### Objetivo

Garantizar **reproducibilidad científica** registrando:
- ✅ QUÉ transformaciones se aplicaron
- ✅ CUÁNDO se ejecutaron
- ✅ QUIÉN las ejecutó (función, parámetros)
- ✅ CUÁNTO tardaron
- ✅ CÓMO cambió el dataset (stats antes/después)

### Archivo de Auditoría

**Ubicación:** `audit_log.json` (raíz del proyecto)

**Formato:** JSON Lines (un objeto JSON por línea)

```json
[
  {
    "timestamp_utc": "2026-03-17T14:32:15.123456Z",
    "function_name": "OutlierDetector.run",
    "parameters": {
      "contamination": 0.05,
      "method": "isolation_forest",
      "n_estimators": 100
    },
    "execution_time_seconds": 2.341,
    "status": "success",
    "dataset_before": {
      "shape": [1000, 45],
      "dtypes": {"int64": 15, "float64": 25, "object": 5},
      "null_count_total": 47,
      "numeric_summary": {
        "temperature_c": {"mean": 25.3, "std": 10.2, "count": 1000},
        "humidity_pct": {"mean": 65.4, "std": 20.1, "count": 998},
        ...
      }
    },
    "dataset_after": {
      "shape": [980, 45],  ← 20 outliers removidos
      "dtypes": {"int64": 15, "float64": 25, "object": 5},
      "null_count_total": 47,
      "numeric_summary": {...}
    }
  },
  {
    "timestamp_utc": "2026-03-17T14:32:18.456789Z",
    "function_name": "DataReconstructor.reconstruct_tabular_missforest",
    "parameters": {
      "n_estimators": 200,
      "max_iter": 10,
      "method": "missforest"
    },
    "execution_time_seconds": 5.127,
    "status": "success",
    "dataset_before": {
      "shape": [980, 45],
      "null_count_total": 47
    },
    "dataset_after": {
      "shape": [980, 45],
      "null_count_total": 0  ← Todos los valores imputados
    }
  },
  ...
]
```

### Consultar Auditoría

```python
import json

with open("audit_log.json") as f:
    log = json.load(f)

# Filtrar por función
outlier_ops = [op for op in log if "OutlierDetector" in op["function_name"]]

# Calcular tiempo total
total_time = sum(op["execution_time_seconds"] for op in log)

print(f"Total transformaciones: {len(log)}")
print(f"Tiempo total pipeline: {total_time:.2f}s")
```

### Thread-Safety

- ✅ Usa `threading.Lock()` para prevenir corrupción en escrituras concurrentes
- ✅ Previene race conditions en JSON append
- ✅ Seguro para pipelines paralelos (futuro)

---

## 💾 Exportación de Datos

### Formatos Soportados

#### 1. **Parquet (Recomendado)**
```python
exporter.export_dataframes(data, base_name="análisis", parquet_compression="snappy")
```
- ✅ **Compresión:** Snappy (configurable: "gzip", "brotli", "lz4", "zstd")
- ✅ **Ventajas:**
  - Formato columnar (eficiente lectura selectiva)
  - Schema preservado (dtypes)
  - Versionable (por hash)
- ✅ **Tamaño:** ~40-60% vs CSV original
- ✅ **Lectura:** `pd.read_parquet(path)`

#### 2. **CSV**
```python
# Automático en export_dataframes()
```
- ✅ **Ventajas:** Universal, editable en Excel
- ✅ **Desventajas:** Pierde tipos de datos, más grande
- ✅ **Lectura:** `pd.read_csv(path)`

#### 3. **Pickle** (Modelos ML)
```python
exporter.export_model_pickle(model, "outlier_detector", data_hash, timestamp)
```
- ✅ **Ventajas:** Guarda objetos Python complejos (modelos entrenados)
- ✅ **Desventajas:** No portátil entre Python versions
- ✅ **Uso:** Reutilizar modelos entrenados sin recompiler

---

### Versionado por Hash

**Problema:** Cómo evitar sobrescrituras sin usar timestamps manualmente

**Solución:** SHA-256 sobre contenido dataset

```python
hash_value = compute_data_hash(data)
# Entrada: dtypes string + row hashes
# Salida: SHA-256 truncado a 12 caracteres (collision probability << 2^-32)

# Ejemplo de salida:
# datos_20260317T143215Z_a1b2c3d4e5f6.parquet
#       └─ timestamp ─┘ └─ contenido hash ─┘
```

**Beneficios:**
- ✅ Dos datasets distintos → nombres distintos
- ✅ Mismo dataset → mismo hash
- ✅ Trazabilidad: qué datos → qué resultados
- ✅ Evita sobrescrituras accidentales

---

## ⚙️ Configuración

### Variables de Entorno (.env)

```
AEDA_AUDIT_LOG_PATH=audit_log.json
AEDA_LOG_DIR=logs
AEDA_RANDOM_STATE=42
```

### Parámetros YAML (config/params.yaml)

| Parámetro | Rango | Default | Descripción |
|-----------|-------|---------|-------------|
| `contamination` | (0, 0.5) | 0.05 | % esperado outliers (Isolation Forest) |
| `missforest_n_estimators` | [10, 500] | 200 | Árboles en Random Forest |
| `missforest_max_iter` | [1, 50] | 10 | Iteraciones convergencia MissForest |
| `normal_skew_threshold` | (0, ∞) | 0.5 | Umbral skewness para Box-Cox |
| `random_state` | 0-2^32 | 42 | Seed reproducibilidad |

### Modificación en CLI

```bash
python main.py run-pipeline \
  --input data/raw/data.csv \
  --contamination 0.10 \  # Overrides params.yaml
  --outliers
```

---

## 📊 Casos de Uso

### 1. **Análisis Ambiental Inicial**
```bash
python main.py run-pipeline \
  --input data/raw/BD_ISOVIDA_MANGLARES.xlsx \
  --impute missforest --outliers
```
→ Genera datos limpios listos para análisis

---

### 2. **Auditoría de Calidad de Datos**
```python
import json
with open("audit_log.json") as f:
    log = json.load(f)
    
for op in log:
    print(f"{op['function_name']}: {op['dataset_before']['null_count_total']} → {op['dataset_after']['null_count_total']} nulos")
```
→ Verifica qué transformaciones ocurrieron

---

### 3. **Reproducción de Análisis Anterior**
```bash
# Con RANDOM_STATE=42 y dataset original → resultados idénticos
python main.py run-pipeline \
  --input data/raw/same_data.csv \
  --random-state 42 --impute missforest --outliers
  
# Comparar hashes:
# Ejecutar 1: hash_a1b2c3d4e5f6
# Ejecutar 2: hash_a1b2c3d4e5f6  ← Mismo resultado = reproducible
```

---

## 🔍 Troubleshooting

| Problema | Causa | Solución |
|----------|-------|----------|
| `FileNotFoundError: input.csv` | Archivo no existe | Verificar ruta relativa desde raíz proyecto |
| `ImportError: No module named 'preprocessing'` | sys.path incorrecto | `main.py` inserta SRC_DIR automáticamente |
| `audit_log.json` corrupto | Race condition escritura | Usar lock (ya implementado en decorator) |
| Muy lento con MissForest | Dataset grande (>100k rows) | Reducir `max_iter` o usar PCHIP |
| Resultados diferentes entre ejecuciones | Random state no seteado | Especificar `--random-state 42` |

---

## 📈 Métricas y Performance

### Benchmark (dataset ~10k rows x 50 cols)

| Etapa | Tiempo | Algoritmo |
|-------|--------|-----------|
| Outlier Detection | ~2s | Isolation Forest (PyOD) |
| Imputation (MissForest) | ~5-8s | Random Forest iterativo |
| Imputation (PCHIP) | ~1s | Spline cúbica |
| Standardization | ~0.5s | StandardScaler |
| **Total pipeline** | **~10s** | - |

---

## 🚀 Próximos Pasos / Roadmap

- [ ] Soporte para datos geoespaciales (lat/lon)
- [ ] Dashboard Streamlit para visualización interactiva
- [ ] Paralelización de pasos independientes
- [ ] Exportación a bases de datos (SQL, MongoDB)
- [ ] API REST (Flask/FastAPI)
- [ ] Modelos de predicción (ARIMA, Prophet para timeseries)

---

## 📚 Referencias Científicas

1. **Isolation Forest:** Liu et al. (2008) - "Isolation Forest" - IEEE ICDM
2. **MissForest:** Stekhoven & Bühlmann (2012) - "MissForest–non-parametric missing value imputation"
3. **Box-Cox:** Box & Cox (1964) - "An analysis of transformations"
4. **PCHIP:** Fritsch & Carlson (1980) - "Monotone piecewise cubic interpolation"

---

## 📞 Contacto & Soporte

- **Repositorio:** https://github.com/cmhh22/aeda-framework
- **Autor:** Tesista AEDA
- **Fecha:** Marzo 2026
- **Versión Actual:** 1.0 (Stable)

---

## 📋 Historial de Versiones

### v1.0 (2026-03-17) ✅ Actual
- ✅ Pipeline orchestrator (Chain of Responsibility)
- ✅ Audit decorator (@track_transformation)
- ✅ Detección outliers (Isolation Forest + reglas físicas)
- ✅ Imputación (MissForest, PCHIP)
- ✅ Estandarización (StandardScaler, Box-Cox, log1p)
- ✅ CLI completo con argparse
- ✅ Exportación multiformat con versionado SHA-256
- ✅ Configuración centralizada (YAML)
- ✅ Tests unitarios (13+ casos)

### v0.9 (Anterior)
- Estructura base del framework
- Primeros prototipos de módulos

---

## ✅ Checklist de Funcionalidad

- [x] ✅ Pipeline orchestration
- [x] ✅ ML/AI algorithms (3 modelos)
- [x] ✅ Audit logging (JSON thread-safe)
- [x] ✅ Data export (Parquet/CSV/Pickle)
- [x] ✅ CLI interface
- [x] ✅ Configuration management
- [x] ✅ Data versioning (SHA-256)
- [x] ✅ Unit tests
- [x] ✅ Git repository
- [x] ✅ GitHub upload

---

**Última actualización:** 2026-03-17 14:35 UTC  
**Estado:** ✅ Production Ready
