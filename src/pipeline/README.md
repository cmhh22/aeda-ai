# AEDA_Pipeline Architecture & Usage Guide

## 📋 Overview

**AEDA_Pipeline** implements the **Chain of Responsibility** design pattern for sequential data processing in environmental data analysis pipelines. It provides robust orchestration of preprocessing steps with comprehensive error handling, logging, and state introspection.

## 🏗️ Architecture

### Components

```
AEDA_Pipeline (Main Orchestrator)
├── register_step(step) → Register processing steps
├── execute(data) → Execute chain sequences
└── get_execution_report() → Retrieve execution metrics

PipelineStep (Abstract Base)
├── name: str → Unique identifier
├── execute(data: DataFrame) → Transform data
└── get_metadata() → Step metrics

PipelineLogger (Context & State)
├── capture_snapshot(data, step_name) → Save DataFrame state
├── log_event(event_type, detail) → Record execution events
├── generate_failure_report() → Detailed error context
└── save_logs() → Persist logs to JSON
```

### Design Pattern: Chain of Responsibility

Each `PipelineStep` is a handler in the chain:
- **Step 1 (Data Ingestion)** → reads raw data
- **Step 2 (Outlier Detection)** → cleans anomalies
- **Step 3 (Reconstruction)** → imputes missing values
- **Step 4 (Standardization)** → normalizes scales

If any step fails, the pipeline:
1. ✋ **Stops execution** (configurable)
2. 📸 **Captures DataFrame snapshot** (state before failure)
3. 📝 **Logs detailed context** (dtypes, nulls, sample data)
4. 💾 **Saves failure report** to disk

## 🚀 Quick Start

### 1. Define Custom Steps

```python
from pipeline_step import PipelineStep
import pandas as pd

class CustomProcessingStep(PipelineStep):
    def __init__(self):
        super().__init__(
            name="my_custom_step",
            description="Apply custom transformation"
        )
    
    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        # Your processing logic
        transformed = data.copy()
        transformed['new_column'] = transformed['source_col'].apply(lambda x: x * 2)
        return transformed
```

### 2. Build Pipeline

```python
from aeda_pipeline import AEDA_Pipeline

pipeline = AEDA_Pipeline(
    name="environmental_data_pipeline",
    log_dir="pipeline_logs",
    capture_snapshots=True  # Enable state snapshots
)

# Register steps (fluent interface)
pipeline.register_step(DataIngestionStep())
pipeline.register_step(OutlierDetectionStep())
pipeline.register_step(DataReconstructionStep())
pipeline.register_step(DataStandardizationStep())
```

### 3. Execute Pipeline

```python
data = pd.read_csv("environmental_data.csv")

try:
    result = pipeline.execute(
        data,
        stop_on_error=True,      # Stop on first failure
        partial_ok=False         # Don't return partial results
    )
    print(f"Pipeline completed. Output shape: {result.shape}")
    
except PipelineExecutionError as e:
    print(f"Pipeline failed: {e}")
    # Failure report is automatically saved
```

### 4. Inspect Results

```python
# Get execution report
report = pipeline.get_execution_report()

print(f"Total execution time: {report['total_execution_time_seconds']:.2f}s")
print(f"Failed at: {report['failed_at_step']}")

for step in report['steps']:
    print(f"  {step['name']}: {step['execution_time']:.3f}s, "
          f"rows_affected={step['rows_affected']}")

# Save logs
log_file = pipeline.save_execution_logs()
print(f"Logs saved to: {log_file}")
```

## 📊 Features

### 1. **Automatic State Snapshots**
Before each step executes, AEDA_Pipeline captures:
- DataFrame shape (rows × columns)
- Memory usage
- Data types for all columns
- Null counts per column
- Basic statistics (min, max, mean, std) for numeric columns
- Sample rows (first N rows)

### 2. **Detailed Error Context**
When a step fails, the failure report includes:
- Exception type and message
- DataFrame state before failed step
- Complete execution log
- All captured snapshots
- Step metadata

### 3. **Flexible Execution**

```python
# Execute all steps
result = pipeline.execute(data)

# Execute starting from specific step (skip earlier steps)
result = pipeline.execute_from(data, step_name="outlier_detection")

# Continue on error, but skip failed steps
result = pipeline.execute(data, stop_on_error=False, partial_ok=True)
```

### 4. **Method Chaining**
```python
result = (AEDA_Pipeline("my_pipeline")
    .register_step(Step1())
    .register_step(Step2())
    .register_step(Step3())
    .execute(data))
```

## 📝 Real-World Example

```python
from pipeline.aeda_pipeline import AEDA_Pipeline
from pipeline.example_usage import (
    RawDataIngestorStep,
    OutlierDetectionStep,
    DataReconstructionStep,
    DataStandardizationStep
)

# Define schema
chemical_schema = {
    "O3": "ppm",           # Ozone
    "NO2": "ppm",          # Nitrogen dioxide
    "PM25": "µg/m³",       # Fine particulate
    "PM10": "µg/m³",       # Particulate matter
    "SO2": "ppm",          # Sulfur dioxide
}

# Define domain knowledge limits
physical_limits = {
    "O3": (0, 200),
    "NO2": (0, 500),
    "PM25": (0, 200),
    "PM10": (0, 300),
    "SO2": (0, 500),
}

# Build pipeline
pipeline = AEDA_Pipeline(
    name="air_quality_processing",
    log_dir="air_quality_logs"
)

pipeline.register_steps([
    RawDataIngestorStep(chemical_schema),
    OutlierDetectionStep(physical_limits),
    DataReconstructionStep(strategy="missforest"),
    DataStandardizationStep(),
])

# Execute
data = pd.read_csv("air_quality_data.csv")
try:
    clean_data = pipeline.execute(data)
    print(f"✓ Processing completed: {data.shape} → {clean_data.shape}")
except PipelineExecutionError as e:
    print(f"✗ Processing failed: {e}")
```

## 🔍 Implementation Details

### PipelineStep Lifecycle

1. **Registration**: Step added to pipeline
2. **Snapshot**: DataFrame state captured (if enabled)
3. **Execution**: `execute()` method called
4. **Validation**: Output type verified (must be DataFrame)
5. **Metrics**: Execution time, rows/columns affected tracked
6. **Logging**: Success or error logged

### Memory Efficiency

- Snapshots are point-in-time captures (not full copies)
- Sample data limited to configurable number of rows (default: 5)
- Numeric statistics only for numeric columns
- Logs stored as JSON (highly compressible)

### Error Recovery

```python
# Scenario: Step 3 fails
pipeline = AEDA_Pipeline()
pipeline.register_steps([
    Step1(),  # ✓ Executes
    Step2(),  # ✓ Executes
    Step3(),  # ✗ Fails here
    Step4(),  # ⏭️ Skipped
])

result = pipeline.execute(data)
# File: pipeline_logs/aeda_pipeline_20260317_161200.json
# Contains all snapshots and execution trace for debugging
```

## 📋 Testing

Unit tests (13+ cases) in `test_aeda_pipeline.py`:

```bash
pytest src/pipeline/test_aeda_pipeline.py -v
```

Covers:
- Single and multiple step execution
- Error handling and recovery
- State snapshots and logging
- Method chaining
- Execution reports
- Partial execution
- Type validation

## 🎯 Best Practices

1. **Explicit Step Names**: Use descriptive names for debugging
2. **Error Messages**: Provide clear error descriptions
3. **Type Validation**: Always return DataFrame from `execute()`
4. **Documentation**: Describe step transformations
5. **Testing**: Unit test each step independently
6. **Logging**: Review JSON logs for production issues

## 📦 Files Structure

```
src/pipeline/
├── __init__.py                 # Module exports
├── aeda_pipeline.py           # Main orchestrator (250+ lines)
├── pipeline_step.py           # Abstract base class
├── pipeline_logger.py         # Logging & snapshots (250+ lines)
├── example_usage.py           # Real-world example
├── test_aeda_pipeline.py      # 13+ unit tests
└── README.md                  # This guide
```

## 🔗 Integration with AEDA Modules

Pipeline perfectly integrates with existing AEDA components:

```
AEDA_Pipeline orchestrates:
├── RawDataIngestor (reading + validation)
├── OutlierDetector (hybrid rule-based + ML)
├── DataReconstructor (PCHIP + missForest)
└── DataStandardizer (distribution-aware transforms)
```

Each module's output becomes next step's input ✓

---

**Next Steps**: Execute the example with real data to see complete pipeline in action!
