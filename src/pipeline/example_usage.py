"""
Example usage of AEDA_Pipeline with real preprocessing steps.
Demonstrates Chain of Responsibility pattern with error handling.
"""

from pathlib import Path

import pandas as pd

from pipeline.aeda_pipeline import AEDA_Pipeline
from pipeline.pipeline_step import PipelineStep

# Import actual preprocessing modules from outside pipeline package
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.raw_data_ingestor import RawDataIngestor
from preprocessing.outlier_detector import OutlierDetector
from preprocessing.data_reconstructor import DataReconstructor
from preprocessing.data_standardizer import DataStandardizer


class RawDataIngestorStep(PipelineStep):
    """Wrapper to integrate RawDataIngestor into pipeline."""

    def __init__(
        self,
        chemical_schema: dict[str, str],
        file_path: str | Path | None = None,
        target_unit: str = "%",
    ):
        super().__init__(
            name="raw_data_ingestion",
            description="Read and validate raw environmental data from file",
        )
        self.ingestor = RawDataIngestor(
            chemical_schema=chemical_schema,
            target_unit=target_unit,
        )
        self.file_path = file_path

    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.file_path:
            result = self.ingestor.run(str(self.file_path))
            return result["data"]
        # If no file provided, assume data is already loaded
        # and just validate schema
        self.ingestor._validate_schema(data)
        return data


class OutlierDetectionStep(PipelineStep):
    """Wrapper to integrate OutlierDetector into pipeline."""

    def __init__(self, physical_limits: dict[str, tuple[float | None, float | None]]):
        super().__init__(
            name="outlier_detection",
            description="Detect and mark outliers using hybrid approach (rules + IForest)",
        )
        self.detector = OutlierDetector(
            physical_limits=physical_limits,
            contamination=0.05,
        )

    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        result = self.detector.run(data)
        return result["data"]


class DataReconstructionStep(PipelineStep):
    """Wrapper to integrate DataReconstructor into pipeline."""

    def __init__(
        self,
        strategy: str = "missforest",
        columns: list[str] | None = None,
    ):
        super().__init__(
            name="data_reconstruction",
            description="Impute missing values using specified strategy (PCHIP or missForest)",
        )
        self.reconstructor = DataReconstructor()
        self.strategy = strategy
        self.columns = columns

    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.strategy == "pchip":
            return self.reconstructor.reconstruct_time_series_pchip(
                data, columns=self.columns
            )
        elif self.strategy == "missforest":
            return self.reconstructor.reconstruct_tabular_missforest(
                data, columns=self.columns
            )
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")


class DataStandardizationStep(PipelineStep):
    """Wrapper to integrate DataStandardizer into pipeline."""

    def __init__(self, columns: list[str] | None = None):
        super().__init__(
            name="data_standardization",
            description="Detect distribution and apply intelligent standardization transformations",
        )
        self.standardizer = DataStandardizer()
        self.columns = columns

    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        result = self.standardizer.run(data, columns=self.columns)
        return result["data"]


# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = AEDA_Pipeline(
        name="environmental_data_pipeline",
        log_dir="pipeline_logs",
        capture_snapshots=True,
    )

    # Define chemical schema for environmental data
    chemical_schema = {
        "O3": "ppm",        # Ozone
        "NO2": "ppm",       # Nitrogen dioxide
        "PM25": "µg/m³",    # Fine particulate matter
        "PM10": "µg/m³",    # Particulate matter
        "SO2": "ppm",       # Sulfur dioxide
    }

    # Define physical limits (domain knowledge)
    physical_limits = {
        "O3": (0, 200),      # ppb (typical air quality range)
        "NO2": (0, 500),
        "PM25": (0, 200),    # µg/m³
        "PM10": (0, 300),
        "SO2": (0, 500),
    }

    # Register pipeline steps (Chain of Responsibility)
    pipeline.register_steps([
        # Step 1: Data ingestion and validation
        RawDataIngestorStep(
            chemical_schema=chemical_schema,
            target_unit="%",
        ),
        # Step 2: Outlier detection and soft-cleaning
        OutlierDetectionStep(physical_limits=physical_limits),
        # Step 3: Missing data imputation
        DataReconstructionStep(strategy="missforest", columns=list(chemical_schema.keys())),
        # Step 4: Standardization and transformation
        DataStandardizationStep(columns=list(chemical_schema.keys())),
    ])

    # Print pipeline structure
    print("Pipeline Structure:")
    print("=" * 80)
    for step_info in pipeline.get_steps_summary():
        print(f"  • {step_info['name']}: {step_info['description']}")
    print("=" * 80)
    print()

    # Example: Create synthetic data for demonstration
    data = pd.DataFrame({
        "O3": [50, 120, 250, 45, 60, None, 75, 80, 85],
        "NO2": [30, 45, 200, 35, 40, 50, None, 55, 60],
        "PM25": [25, 30, 300, 28, 32, 35, 38, None, 40],
        "PM10": [40, 50, 400, 45, 48, 50, 52, 55, None],
        "SO2": [10, 15, 450, 12, 14, 16, 18, 20, 22],
    })

    print(f"Input Data Shape: {data.shape}")
    print(f"Input Data Nulls:\n{data.isna().sum()}\n")

    try:
        # Execute pipeline
        result = pipeline.execute(data, stop_on_error=True)

        print(f"Output Data Shape: {result.shape}")
        print(f"Output Data Nulls:\n{result.isna().sum()}\n")

        # Get execution report
        report = pipeline.get_execution_report()
        print("Pipeline Execution Report:")
        print("-" * 80)
        for key, value in report.items():
            if key != "steps":
                print(f"  {key}: {value}")
        print()
        print("Step Details:")
        for step_info in report["steps"]:
            status = "✓ PASS" if step_info["success"] else "✗ FAIL"
            print(
                f"  {status} {step_info['name']} "
                f"({step_info['execution_time']:.3f}s)"
            )
        print("-" * 80)

        # Save logs
        log_file = pipeline.save_execution_logs()
        print(f"\nLogs saved to: {log_file}")

    except Exception as e:
        print(f"\n❌ Pipeline Error:\n{str(e)}")
