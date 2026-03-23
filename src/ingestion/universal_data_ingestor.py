from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, Literal

import pandas as pd

from ..data_component import DataComponent
from ..data_component_contracts import ComponentOutput, build_component_output
from .raw_data_ingestor import RawDataIngestor
from .matrix_type_detector import MatrixTypeDetector
from .censored_value_handler import CensoredValueHandler
from .data_quality_reporter import DataQualityReporter


class MatrixTypeDetectionError(Exception):
    pass


class UniversalDataIngestor(DataComponent):
    """
    Universal matrix-agnostic ingestion engine for environmental datasets.

    Key Features:
    - Automatic matrix type detection (sediment, soil, water, air, biota)
    - Intelligent handling of censored values (BDL/AQL) with multiple imputation strategies
    - Comprehensive quality reporting with scientific traceability
    - Matrix-specific preprocessing rules

    Handles complex analytical data including:
    - FRX notation (< LOD, > LOQ, ± uncertainty)
    - Missing values and data gaps
    - Concentration unit conversions
    - Physical properties (granulometry, organic matter, pH, etc.)
    
    Usage:
        ingestor = UniversalDataIngestor(
            analyte_schema={"V_(ppm)": "ppm", "Cr_(ppm)": "ppm", ...},
            matrix_type_hint="sediment"  # optional
        )
        result = ingestor.run("data.xlsx", sheet_name="RAW")
    """

    def __init__(
        self,
        analyte_schema: dict[str, str] | Iterable[str],
        metadata_columns: Iterable[str] | None = None,
        target_unit: str = "ppm",
        strict_schema: bool = False,
        censored_value_strategy: Literal["lod_half", "ros", "qmle", "percentile"] = "lod_half",
        generate_quality_report: bool = True,
        logger: logging.Logger | None = None,
    ) -> None:
        self.analyte_schema = analyte_schema
        self.metadata_columns = set(metadata_columns or [])
        self.target_unit = target_unit
        self.strict_schema = strict_schema
        self.censored_value_strategy = censored_value_strategy
        self.generate_quality_report = generate_quality_report
        self.logger = logger or logging.getLogger(__name__)
        self.matrix_detector = MatrixTypeDetector(logger=self.logger)
        self.quality_reporter = DataQualityReporter()

    @staticmethod
    def _normalize_columns(columns: Iterable[str]) -> list[str]:
        return [str(column).strip().lower().replace(" ", "_") for column in columns]

    def run(
        self,
        file_path: str | Path,
        sheet_name: str | None = None,
        matrix_type_hint: str | None = None,
    ) -> ComponentOutput:
        """
        Ingest environmental data with universal preprocessing.
        
        Args:
            file_path: Path to Excel/CSV file
            sheet_name: For Excel files, specify sheet (defaults to first if not raw_data_ingestor)
            matrix_type_hint: Optional matrix type hint (sediment, soil, water, air, biota)
            
        Returns:
            ComponentOutput with:
            - data: Processed DataFrame
            - metadata: Ingestion metadata including matrix type and censoring info
            - extra: Raw data, parsed data, quality report, and flags
        """
        self.logger.info(f"Starting universal ingestion from {file_path}")
        
        # Step 1: Run raw data ingestor
        raw_ingestor = RawDataIngestor(
            chemical_schema=self.analyte_schema,
            metadata_columns=self.metadata_columns,
            target_unit=self.target_unit,
            strict_schema=self.strict_schema,
        )
        
        result = raw_ingestor.run(file_path)
        if not isinstance(result.get("data"), pd.DataFrame):
            raise ValueError("RawDataIngestor did not return a valid DataFrame under key 'data'")
        
        raw_data = result.get("raw_data", result["data"])
        processed_data = result["data"]
        quality_flags = result.get("metadata", {}).get("ingestion", {}).get("quality_flags", {})
        
        self.logger.info(f"Raw data ingested: {processed_data.shape[0]} samples, {processed_data.shape[1]} columns")
        
        # Step 2: Detect matrix type
        detected_matrix, matrix_profile = self.matrix_detector.detect(
            processed_data,
            matrix_type_hint=matrix_type_hint,
            confidence_threshold=0.5
        )
        
        # Step 3: Handle censored values (BDL/AQL)
        censored_handler = CensoredValueHandler(
            imputation_strategy=self.censored_value_strategy,
            logger=self.logger
        )
        
        # For each chemical column, apply censored value handling
        censored_metadata = []
        for col in self.analyte_schema.keys() if isinstance(self.analyte_schema, dict) else self.analyte_schema:
            if col in processed_data.columns:
                # Try to infer LOD from metadata or use a reasonable default
                lod = self._infer_lod(processed_data[col])
                processed_data[col] = censored_handler.handle_column(
                    processed_data[col],
                    col,
                    lod
                )
        
        censored_metadata = censored_handler.metadata
        self.logger.info(f"Censored values handled: {len(censored_metadata)} columns processed")
        
        # Step 4: Generate quality report
        quality_report_text = ""
        if self.generate_quality_report:
            quality_report_text = self.quality_reporter.generate_report(
                raw_data=raw_data,
                processed_data=processed_data,
                matrix_type=detected_matrix,
                censored_metadata=censored_metadata,
                quality_flags=quality_flags
            )
            self.logger.debug(f"Quality report generated ({len(quality_report_text)} chars)")
        
        # Step 5: Build output
        lineage = {
            "matrix_type_detected": detected_matrix,
            "matrix_type_confidence": matrix_profile["confidence"],
            "matrix_detection_indicators": matrix_profile["indicators"],
            "censored_value_strategy": self.censored_value_strategy,
            "source": "UniversalDataIngestor",
        }
        lineage.update(result.get("metadata", {}))
        
        return build_component_output(
            data=processed_data,
            metadata=lineage,
            extra={
                "raw_data": raw_data,
                "parsed_data": result.get("parsed_data"),
                "quality_flags": quality_flags,
                "matrix_type_detected": detected_matrix,
                "matrix_profile": matrix_profile,
                "censored_metadata": censored_metadata,
                "quality_report": quality_report_text,
                "censored_summary": censored_handler.get_quality_report() if censored_metadata else {},
            },
        )
    
    @staticmethod
    def _infer_lod(series: pd.Series, percentile: float = 0.05) -> float:
        """
        Infer Limit of Detection from data distribution.
        Uses the lower percentile of the distribution as proxy.
        
        For real lab data, LOD should be provided externally.
        """
        numeric_series = pd.to_numeric(series, errors="coerce").dropna()
        if len(numeric_series) == 0:
            return 1.0
        
        # Use 5th percentile as LOD proxy, minimum 10% of median
        percentile_val = numeric_series.quantile(percentile)
        median_val = numeric_series.median()
        estimated_lod = max(percentile_val, median_val * 0.1)
        
        return float(estimated_lod)
