from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypedDict
from uuid import uuid4

import numpy as np
import pandas as pd
from scipy.stats import boxcox, normaltest, skew
from sklearn.preprocessing import StandardScaler

from data_component import DataComponent


TransformationName = Literal["standard_scaler", "log1p", "boxcox"]


class DistributionInfo(TypedDict):
    skewness: float
    normaltest_pvalue: float | None
    distribution: Literal["normal", "skewed", "unknown"]
    is_normal: bool


class StandardizerOutput(TypedDict):
    standardized_data: pd.DataFrame
    column_reports: list[ColumnStandardizationReport]
    metadata: dict[str, Any]
    metadata_path: str


@dataclass
class ColumnStandardizationReport:
    column: str
    n_observations: int
    skewness: float
    normaltest_pvalue: float | None
    distribution: str
    transformation: str
    parameters: dict[str, float]


@dataclass
class TransformationResult:
    transformed: pd.Series
    parameters: dict[str, float]


class DistributionAnalyzer:
    def __init__(
        self,
        normal_skew_threshold: float,
        normaltest_alpha: float,
    ) -> None:
        self.normal_skew_threshold = normal_skew_threshold
        self.normaltest_alpha = normaltest_alpha

    def analyze(self, series: pd.Series) -> DistributionInfo:
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if len(numeric) == 0:
            return {
                "skewness": 0.0,
                "normaltest_pvalue": None,
                "distribution": "unknown",
                "is_normal": False,
            }

        skewness_value = float(skew(numeric, bias=False)) if len(numeric) > 2 else 0.0

        pvalue: float | None = None
        if len(numeric) >= 8:
            try:
                _, p = normaltest(numeric)
                pvalue = float(p)
            except Exception:
                pvalue = None

        is_normal_by_pvalue = pvalue is not None and pvalue > self.normaltest_alpha
        is_normal_by_skew = abs(skewness_value) <= self.normal_skew_threshold
        is_normal = is_normal_by_pvalue or is_normal_by_skew

        return {
            "skewness": skewness_value,
            "normaltest_pvalue": pvalue,
            "distribution": "normal" if is_normal else "skewed",
            "is_normal": is_normal,
        }


class TransformStrategy(ABC):
    @abstractmethod
    def apply(self, series: pd.Series) -> TransformationResult:
        raise NotImplementedError


class StandardScalerStrategy(TransformStrategy):
    def apply(self, series: pd.Series) -> TransformationResult:
        numeric = pd.to_numeric(series, errors="coerce")
        output = pd.Series(np.nan, index=series.index, dtype=float)
        valid_mask = numeric.notna()
        if valid_mask.sum() == 0:
            return TransformationResult(transformed=output, parameters={})

        values = numeric.loc[valid_mask].to_numpy(dtype=float)
        scaler = StandardScaler()
        transformed = scaler.fit_transform(values.reshape(-1, 1)).ravel()
        output.loc[valid_mask] = transformed
        return TransformationResult(
            transformed=output,
            parameters={
                "mean": float(scaler.mean_[0]),
                "scale": float(scaler.scale_[0]),
            },
        )


class Log1PStrategy(TransformStrategy):
    def __init__(self, epsilon: float) -> None:
        self.epsilon = epsilon

    def apply(self, series: pd.Series) -> TransformationResult:
        numeric = pd.to_numeric(series, errors="coerce")
        output = pd.Series(np.nan, index=series.index, dtype=float)
        valid_mask = numeric.notna()
        if valid_mask.sum() == 0:
            return TransformationResult(transformed=output, parameters={})

        values = numeric.loc[valid_mask].to_numpy(dtype=float)
        min_value = float(np.min(values))
        shift = 0.0
        if min_value <= -1.0:
            shift = abs(min_value) + 1.0 + self.epsilon
        transformed = np.log1p(values + shift)
        output.loc[valid_mask] = transformed
        return TransformationResult(transformed=output, parameters={"shift": float(shift)})


class BoxCoxStrategy(TransformStrategy):
    def __init__(self, epsilon: float) -> None:
        self.epsilon = epsilon

    def apply(self, series: pd.Series) -> TransformationResult:
        numeric = pd.to_numeric(series, errors="coerce")
        output = pd.Series(np.nan, index=series.index, dtype=float)
        valid_mask = numeric.notna()
        if valid_mask.sum() == 0:
            return TransformationResult(transformed=output, parameters={})

        values = numeric.loc[valid_mask].to_numpy(dtype=float)
        min_value = float(np.min(values))
        shift = 0.0
        positive_values = values.copy()
        if min_value <= 0:
            shift = abs(min_value) + self.epsilon
            positive_values = values + shift

        transformed, lambda_value = boxcox(positive_values)
        output.loc[valid_mask] = transformed
        return TransformationResult(
            transformed=output,
            parameters={
                "lambda": float(lambda_value),
                "shift": float(shift),
            },
        )


class MetadataBuilder:
    @staticmethod
    def build(
        *,
        selected_columns: list[str],
        standardized_data: pd.DataFrame,
        column_reports: list[ColumnStandardizationReport],
        metadata_path: str | Path,
        dataset_name: str,
        processed_by: str,
        filters_applied: list[str],
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        report_payload = [asdict(report) for report in column_reports]
        return {
            "dataset_id": str(uuid4()),
            "dataset_name": dataset_name,
            "generated_at_utc": now,
            "fair": {
                "findable": {
                    "identifier": "uuid4",
                    "keywords": ["AEDA", "environmental-data", "standardization", "FAIR"],
                },
                "accessible": {
                    "format": "application/json",
                    "metadata_file": str(Path(metadata_path)),
                },
                "interoperable": {
                    "schema": {
                        "columns": selected_columns,
                        "dtypes": {column: str(standardized_data[column].dtype) for column in selected_columns},
                    }
                },
                "reusable": {
                    "processed_by": processed_by,
                    "processing_module": "DataStandardizer",
                    "processing_time_utc": now,
                    "filters_applied": filters_applied,
                    "transformations": report_payload,
                },
            },
            "lineage": {
                "actor": processed_by,
                "timestamp_utc": now,
                "filters_applied": filters_applied,
                "transformations_applied": report_payload,
            },
        }


class DataStandardizer(DataComponent):
    def __init__(
        self,
        normal_skew_threshold: float = 0.5,
        high_skew_threshold: float = 1.0,
        normaltest_alpha: float = 0.05,
        epsilon: float = 1e-6,
    ) -> None:
        self.normal_skew_threshold = normal_skew_threshold
        self.high_skew_threshold = high_skew_threshold
        self.normaltest_alpha = normaltest_alpha
        self.epsilon = epsilon
        self.distribution_analyzer = DistributionAnalyzer(
            normal_skew_threshold=normal_skew_threshold,
            normaltest_alpha=normaltest_alpha,
        )
        self._strategies: dict[TransformationName, TransformStrategy] = {
            "standard_scaler": StandardScalerStrategy(),
            "log1p": Log1PStrategy(epsilon=epsilon),
            "boxcox": BoxCoxStrategy(epsilon=epsilon),
        }

    @staticmethod
    def _resolve_columns(data: pd.DataFrame, columns: list[str] | None) -> list[str]:
        if columns is None:
            numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_columns:
                raise ValueError("No hay columnas numéricas para estandarizar")
            return numeric_columns

        missing = [column for column in columns if column not in data.columns]
        if missing:
            raise ValueError(f"Columnas inexistentes en DataFrame: {missing}")
        return columns

    def _select_transformation(self, skewness_value: float, series_min: float) -> TransformationName:
        abs_skew = abs(skewness_value)

        if abs_skew <= self.normal_skew_threshold:
            return "standard_scaler"

        if abs_skew <= self.high_skew_threshold:
            return "log1p"

        if series_min > 0:
            return "boxcox"

        return "log1p"

    def _apply_transformation(self, series: pd.Series, method: TransformationName) -> TransformationResult:
        strategy = self._strategies[method]
        return strategy.apply(series)

    def _write_metadata(
        self,
        metadata_path: str | Path,
        metadata: dict[str, Any],
    ) -> Path:
        destination = Path(metadata_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        return destination

    def run(
        self,
        data: pd.DataFrame,
        columns: list[str] | None = None,
        processed_by: str = "unknown",
        filters_applied: list[str] | None = None,
        metadata_path: str | Path = "data/processed/metadata.json",
        dataset_name: str = "AEDA-Framework Dataset",
    ) -> StandardizerOutput:
        if data.empty:
            raise ValueError("El DataFrame de entrada está vacío")

        selected_columns = self._resolve_columns(data, columns)
        standardized_data = data.copy()
        column_reports: list[ColumnStandardizationReport] = []

        for column in selected_columns:
            numeric_series = pd.to_numeric(standardized_data[column], errors="coerce")
            distribution_info = self.distribution_analyzer.analyze(numeric_series)
            method = self._select_transformation(
                skewness_value=distribution_info["skewness"],
                series_min=float(numeric_series.min(skipna=True)) if numeric_series.notna().any() else 0.0,
            )

            transformation_result = self._apply_transformation(numeric_series, method)
            standardized_data[column] = transformation_result.transformed

            column_reports.append(
                ColumnStandardizationReport(
                    column=column,
                    n_observations=int(numeric_series.notna().sum()),
                    skewness=float(distribution_info["skewness"]),
                    normaltest_pvalue=distribution_info["normaltest_pvalue"],
                    distribution=str(distribution_info["distribution"]),
                    transformation=method,
                    parameters=transformation_result.parameters,
                )
            )

        metadata = MetadataBuilder.build(
            selected_columns=selected_columns,
            standardized_data=standardized_data,
            column_reports=column_reports,
            metadata_path=metadata_path,
            dataset_name=dataset_name,
            processed_by=processed_by,
            filters_applied=filters_applied or [],
        )

        metadata_file = self._write_metadata(metadata_path=metadata_path, metadata=metadata)

        return {
            "standardized_data": standardized_data,
            "column_reports": column_reports,
            "metadata": metadata,
            "metadata_path": str(metadata_file),
        }
