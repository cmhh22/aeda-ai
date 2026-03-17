from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pyod.models.iforest import IForest
from sklearn.impute import SimpleImputer

from data_component import DataComponent


DEFAULT_NAAQS_LIMITS: dict[str, tuple[float | None, float | None]] = {
    "PM2_5": (0.0, 35.0),
    "PM10": (0.0, 150.0),
    "NO2": (0.0, 100.0),
    "SO2": (0.0, 75.0),
    "CO": (0.0, 9.0),
    "O3": (0.0, 70.0),
}


@dataclass
class OutlierReport:
    total_rows: int
    total_columns_evaluated: int
    rule_based_outlier_cells: int
    rule_based_outlier_rows: int
    iforest_outlier_rows: int
    combined_outlier_rows: int


class OutlierDetector(DataComponent):
    def __init__(
        self,
        physical_limits: dict[str, tuple[float | None, float | None]],
        contamination: float = 0.05,
        random_state: int = 42,
    ) -> None:
        if not physical_limits:
            raise ValueError("physical_limits no puede estar vacío")
        if contamination <= 0 or contamination >= 0.5:
            raise ValueError("contamination debe estar en el intervalo (0, 0.5)")

        self.physical_limits = physical_limits
        self.contamination = contamination
        self.random_state = random_state

    def _rule_based_mask(self, data: pd.DataFrame) -> pd.DataFrame:
        mask = pd.DataFrame(False, index=data.index, columns=data.columns)

        for column, (min_value, max_value) in self.physical_limits.items():
            if column not in data.columns:
                continue

            numeric_series = pd.to_numeric(data[column], errors="coerce")
            col_mask = pd.Series(False, index=data.index)

            if min_value is not None:
                col_mask = col_mask | (numeric_series < min_value)
            if max_value is not None:
                col_mask = col_mask | (numeric_series > max_value)

            mask.loc[col_mask, column] = True

        return mask

    @staticmethod
    def _select_feature_columns(data: pd.DataFrame, candidate_columns: list[str]) -> list[str]:
        selected: list[str] = []
        for column in candidate_columns:
            if column not in data.columns:
                continue
            numeric_series = pd.to_numeric(data[column], errors="coerce")
            if numeric_series.notna().any():
                selected.append(column)
        return selected

    def _iforest_mask(self, data: pd.DataFrame, feature_columns: list[str]) -> pd.Series:
        if not feature_columns:
            return pd.Series(False, index=data.index)

        numeric_frame = data[feature_columns].apply(pd.to_numeric, errors="coerce")
        imputer = SimpleImputer(strategy="median")
        imputed_array = imputer.fit_transform(numeric_frame)

        detector = IForest(
            contamination=self.contamination,
            random_state=self.random_state,
        )
        detector.fit(imputed_array)
        labels = detector.predict(imputed_array)

        return pd.Series(labels == 1, index=data.index)

    def run(self, data: pd.DataFrame) -> dict[str, Any]:
        if data.empty:
            raise ValueError("El DataFrame de entrada está vacío")

        cleaned_data = data.copy()

        rule_cell_mask = self._rule_based_mask(cleaned_data)
        rule_row_mask = rule_cell_mask.any(axis=1)

        feature_columns = self._select_feature_columns(cleaned_data, list(self.physical_limits.keys()))
        iforest_row_mask = self._iforest_mask(cleaned_data, feature_columns)

        combined_row_mask = rule_row_mask | iforest_row_mask

        cleaned_data = cleaned_data.mask(rule_cell_mask)
        if feature_columns:
            cleaned_data.loc[iforest_row_mask, feature_columns] = np.nan

        report = OutlierReport(
            total_rows=len(cleaned_data),
            total_columns_evaluated=len(feature_columns),
            rule_based_outlier_cells=int(rule_cell_mask.to_numpy().sum()),
            rule_based_outlier_rows=int(rule_row_mask.sum()),
            iforest_outlier_rows=int(iforest_row_mask.sum()),
            combined_outlier_rows=int(combined_row_mask.sum()),
        )

        return {
            "cleaned_data": cleaned_data,
            "outlier_mask": combined_row_mask,
            "rule_based_mask": rule_cell_mask,
            "iforest_mask": iforest_row_mask,
            "report": report,
            "report_dict": report.__dict__,
        }
