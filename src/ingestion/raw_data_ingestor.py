from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd

from ..data_component import DataComponent
from ..data_component_contracts import ComponentOutput, build_component_output


class FileReadError(Exception):
    pass


class SchemaValidationError(Exception):
    pass


class DataValidationError(Exception):
    pass


class UnitConversionError(Exception):
    pass


class RawDataIngestor(DataComponent):
    def __init__(
        self,
        chemical_schema: dict[str, str] | Iterable[str],
        metadata_columns: Iterable[str] | None = None,
        target_unit: str = "%",
        strict_schema: bool = False,
    ) -> None:
        self.chemical_units = self._normalize_schema(chemical_schema)
        self.chemical_columns = list(self.chemical_units.keys())
        self.metadata_columns = set(metadata_columns or [])
        self.target_unit = target_unit
        self.strict_schema = strict_schema
        self.conversion_registry: dict[tuple[str, str], Callable[[pd.Series], pd.Series]] = {
            ("ppm", "%"): lambda series: series / 10000,
            ("%", "ppm"): lambda series: series * 10000,
        }
        self._lt_pattern = re.compile(r"^\s*<\s*(-?\d+(?:\.\d+)?)\s*$")
        self._gt_pattern = re.compile(
            r"^\s*>\s*(-?\d+(?:\.\d+)?)(?:\s*\(\s*(-?\d+(?:\.\d+)?)\s*\))?\s*$"
        )
        self._plus_minus_pattern = re.compile(
            r"^\s*(-?\d+(?:\.\d+)?)\s*(?:±|\+/-)\s*(\d+(?:\.\d+)?)\s*$"
        )

    @staticmethod
    def _normalize_schema(chemical_schema: dict[str, str] | Iterable[str]) -> dict[str, str]:
        if isinstance(chemical_schema, dict):
            if not chemical_schema:
                raise ValueError("chemical_schema cannot be empty")
            return {column: str(unit).strip() for column, unit in chemical_schema.items()}

        columns = [column for column in chemical_schema]
        if not columns:
            raise ValueError("chemical_schema cannot be empty")
        return {column: "ppm" for column in columns}

    def _read_file(self, file_path: str | Path) -> pd.DataFrame:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileReadError(f"File not found: {file_path}")

        try:
            if file_path.suffix.lower() == ".csv":
                return pd.read_csv(file_path)
            if file_path.suffix.lower() in {".xlsx", ".xls"}:
                return pd.read_excel(file_path)
        except (pd.errors.EmptyDataError, UnicodeDecodeError, ValueError) as error:
            raise FileReadError(f"Failed to read file {file_path}: {error}") from error
        except Exception as error:
            raise FileReadError(f"Unexpected error reading {file_path}: {error}") from error

        raise FileReadError(
            f"Unsupported format: {file_path.suffix}. Use CSV or Excel files (.xlsx/.xls)."
        )

    def _validate_schema(self, data: pd.DataFrame) -> None:
        expected = set(self.chemical_columns)
        present = set(data.columns)
        missing = sorted(expected - present)
        if missing:
            raise SchemaValidationError(
                f"Missing chemical columns in file: {missing}. Expected schema: {sorted(expected)}"
            )

        if self.strict_schema:
            allowed = expected.union(self.metadata_columns)
            extras = sorted(present - allowed)
            if extras:
                raise SchemaValidationError(
                    f"Schema validation failed - unexpected columns: {extras}. Allowed columns: {sorted(allowed)}"
                )

    def _validate_non_negative(self, data: pd.DataFrame) -> None:
        for column in self.chemical_columns:
            numeric_column = pd.to_numeric(data[column], errors="coerce")
            invalid_mask = numeric_column < 0
            if invalid_mask.any():
                invalid_rows = data.index[invalid_mask].tolist()
                raise DataValidationError(
                    f"Negative concentrations detected in '{column}' at rows {invalid_rows}."
                )

    def _convert_units(self, data: pd.DataFrame) -> pd.DataFrame:
        converted = data.copy()

        for column, source_unit in self.chemical_units.items():
            source = source_unit.lower().strip()
            target = self.target_unit.lower().strip()

            if source == target:
                continue

            conversion_fn = self.conversion_registry.get((source, target))
            if conversion_fn is None:
                raise UnitConversionError(
                    f"No unit conversion implemented from '{source_unit}' to '{self.target_unit}' for '{column}'."
                )

            numeric_series = pd.to_numeric(converted[column], errors="coerce")
            converted[column] = conversion_fn(numeric_series)

        return converted

    def _parse_analytical_value(self, value: Any) -> tuple[float | None, str | None, float | None]:
        if value is None:
            return None, None, None

        if isinstance(value, (int, float)) and pd.notna(value):
            return float(value), None, None

        text_value = str(value).strip()
        if text_value == "" or text_value.lower() in {"nan", "none"}:
            return None, None, None

        lt_match = self._lt_pattern.match(text_value)
        if lt_match:
            lod = float(lt_match.group(1))
            return lod, "lt_lod", None

        gt_match = self._gt_pattern.match(text_value)
        if gt_match:
            real_value = gt_match.group(2)
            parsed = float(real_value) if real_value is not None else float(gt_match.group(1))
            return parsed, "gt_limit", None

        plus_minus_match = self._plus_minus_pattern.match(text_value)
        if plus_minus_match:
            return (
                float(plus_minus_match.group(1)),
                "has_uncertainty",
                float(plus_minus_match.group(2)),
            )

        normalized = text_value.replace(",", "")
        try:
            return float(normalized), None, None
        except ValueError:
            return None, "parse_error", None

    def _parse_analytical_columns(self, data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        parsed_data = data.copy()
        quality_flags: dict[str, dict[str, int]] = {}

        for column in self.chemical_columns:
            if column not in parsed_data.columns:
                continue

            parsed_values: list[float | None] = []
            uncertainty_values: list[float | None] = []
            counts = {
                "lt_lod": 0,
                "gt_limit": 0,
                "has_uncertainty": 0,
                "parse_error": 0,
            }

            for raw_value in parsed_data[column]:
                numeric_value, flag, uncertainty = self._parse_analytical_value(raw_value)
                parsed_values.append(numeric_value)
                uncertainty_values.append(uncertainty)
                if flag is not None:
                    counts[flag] += 1

            parsed_series = pd.to_numeric(pd.Series(parsed_values, index=parsed_data.index), errors="coerce")
            parsed_data[column] = parsed_series

            uncertainty_column = f"U_{column}"
            if uncertainty_column in parsed_data.columns:
                existing_uncertainty = pd.to_numeric(parsed_data[uncertainty_column], errors="coerce")
                parsed_uncertainty = pd.to_numeric(pd.Series(uncertainty_values, index=parsed_data.index), errors="coerce")
                parsed_data[uncertainty_column] = existing_uncertainty.combine_first(parsed_uncertainty)
            elif any(value is not None for value in uncertainty_values):
                parsed_data[uncertainty_column] = pd.to_numeric(
                    pd.Series(uncertainty_values, index=parsed_data.index),
                    errors="coerce",
                )

            quality_flags[column] = counts

        return parsed_data, {
            "quality_flags": quality_flags,
        }

    def run(self, file_path: str | Path) -> ComponentOutput:
        data = self._read_file(file_path)
        self._validate_schema(data)
        parsed_data, parsing_metadata = self._parse_analytical_columns(data)
        self._validate_non_negative(parsed_data)
        converted_data = self._convert_units(parsed_data)

        metadata = {
            "ingestion": {
                "source_file": str(Path(file_path)),
                "target_unit": self.target_unit,
                "strict_schema": self.strict_schema,
                **parsing_metadata,
            }
        }

        return build_component_output(
            data=converted_data,
            metadata=metadata,
            extra={
                "raw_data": data,
                "parsed_data": parsed_data,
            },
        )
