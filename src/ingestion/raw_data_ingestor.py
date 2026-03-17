from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd

from data_component import DataComponent


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

    @staticmethod
    def _normalize_schema(chemical_schema: dict[str, str] | Iterable[str]) -> dict[str, str]:
        if isinstance(chemical_schema, dict):
            if not chemical_schema:
                raise ValueError("chemical_schema no puede estar vacío")
            return {column: str(unit).strip() for column, unit in chemical_schema.items()}

        columns = [column for column in chemical_schema]
        if not columns:
            raise ValueError("chemical_schema no puede estar vacío")
        return {column: "ppm" for column in columns}

    def _read_file(self, file_path: str | Path) -> pd.DataFrame:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileReadError(f"Archivo no encontrado: {file_path}")

        try:
            if file_path.suffix.lower() == ".csv":
                return pd.read_csv(file_path)
            if file_path.suffix.lower() in {".xlsx", ".xls"}:
                return pd.read_excel(file_path)
        except (pd.errors.EmptyDataError, UnicodeDecodeError, ValueError) as error:
            raise FileReadError(f"No se pudo leer el archivo {file_path}: {error}") from error
        except Exception as error:
            raise FileReadError(f"Error inesperado leyendo {file_path}: {error}") from error

        raise FileReadError(
            f"Formato no soportado: {file_path.suffix}. Use archivos CSV o Excel (.xlsx/.xls)."
        )

    def _validate_schema(self, data: pd.DataFrame) -> None:
        expected = set(self.chemical_columns)
        present = set(data.columns)
        missing = sorted(expected - present)
        if missing:
            raise SchemaValidationError(
                f"Columnas químicas faltantes en el archivo: {missing}. Esquema esperado: {sorted(expected)}"
            )

        if self.strict_schema:
            allowed = expected.union(self.metadata_columns)
            extras = sorted(present - allowed)
            if extras:
                raise SchemaValidationError(
                    f"Columnas no permitidas por el esquema: {extras}. Columnas permitidas: {sorted(allowed)}"
                )

    def _validate_non_negative(self, data: pd.DataFrame) -> None:
        for column in self.chemical_columns:
            numeric_column = pd.to_numeric(data[column], errors="coerce")
            invalid_mask = numeric_column < 0
            if invalid_mask.any():
                invalid_rows = data.index[invalid_mask].tolist()
                raise DataValidationError(
                    f"Se detectaron concentraciones negativas en '{column}' en filas {invalid_rows}."
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
                    f"No existe conversión implementada de '{source_unit}' a '{self.target_unit}' para '{column}'."
                )

            numeric_series = pd.to_numeric(converted[column], errors="coerce")
            converted[column] = conversion_fn(numeric_series)

        return converted

    def run(self, file_path: str | Path) -> pd.DataFrame:
        data = self._read_file(file_path)
        self._validate_schema(data)
        self._validate_non_negative(data)
        return self._convert_units(data)
