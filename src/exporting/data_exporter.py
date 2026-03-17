from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class ExportResult:
    data_hash: str
    timestamp_utc: str
    parquet_path: str
    csv_path: str
    model_pickle_paths: dict[str, str]


class DataExporter:
    """Exports processed datasets and trained artifacts with reproducible naming."""

    def __init__(
        self,
        output_dir: str | Path = "data/processed/exports",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def compute_data_hash(data: pd.DataFrame) -> str:
        """Compute a stable hash for a DataFrame version."""
        schema_part = "|".join(f"{column}:{dtype}" for column, dtype in data.dtypes.items())
        row_hashes = pd.util.hash_pandas_object(data, index=True).to_numpy()

        digest = hashlib.sha256()
        digest.update(schema_part.encode("utf-8"))
        digest.update(row_hashes.tobytes())
        return digest.hexdigest()[:12]

    @staticmethod
    def _timestamp_utc() -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    def _build_name(self, base_name: str, timestamp_utc: str, data_hash: str, extension: str) -> str:
        safe_base = base_name.strip().replace(" ", "_")
        return f"{safe_base}_{timestamp_utc}_{data_hash}.{extension}"

    def export_dataframes(
        self,
        data: pd.DataFrame,
        base_name: str = "dataset",
        include_index: bool = False,
        parquet_compression: str = "snappy",
    ) -> tuple[str, str, str, str]:
        """
        Export the same dataset to Parquet and CSV.

        Returns:
            tuple[data_hash, timestamp_utc, parquet_path, csv_path]
        """
        timestamp_utc = self._timestamp_utc()
        data_hash = self.compute_data_hash(data)

        parquet_file = self.output_dir / self._build_name(base_name, timestamp_utc, data_hash, "parquet")
        csv_file = self.output_dir / self._build_name(base_name, timestamp_utc, data_hash, "csv")

        data.to_parquet(parquet_file, index=include_index, compression=parquet_compression)
        data.to_csv(csv_file, index=include_index)

        return data_hash, timestamp_utc, str(parquet_file), str(csv_file)

    def export_model_pickle(
        self,
        model_object: Any,
        model_name: str,
        data_hash: str,
        timestamp_utc: str | None = None,
    ) -> str:
        """
        Export a trained model/transformer object to Pickle using data-versioned naming.
        """
        ts = timestamp_utc or self._timestamp_utc()
        pickle_file = self.output_dir / self._build_name(model_name, ts, data_hash, "pkl")

        with open(pickle_file, "wb") as handle:
            pickle.dump(model_object, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return str(pickle_file)

    def export_all(
        self,
        data: pd.DataFrame,
        base_name: str = "dataset",
        model_objects: dict[str, Any] | None = None,
        include_index: bool = False,
    ) -> ExportResult:
        """
        Export dataset to Parquet and CSV and optional model artifacts to Pickle.
        """
        data_hash, timestamp_utc, parquet_path, csv_path = self.export_dataframes(
            data=data,
            base_name=base_name,
            include_index=include_index,
        )

        model_pickle_paths: dict[str, str] = {}
        for model_name, model_object in (model_objects or {}).items():
            model_pickle_paths[model_name] = self.export_model_pickle(
                model_object=model_object,
                model_name=model_name,
                data_hash=data_hash,
                timestamp_utc=timestamp_utc,
            )

        return ExportResult(
            data_hash=data_hash,
            timestamp_utc=timestamp_utc,
            parquet_path=parquet_path,
            csv_path=csv_path,
            model_pickle_paths=model_pickle_paths,
        )
