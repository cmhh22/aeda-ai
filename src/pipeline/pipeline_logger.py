from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class DataFrameSnapshot:
    """Captures the state of a DataFrame at a point in time."""

    timestamp: str  # ISO 8601 format
    step_name: str
    shape: tuple[int, int]  # (rows, columns)
    memory_usage_mb: float
    dtypes: dict[str, str]  # column -> dtype string
    null_counts: dict[str, int]  # column -> count of nulls
    numeric_stats: dict[str, dict[str, Any]]  # column -> {min, max, mean, std}
    sample_rows: int  # First N rows captured
    sample_data: list[dict[str, Any]]  # Serializable sample


class PipelineLogger:
    """
    Logging system for AEDA_Pipeline.
    Captures DataFrame snapshots, step progress, and error context.
    """

    def __init__(self, log_dir: str | Path = "logs") -> None:
        """
        Initialize pipeline logger.

        Args:
            log_dir: Directory where logs will be saved
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.snapshots: list[DataFrameSnapshot] = []
        self.events: list[dict[str, Any]] = []
        self.session_start = datetime.utcnow().isoformat()

    def capture_snapshot(
        self,
        data: pd.DataFrame,
        step_name: str,
        sample_rows: int = 5,
    ) -> DataFrameSnapshot:
        """
        Create a snapshot of DataFrame state.

        Args:
            data: DataFrame to snapshot
            step_name: Name of the pipeline step
            sample_rows: Number of sample rows to capture

        Returns:
            DataFrameSnapshot object
        """
        # Numeric statistics
        numeric_stats: dict[str, dict[str, Any]] = {}
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            col_data = data[col].dropna()
            if len(col_data) > 0:
                numeric_stats[str(col)] = {
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                    "mean": float(col_data.mean()),
                    "std": float(col_data.std()),
                    "count": int(len(col_data)),
                }

        # Serialize sample data
        sample_data = data.head(sample_rows).to_dict(orient="records")
        for record in sample_data:
            for key, value in record.items():
                if pd.isna(value):
                    record[key] = None
                elif isinstance(value, (np.integer, np.floating)):
                    record[key] = float(value)

        snapshot = DataFrameSnapshot(
            timestamp=datetime.utcnow().isoformat(),
            step_name=step_name,
            shape=data.shape,
            memory_usage_mb=float(data.memory_usage(deep=True).sum() / 1024 ** 2),
            dtypes={col: str(dtype) for col, dtype in data.dtypes.items()},
            null_counts={col: int(data[col].isna().sum()) for col in data.columns},
            numeric_stats=numeric_stats,
            sample_rows=sample_rows,
            sample_data=sample_data,
        )
        self.snapshots.append(snapshot)
        return snapshot

    def log_event(
        self,
        event_type: str,
        step_name: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """
        Log a pipeline event (start, complete, error, warning).

        Args:
            event_type: Type of event ('start', 'complete', 'error', 'warning')
            step_name: Name of the step
            message: Human-readable message
            details: Additional context dictionary
        """
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "step_name": step_name,
            "message": message,
            "details": details or {},
        }
        self.events.append(event)

    def generate_failure_report(
        self,
        failed_step: str,
        error: Exception,
        previous_snapshot: DataFrameSnapshot | None = None,
    ) -> str:
        """
        Generate a detailed failure report when a step fails.

        Args:
            failed_step: Name of the step that failed
            error: The exception that occurred
            previous_snapshot: State before the failed step

        Returns:
            Formatted failure report as string
        """
        report = []
        report.append("=" * 80)
        report.append("AEDA PIPELINE FAILURE REPORT")
        report.append("=" * 80)
        report.append(f"Session: {self.session_start}")
        report.append(f"Failed Step: {failed_step}")
        report.append(f"Failure Time: {datetime.utcnow().isoformat()}")
        report.append("")

        report.append("ERROR DETAILS:")
        report.append("-" * 80)
        report.append(f"Exception Type: {type(error).__name__}")
        report.append(f"Error Message: {str(error)}")
        report.append("")

        if previous_snapshot:
            report.append("DATA STATE BEFORE FAILURE:")
            report.append("-" * 80)
            report.append(f"Shape: {previous_snapshot.shape[0]} rows × {previous_snapshot.shape[1]} columns")
            report.append(f"Memory: {previous_snapshot.memory_usage_mb:.2f} MB")
            report.append("\nColumn Data Types:")
            for col, dtype in previous_snapshot.dtypes.items():
                null_count = previous_snapshot.null_counts.get(col, 0)
                report.append(f"  {col}: {dtype} (nulls: {null_count})")
            report.append("\nNumeric Statistics:")
            for col, stats in previous_snapshot.numeric_stats.items():
                report.append(f"  {col}: min={stats.get('min')}, max={stats.get('max')}, mean={stats.get('mean'):.2f}")
            report.append("\nSample Data (first rows):")
            for idx, row in enumerate(previous_snapshot.sample_data, 1):
                report.append(f"  Row {idx}: {row}")
            report.append("")

        report.append("PIPELINE EXECUTION LOG:")
        report.append("-" * 80)
        for event in self.events:
            report.append(
                f"[{event['timestamp']}] {event['event_type'].upper()}: "
                f"{event['step_name']} - {event['message']}"
            )
        report.append("")
        report.append("=" * 80)

        return "\n".join(report)

    def save_logs(self, pipeline_name: str = "aeda_pipeline") -> Path:
        """
        Save all logs and snapshots to disk.

        Args:
            pipeline_name: Name prefix for log files

        Returns:
            Path to the saved log file
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"{pipeline_name}_{timestamp}.json"

        log_data = {
            "session_start": self.session_start,
            "session_end": datetime.utcnow().isoformat(),
            "total_snapshots": len(self.snapshots),
            "total_events": len(self.events),
            "snapshots": [asdict(snap) for snap in self.snapshots],
            "events": self.events,
        }

        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=2, default=str)

        return log_file
