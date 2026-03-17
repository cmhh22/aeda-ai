from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .pipeline_logger import PipelineLogger, DataFrameSnapshot
from .pipeline_step import PipelineStep


class PipelineExecutionError(Exception):
    """Raised when pipeline execution fails."""

    pass


class AEDA_Pipeline:
    """
    Robust sequential pipeline orchestrator using Chain of Responsibility pattern.

    Features:
    - Register processing steps dynamically
    - Execute steps sequentially with state tracking
    - Automatic DataFrame snapshots before/after each step
    - Global error handling with detailed context logging
    - Rollback capability (state snapshot before each step)
    - Pandas-agnostic (works with any DataFrame transformations)
    """

    def __init__(
        self,
        name: str = "AEDA_Pipeline",
        log_dir: str | Path = "logs",
        capture_snapshots: bool = True,
    ) -> None:
        """
        Initialize AEDA_Pipeline.

        Args:
            name: Pipeline identifier
            log_dir: Directory for logs and snapshots
            capture_snapshots: Whether to capture DataFrame states between steps
        """
        self.name = name
        self.logger = PipelineLogger(log_dir=log_dir)
        self.capture_snapshots = capture_snapshots
        self.steps: list[PipelineStep] = []
        self.is_executed: bool = False
        self.execution_time: float = 0.0
        self.failed_at_step: str | None = None
        self.last_exception: Exception | None = None

    def register_step(self, step: PipelineStep) -> AEDA_Pipeline:
        """
        Register a processing step (Chain of Responsibility builder).

        Args:
            step: PipelineStep instance to register

        Returns:
            Self for method chaining

        Raises:
            ValueError: If step name is already registered
        """
        existing_names = [s.name for s in self.steps]
        if step.name in existing_names:
            raise ValueError(
                f"Step '{step.name}' already registered. "
                f"Registered steps: {existing_names}"
            )

        self.steps.append(step)
        self.logger.log_event(
            "step_registered",
            step.name,
            f"Registered step (total steps: {len(self.steps)})",
            {"description": step.description},
        )
        return self

    def register_steps(self, steps: list[PipelineStep]) -> AEDA_Pipeline:
        """
        Register multiple steps at once.

        Args:
            steps: List of PipelineStep instances

        Returns:
            Self for method chaining
        """
        for step in steps:
            self.register_step(step)
        return self

    def _get_step_index(self, step_name: str) -> int:
        """Get index of step by name."""
        for idx, step in enumerate(self.steps):
            if step.name == step_name:
                return idx
        raise ValueError(f"Step '{step_name}' not found in pipeline")

    def get_steps_summary(self) -> list[dict[str, str]]:
        """
        Get summary of all registered steps.

        Returns:
            List of step dictionaries with name and description
        """
        return [
            {"name": step.name, "description": step.description}
            for step in self.steps
        ]

    def execute(
        self,
        data: pd.DataFrame,
        stop_on_error: bool = True,
        partial_ok: bool = False,
    ) -> pd.DataFrame:
        """
        Execute pipeline sequentially (Chain of Responsibility execution).

        Args:
            data: Input DataFrame
            stop_on_error: If True, stop on first error; if False, skip failed steps
            partial_ok: If True, return partial result even if some steps fail

        Returns:
            Transformed DataFrame

        Raises:
            PipelineExecutionError: If stop_on_error=True and any step fails
        """
        if not self.steps:
            raise ValueError("No steps registered. Use register_step() first.")

        start_time = time.time()
        result = data.copy()
        previous_snapshot: DataFrameSnapshot | None = None

        self.logger.log_event(
            "pipeline_start",
            self.name,
            f"Starting pipeline execution with {len(self.steps)} steps",
            {"input_shape": result.shape},
        )

        try:
            for step_idx, step in enumerate(self.steps, 1):
                try:
                    # Capture state before step
                    if self.capture_snapshots:
                        previous_snapshot = self.logger.capture_snapshot(
                            result, step.name
                        )

                    self.logger.log_event(
                        "step_start",
                        step.name,
                        f"Executing step {step_idx}/{len(self.steps)}",
                        {"input_shape": result.shape},
                    )

                    # Execute step with timing
                    step_start = time.time()
                    result = step.execute(result)
                    step.execution_time = time.time() - step_start

                    # Validate output
                    if not isinstance(result, pd.DataFrame):
                        raise TypeError(
                            f"Step '{step.name}' returned {type(result)} "
                            f"instead of pd.DataFrame"
                        )

                    # Track changes
                    step.rows_affected = (
                        data.shape[0] - result.shape[0]
                        if data.shape[0] > result.shape[0]
                        else 0
                    )
                    new_cols = set(result.columns) - set(data.columns)
                    removed_cols = set(data.columns) - set(result.columns)
                    step.columns_affected = list(new_cols.union(removed_cols))

                    step.success = True

                    self.logger.log_event(
                        "step_complete",
                        step.name,
                        f"Step completed successfully",
                        {
                            "execution_time": step.execution_time,
                            "output_shape": result.shape,
                            "rows_affected": step.rows_affected,
                            "columns_changed": step.columns_affected,
                        },
                    )

                except Exception as step_error:
                    step.success = False
                    step.error_message = str(step_error)
                    self.failed_at_step = step.name
                    self.last_exception = step_error

                    self.logger.log_event(
                        "step_error",
                        step.name,
                        f"Step failed with {type(step_error).__name__}",
                        {"error": str(step_error)},
                    )

                    failure_report = self.logger.generate_failure_report(
                        step.name, step_error, previous_snapshot
                    )

                    if stop_on_error:
                        log_file = self.logger.save_logs(self.name)
                        raise PipelineExecutionError(
                            f"Pipeline stopped at step '{step.name}': {step_error}\n\n"
                            f"Failure Report:\n{failure_report}\n\n"
                            f"Full logs saved to: {log_file}"
                        ) from step_error

                    if not partial_ok:
                        return result

        finally:
            self.execution_time = time.time() - start_time
            self.is_executed = True

        self.logger.log_event(
            "pipeline_complete",
            self.name,
            "Pipeline execution completed successfully",
            {
                "total_execution_time": self.execution_time,
                "output_shape": result.shape,
                "steps_executed": len(self.steps),
            },
        )

        return result

    def execute_from(
        self,
        data: pd.DataFrame,
        step_name: str,
    ) -> pd.DataFrame:
        """
        Execute pipeline starting from a specific step (skip earlier steps).

        Args:
            data: Input DataFrame
            step_name: Name of step to start from

        Returns:
            Transformed DataFrame

        Raises:
            ValueError: If step_name not found
            PipelineExecutionError: If execution fails
        """
        start_idx = self._get_step_index(step_name)
        remaining_steps = self.steps[start_idx:]

        self.logger.log_event(
            "pipeline_partial_start",
            self.name,
            f"Starting pipeline from step '{step_name}' (skipping {start_idx} steps)",
            {"remaining_steps": len(remaining_steps)},
        )

        # Temporarily replace steps
        original_steps = self.steps
        self.steps = remaining_steps

        try:
            return self.execute(data, stop_on_error=True)
        finally:
            self.steps = original_steps

    def get_execution_report(self) -> dict[str, Any]:
        """
        Get comprehensive execution report.

        Returns:
            Dictionary with execution summary
        """
        return {
            "pipeline_name": self.name,
            "is_executed": self.is_executed,
            "total_execution_time_seconds": self.execution_time,
            "total_steps": len(self.steps),
            "failed_at_step": self.failed_at_step,
            "last_error": str(self.last_exception) if self.last_exception else None,
            "total_snapshots": len(self.logger.snapshots),
            "total_events": len(self.logger.events),
            "steps": [
                {
                    "name": step.name,
                    "description": step.description,
                    "success": step.success,
                    "execution_time": step.execution_time,
                    "rows_affected": step.rows_affected,
                    "columns_affected": step.columns_affected,
                    "error": step.error_message,
                }
                for step in self.steps
            ],
        }

    def save_execution_logs(self) -> Path:
        """
        Save pipeline logs and snapshots to disk.

        Returns:
            Path to saved log file
        """
        return self.logger.save_logs(self.name)

    def __repr__(self) -> str:
        return (
            f"AEDA_Pipeline(name='{self.name}', steps={len(self.steps)}, "
            f"executed={self.is_executed})"
        )
