from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class PipelineStep(ABC):
    """
    Abstract base class for pipeline steps.
    Implements the interface for Chain of Responsibility pattern.
    """

    def __init__(self, name: str, description: str = "") -> None:
        """
        Initialize pipeline step.

        Args:
            name: Unique identifier for this step
            description: Human-readable description of what this step does
        """
        self.name = name
        self.description = description
        self.execution_time: float = 0.0  # seconds
        self.rows_affected: int = 0
        self.columns_affected: list[str] = []
        self.success: bool = False
        self.error_message: str | None = None

    @abstractmethod
    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Execute the transformation on the input DataFrame.

        Args:
            data: Input DataFrame to transform

        Returns:
            Transformed DataFrame

        Raises:
            Exception: Any error during execution will be caught by
                      AEDA_Pipeline with full context logging
        """
        pass

    def get_metadata(self) -> dict[str, Any]:
        """
        Return execution metadata for logging.

        Returns:
            Dictionary with execution details
        """
        return {
            "step_name": self.name,
            "description": self.description,
            "execution_time_seconds": self.execution_time,
            "rows_affected": self.rows_affected,
            "columns_affected": self.columns_affected,
            "success": self.success,
            "error_message": self.error_message,
        }
