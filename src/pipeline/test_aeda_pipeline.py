"""
Unit tests for AEDA_Pipeline and PipelineStep components.
Demonstrates Chain of Responsibility pattern and error handling.
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from pipeline.aeda_pipeline import AEDA_Pipeline, PipelineExecutionError
from pipeline.pipeline_step import PipelineStep


class AddOneStep(PipelineStep):
    """Simple test step: add 1 to all numeric columns."""

    def __init__(self):
        super().__init__(
            name="add_one",
            description="Add 1 to all numeric columns",
        )

    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.apply(lambda x: x + 1 if x.dtype in ["int64", "float64"] else x)


class MultiplyByTwoStep(PipelineStep):
    """Simple test step: multiply all numeric columns by 2."""

    def __init__(self):
        super().__init__(
            name="multiply_by_two",
            description="Multiply all numeric columns by 2",
        )

    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.apply(lambda x: x * 2 if x.dtype in ["int64", "float64"] else x)


class FailingStep(PipelineStep):
    """Test step that always fails."""

    def __init__(self, message: str = "Intentional failure"):
        super().__init__(
            name="failing_step",
            description="Always fails for testing",
        )
        self.message = message

    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        raise ValueError(self.message)


class RenameColumnsStep(PipelineStep):
    """Test step: rename columns by adding suffix."""

    def __init__(self, suffix: str = "_processed"):
        super().__init__(
            name="rename_columns",
            description="Rename columns",
        )
        self.suffix = suffix

    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.rename(columns={col: f"{col}{self.suffix}" for col in data.columns})


def test_single_step_execution():
    """Test executing pipeline with single step."""
    pipeline = AEDA_Pipeline(name="test_single_step")
    pipeline.register_step(AddOneStep())

    data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = pipeline.execute(data)

    assert result["a"].tolist() == [2, 3, 4]
    assert result["b"].tolist() == [5, 6, 7]
    assert pipeline.is_executed


def test_multiple_steps_chain():
    """Test executing pipeline with multiple steps (Chain of Responsibility)."""
    pipeline = AEDA_Pipeline(name="test_chain")
    pipeline.register_steps([
        AddOneStep(),         # x + 1
        MultiplyByTwoStep(),  # (x + 1) * 2
    ])

    data = pd.DataFrame({"a": [1, 2, 3]})
    # Expected: (1 + 1) * 2 = 4, (2 + 1) * 2 = 6, (3 + 1) * 2 = 8
    result = pipeline.execute(data)

    assert result["a"].tolist() == [4, 6, 8]
    assert len(pipeline.get_execution_report()["steps"]) == 2


def test_method_chaining():
    """Test fluent interface with method chaining."""
    result = (
        AEDA_Pipeline(name="test_chaining")
        .register_step(AddOneStep())
        .register_step(MultiplyByTwoStep())
    )

    assert isinstance(result, AEDA_Pipeline)
    assert len(result.steps) == 2


def test_error_stops_pipeline():
    """Test that error stops pipeline execution."""
    pipeline = AEDA_Pipeline(name="test_error")
    pipeline.register_steps([
        AddOneStep(),
        FailingStep("Test error"),
        MultiplyByTwoStep(),  # Should not execute
    ])

    data = pd.DataFrame({"a": [1, 2, 3]})

    with pytest.raises(PipelineExecutionError) as exc_info:
        pipeline.execute(data, stop_on_error=True)

    assert pipeline.failed_at_step == "failing_step"
    assert "Test error" in str(exc_info.value)


def test_duplicate_step_name_rejected():
    """Test that duplicate step names are rejected."""
    pipeline = AEDA_Pipeline()
    pipeline.register_step(AddOneStep())

    with pytest.raises(ValueError, match="already registered"):
        pipeline.register_step(AddOneStep())


def test_execution_report():
    """Test execution report generation."""
    pipeline = AEDA_Pipeline(name="test_report")
    pipeline.register_steps([
        AddOneStep(),
        MultiplyByTwoStep(),
    ])

    data = pd.DataFrame({"a": [1, 2, 3]})
    pipeline.execute(data)

    report = pipeline.get_execution_report()
    assert report["pipeline_name"] == "test_report"
    assert report["is_executed"] is True
    assert report["total_steps"] == 2
    assert report["failed_at_step"] is None
    assert all(step["success"] for step in report["steps"])


def test_snapshots_capture():
    """Test that DataFrame snapshots are captured."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pipeline = AEDA_Pipeline(
            name="test_snapshots",
            log_dir=tmpdir,
            capture_snapshots=True,
        )
        pipeline.register_steps([
            AddOneStep(),
            RenameColumnsStep(suffix="_v1"),
        ])

        data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        pipeline.execute(data)

        # Snapshots capture before each step
        assert len(pipeline.logger.snapshots) >= 2


def test_log_persistence():
    """Test that logs are saved to disk."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pipeline = AEDA_Pipeline(
            name="test_logs",
            log_dir=tmpdir,
        )
        pipeline.register_step(AddOneStep())

        data = pd.DataFrame({"a": [1, 2, 3]})
        pipeline.execute(data)

        log_file = pipeline.save_execution_logs()
        assert log_file.exists()
        assert log_file.suffix == ".json"


def test_steps_summary():
    """Test steps summary retrieval."""
    pipeline = AEDA_Pipeline()
    pipeline.register_steps([
        AddOneStep(),
        MultiplyByTwoStep(),
    ])

    summary = pipeline.get_steps_summary()
    assert len(summary) == 2
    assert summary[0]["name"] == "add_one"
    assert summary[1]["name"] == "multiply_by_two"


def test_no_steps_error():
    """Test error when no steps registered."""
    pipeline = AEDA_Pipeline()

    data = pd.DataFrame({"a": [1, 2, 3]})

    with pytest.raises(ValueError, match="No steps registered"):
        pipeline.execute(data)


def test_invalid_step_return_type():
    """Test error when step returns non-DataFrame."""

    class BadStep(PipelineStep):
        def __init__(self):
            super().__init__(name="bad_step")

        def execute(self, data: pd.DataFrame):
            return data.to_dict()  # Returns dict instead of DataFrame

    pipeline = AEDA_Pipeline()
    pipeline.register_step(BadStep())

    data = pd.DataFrame({"a": [1, 2, 3]})

    with pytest.raises(PipelineExecutionError, match="instead of pd.DataFrame"):
        pipeline.execute(data)


def test_row_column_tracking():
    """Test that step tracks rows and columns affected."""

    class DropRowsStep(PipelineStep):
        def __init__(self):
            super().__init__(name="drop_rows")

        def execute(self, data: pd.DataFrame) -> pd.DataFrame:
            return data.iloc[:2]  # Keep only first 2 rows

    pipeline = AEDA_Pipeline()
    pipeline.register_step(DropRowsStep())

    data = pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})
    pipeline.execute(data)

    step_report = pipeline.get_execution_report()["steps"][0]
    assert step_report["rows_affected"] == 2  # 4 - 2 = 2 rows removed


def test_execution_timing():
    """Test that execution time is tracked."""
    pipeline = AEDA_Pipeline()
    pipeline.register_step(AddOneStep())

    data = pd.DataFrame({"a": [1, 2, 3]})
    pipeline.execute(data)

    report = pipeline.get_execution_report()
    assert report["total_execution_time_seconds"] > 0
    assert report["steps"][0]["execution_time"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
