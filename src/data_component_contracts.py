from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, TypedDict

import pandas as pd


class ComponentOutput(TypedDict, total=False):
    data: pd.DataFrame
    report: dict[str, Any]
    metadata: dict[str, Any]
    success: bool


def serialize_report(report: Any | None) -> dict[str, Any] | None:
    if report is None:
        return None
    if isinstance(report, dict):
        return report
    if is_dataclass(report):
        return asdict(report)
    return {"value": str(report)}


def build_component_output(
    *,
    data: pd.DataFrame,
    report: Any | None = None,
    metadata: dict[str, Any] | None = None,
    success: bool = True,
    extra: dict[str, Any] | None = None,
) -> ComponentOutput:
    output: ComponentOutput = {
        "data": data,
        "success": success,
    }

    serialized_report = serialize_report(report)
    if serialized_report is not None:
        output["report"] = serialized_report

    if metadata is not None:
        output["metadata"] = metadata

    if extra:
        output.update(extra)

    return output
