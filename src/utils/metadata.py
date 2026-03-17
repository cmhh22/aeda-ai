from __future__ import annotations

from typing import Any

import pandas as pd


def dataframe_quick_metadata(data: pd.DataFrame | None) -> dict[str, Any] | None:
    if data is None:
        return None

    numeric = data.select_dtypes(include=["number"])
    summary: dict[str, dict[str, float | int | None]] = {}

    for column in numeric.columns:
        series = pd.to_numeric(numeric[column], errors="coerce")
        summary[str(column)] = {
            "mean": float(series.mean()) if series.notna().any() else None,
            "std": float(series.std()) if series.notna().sum() > 1 else None,
            "count": int(series.count()),
        }

    return {
        "shape": [int(data.shape[0]), int(data.shape[1])],
        "numeric_summary": summary,
    }
