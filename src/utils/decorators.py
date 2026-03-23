from __future__ import annotations

import inspect
import os
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from utils.logs import append_json_log, utc_now_iso
from utils.metadata import dataframe_quick_metadata


def _safe_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_safe_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _safe_json_value(val) for key, val in value.items()}
    return str(value)


def _extract_dataframe(args: tuple[Any, ...], kwargs: dict[str, Any]) -> pd.DataFrame | None:
    for arg in args:
        if isinstance(arg, pd.DataFrame):
            return arg
    for value in kwargs.values():
        if isinstance(value, pd.DataFrame):
            return value
    return None


def _extract_output_dataframe(result: Any) -> pd.DataFrame | None:
    if isinstance(result, pd.DataFrame):
        return result
    if isinstance(result, dict):
        preferred_keys = ["data", "standardized_data", "reconstructed_data", "cleaned_data", "data_cleaned"]
        for key in preferred_keys:
            value = result.get(key)
            if isinstance(value, pd.DataFrame):
                return value
        for value in result.values():
            if isinstance(value, pd.DataFrame):
                return value
    return None


def _extract_call_parameters(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    signature = inspect.signature(func)
    bound = signature.bind_partial(*args, **kwargs)
    bound.apply_defaults()

    parameters: dict[str, Any] = {}
    for name, value in bound.arguments.items():
        if name in {"self", "data"}:
            continue
        parameters[name] = _safe_json_value(value)

    instance = bound.arguments.get("self")
    if instance is not None and hasattr(instance, "__dict__"):
        instance_params: dict[str, Any] = {}
        for key, value in vars(instance).items():
            if key.startswith("_"):
                continue
            if isinstance(value, (str, int, float, bool, Path, list, tuple, dict, type(None))):
                instance_params[key] = _safe_json_value(value)
        if instance_params:
            parameters["instance_config"] = instance_params

    return parameters


def track_transformation(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        before_df = _extract_dataframe(args, kwargs)
        before_stats = dataframe_quick_metadata(before_df)
        parameters = _extract_call_parameters(func, args, kwargs)

        started_at = utc_now_iso()
        start_time = time.perf_counter()

        try:
            result = func(*args, **kwargs)
            status = "success"
            error_message = None
            after_df = _extract_output_dataframe(result)
            return result
        except Exception as error:
            status = "error"
            error_message = str(error)
            after_df = before_df
            raise
        finally:
            event = {
                "timestamp_utc": utc_now_iso(),
                "started_at_utc": started_at,
                "function_name": func.__qualname__,
                "parameters": parameters,
                "execution_time_seconds": round(float(time.perf_counter() - start_time), 6),
                "status": status,
                "error_message": error_message,
                "dataset_before": before_stats,
                "dataset_after": dataframe_quick_metadata(after_df),
            }
            append_json_log(os.getenv("AEDA_AUDIT_LOG_PATH", "audit_log.json"), event)

    return wrapper
