"""
aeda.io.preprocessor
Environmental data preprocessing:
- Cleaning and filtering
- Smart imputation (respects experimental design)
- Scaling (StandardScaler, MinMax, Robust)
- Compositional transforms (CLR for FRX data)
- Categorical variable encoding
"""

import pandas as pd
import numpy as np
from typing import Optional, Literal
from dataclasses import dataclass, field
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer


@dataclass
class PreprocessingLog:
    """Record of all applied transformations (reproducibility)."""
    steps: list[dict] = field(default_factory=list)

    def add(self, step: str, details: dict):
        self.steps.append({"step": step, **details})

    def summary(self) -> str:
        lines = [f"Preprocessing pipeline ({len(self.steps)} steps):"]
        for i, s in enumerate(self.steps, 1):
            step_name = s.pop("step", "?")
            lines.append(f"  {i}. {step_name}: {s}")
            s["step"] = step_name
        return "\n".join(lines)


def select_numeric(
    df: pd.DataFrame,
    exclude_cols: Optional[list[str]] = None,
    exclude_prefixes: tuple[str, ...] = ("U_", "u_"),
) -> pd.DataFrame:
    """
    Select numeric measurement columns only,
    excluding coordinates, IDs, and uncertainty columns.
    """
    numeric = df.select_dtypes(include="number")
    drop = []
    if exclude_cols:
        drop.extend([c for c in exclude_cols if c in numeric.columns])
    for col in numeric.columns:
        if any(col.startswith(p) for p in exclude_prefixes):
            drop.append(col)
    return numeric.drop(columns=drop, errors="ignore")


def drop_constant_columns(df: pd.DataFrame, log: Optional[PreprocessingLog] = None) -> pd.DataFrame:
    """Remove columns with zero variance."""
    constant = [c for c in df.columns if df[c].nunique() <= 1]
    if constant and log:
        log.add("drop_constant", {"columns": constant})
    return df.drop(columns=constant)


def handle_missing(
    df: pd.DataFrame,
    strategy: Literal["drop_rows", "drop_cols", "mean", "median", "knn"] = "median",
    threshold_col: float = 0.5,
    threshold_row: float = 0.5,
    log: Optional[PreprocessingLog] = None,
) -> pd.DataFrame:
    """
    Handle missing values with a configurable strategy.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with numeric data.
    strategy : Literal["drop_rows", "drop_cols", "mean", "median", "knn"]
        Imputation strategy.
    threshold_col : float
        Drop columns with a missing fraction above this threshold (0-1).
    threshold_row : float
        Drop rows with a missing fraction above this threshold (0-1).
    """
    original_shape = df.shape

    # First: drop columns with too many missing values
    null_pct_col = df.isnull().mean()
    high_null_cols = null_pct_col[null_pct_col > threshold_col].index.tolist()
    if high_null_cols:
        df = df.drop(columns=high_null_cols)
        if log:
            log.add("drop_high_null_cols", {
                "columns": high_null_cols,
                "threshold": threshold_col,
            })

    # Second: drop rows with too many missing values
    null_pct_row = df.isnull().mean(axis=1)
    high_null_rows = null_pct_row[null_pct_row > threshold_row].index.tolist()
    if high_null_rows:
        df = df.drop(index=high_null_rows)
        if log:
            log.add("drop_high_null_rows", {
                "n_rows": len(high_null_rows),
                "threshold": threshold_row,
            })

    # Third: impute remaining missing values
    if df.isnull().any().any():
        valid_strategies = ("drop_rows", "drop_cols", "mean", "median", "knn")
        if strategy == "drop_rows":
            df = df.dropna()
        elif strategy == "drop_cols":
            df = df.dropna(axis=1)
        elif strategy in ("mean", "median"):
            imputer = SimpleImputer(strategy=strategy)
            df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)
        elif strategy == "knn":
            imputer = KNNImputer(n_neighbors=5)
            df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)
        else:
            raise ValueError(
                f"Unsupported impute_strategy: '{strategy}'. "
                f"Valid options: {valid_strategies}"
            )

        if log:
            log.add("impute", {"strategy": strategy, "shape_before": original_shape, "shape_after": df.shape})

    return df


def scale(
    df: pd.DataFrame,
    method: Literal["standard", "minmax", "robust"] = "standard",
    log: Optional[PreprocessingLog] = None,
) -> tuple[pd.DataFrame, object]:
    """
    Scale numeric data.

    Returns
    -------
    tuple[pd.DataFrame, object]
        Scaled DataFrame and fitted scaler object for inverse transforms.
    """
    scalers = {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "robust": RobustScaler,
    }
    scaler = scalers[method]()
    scaled = pd.DataFrame(
        scaler.fit_transform(df),
        columns=df.columns,
        index=df.index,
    )
    if log:
        log.add("scale", {"method": method, "n_features": len(df.columns)})
    return scaled, scaler


def log_transform(
    df: pd.DataFrame,
    cols: Optional[list[str]] = None,
    offset: float = 1.0,
    log: Optional[PreprocessingLog] = None,
) -> pd.DataFrame:
    """
    Log transform for skewed variables.
    Uses log(x + offset) to handle zeros.
    """
    if cols is None:
        # Auto-detect highly skewed columns
        skew = df.skew()
        cols = skew[skew.abs() > 2.0].index.tolist()

    if cols:
        df = df.copy()
        for col in cols:
            if col in df.columns:
                df[col] = np.log(df[col] + offset)
        if log:
            log.add("log_transform", {"columns": cols, "offset": offset})

    return df


def clr_transform(
    df: pd.DataFrame,
    cols: Optional[list[str]] = None,
    log: Optional[PreprocessingLog] = None,
) -> pd.DataFrame:
    """
    Centered Log-Ratio transform for compositional data.
    Essential for FRX datasets where concentrations sum to ~100%.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with compositional data (positive numeric columns only).
    cols : list[str], optional
        Columns to transform. If None, all columns are used.
    """
    if cols is None:
        cols = df.columns.tolist()

    comp = df[cols].copy()

    # Replace zeros with a small value (multiplicative replacement)
    min_nonzero = comp[comp > 0].min().min()
    comp = comp.replace(0, min_nonzero * 0.65)

    # CLR: log(x_i / geometric_mean)
    log_comp = np.log(comp)
    geo_mean = log_comp.mean(axis=1)
    clr = log_comp.subtract(geo_mean, axis=0)

    result = df.copy()
    result[cols] = clr

    if log:
        log.add("clr_transform", {"columns": cols, "n_features": len(cols)})

    return result


def preprocess(
    df: pd.DataFrame,
    exclude_cols: Optional[list[str]] = None,
    impute_strategy: str = "median",
    scale_method: str = "standard",
    apply_log: bool = False,
    apply_clr: bool = False,
    clr_cols: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, PreprocessingLog, Optional[object]]:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame (numeric measurement columns only).
    exclude_cols : list[str], optional
        Columns to exclude from analysis.
    impute_strategy : str
        Imputation strategy.
    scale_method : str
        Scaling method.
    apply_log : bool
        Whether to apply log-transform to skewed variables.
    apply_clr : bool
        Whether to apply CLR transform (for FRX compositional data).
    clr_cols : list[str], optional
        Columns for CLR. If None and apply_clr=True, all are used.

    Returns
    -------
    tuple[pd.DataFrame, PreprocessingLog, Optional[object]]
        Processed data, preprocessing log, and fitted scaler.
    """
    proc_log = PreprocessingLog()

    # 1. Select numeric data
    data = select_numeric(df, exclude_cols=exclude_cols)
    proc_log.add("select_numeric", {"n_features": len(data.columns), "columns": data.columns.tolist()})

    # 2. Drop constant columns
    data = drop_constant_columns(data, log=proc_log)

    # 3. Handle missing values
    data = handle_missing(data, strategy=impute_strategy, log=proc_log)

    # 4. Optional transforms
    if apply_clr:
        data = clr_transform(data, cols=clr_cols, log=proc_log)
    elif apply_log:
        data = log_transform(data, log=proc_log)

    # 5. Scale
    data, scaler = scale(data, method=scale_method, log=proc_log)

    return data, proc_log, scaler
