"""
aeda.io.preprocessor
Preprocesamiento de datos ambientales:
- Limpieza y filtrado
- Imputación inteligente (respeta diseño experimental)
- Escalado (StandardScaler, MinMax, Robust)
- Transformaciones composicionales (CLR para datos FRX)
- Encoding de variables categóricas
"""

import pandas as pd
import numpy as np
from typing import Optional, Literal
from dataclasses import dataclass, field
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer


@dataclass
class PreprocessingLog:
    """Registro de todas las transformaciones aplicadas (reproducibilidad)."""
    steps: list[dict] = field(default_factory=list)

    def add(self, step: str, details: dict):
        self.steps.append({"step": step, **details})

    def summary(self) -> str:
        lines = [f"Pipeline de preprocesamiento ({len(self.steps)} pasos):"]
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
    Selecciona solo columnas numéricas de medición,
    excluyendo coordenadas, IDs, y columnas de incertidumbre.
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
    """Elimina columnas con varianza cero."""
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
    Maneja datos faltantes con estrategia configurable.

    Parameters
    ----------
    df : DataFrame con datos numéricos
    strategy : Estrategia de imputación
    threshold_col : Elimina columnas con más de este % de nulos (0-1)
    threshold_row : Elimina filas con más de este % de nulos (0-1)
    """
    original_shape = df.shape

    # Primero: eliminar columnas con demasiados nulos
    null_pct_col = df.isnull().mean()
    high_null_cols = null_pct_col[null_pct_col > threshold_col].index.tolist()
    if high_null_cols:
        df = df.drop(columns=high_null_cols)
        if log:
            log.add("drop_high_null_cols", {
                "columns": high_null_cols,
                "threshold": threshold_col,
            })

    # Segundo: eliminar filas con demasiados nulos
    null_pct_row = df.isnull().mean(axis=1)
    high_null_rows = null_pct_row[null_pct_row > threshold_row].index.tolist()
    if high_null_rows:
        df = df.drop(index=high_null_rows)
        if log:
            log.add("drop_high_null_rows", {
                "n_rows": len(high_null_rows),
                "threshold": threshold_row,
            })

    # Tercero: imputar lo restante
    if df.isnull().any().any():
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

        if log:
            log.add("impute", {"strategy": strategy, "shape_before": original_shape, "shape_after": df.shape})

    return df


def scale(
    df: pd.DataFrame,
    method: Literal["standard", "minmax", "robust"] = "standard",
    log: Optional[PreprocessingLog] = None,
) -> tuple[pd.DataFrame, object]:
    """
    Escala los datos numéricos.

    Returns
    -------
    tuple[pd.DataFrame, scaler]
        DataFrame escalado y el objeto scaler para inversa.
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
    Transformación logarítmica para variables con distribución sesgada.
    Usa log(x + offset) para manejar ceros.
    """
    if cols is None:
        # Auto-detectar columnas con alto skewness
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
    Centered Log-Ratio transform para datos composicionales.
    Esencial para datos de FRX donde las concentraciones suman ~100%.

    Parameters
    ----------
    df : DataFrame con datos composicionales (solo columnas numéricas positivas)
    cols : Columnas a transformar. Si None, usa todas.
    """
    if cols is None:
        cols = df.columns.tolist()

    comp = df[cols].copy()

    # Reemplazar ceros con un valor pequeño (multiplicative replacement)
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
    Pipeline de preprocesamiento completo.

    Parameters
    ----------
    df : DataFrame crudo (solo columnas numéricas de medición)
    exclude_cols : Columnas a excluir del análisis
    impute_strategy : Estrategia de imputación
    scale_method : Método de escalado
    apply_log : Aplicar log-transform a variables sesgadas
    apply_clr : Aplicar CLR transform (para datos composicionales FRX)
    clr_cols : Columnas para CLR. Si None y apply_clr=True, usa todas.

    Returns
    -------
    tuple[pd.DataFrame, PreprocessingLog, scaler]
    """
    proc_log = PreprocessingLog()

    # 1. Seleccionar numéricas
    data = select_numeric(df, exclude_cols=exclude_cols)
    proc_log.add("select_numeric", {"n_features": len(data.columns), "columns": data.columns.tolist()})

    # 2. Eliminar constantes
    data = drop_constant_columns(data, log=proc_log)

    # 3. Manejar faltantes
    data = handle_missing(data, strategy=impute_strategy, log=proc_log)

    # 4. Transformaciones opcionales
    if apply_clr:
        data = clr_transform(data, cols=clr_cols, log=proc_log)
    elif apply_log:
        data = log_transform(data, log=proc_log)

    # 5. Escalar
    data, scaler = scale(data, method=scale_method, log=proc_log)

    return data, proc_log, scaler
