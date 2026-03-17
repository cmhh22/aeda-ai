from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from sklearn.metrics import mean_squared_error

from data_component import DataComponent

try:
    from missingpy import MissForest
except ImportError:  # pragma: no cover
    MissForest = None  # type: ignore[assignment]


@dataclass
class ImputationValidationReport:
    strategy: str
    folds: int
    mask_fraction: float
    mse_per_fold: list[float]
    mse_mean: float
    mse_std: float


class DataReconstructor(DataComponent):
    def __init__(
        self,
        random_state: int = 42,
        missforest_n_estimators: int = 200,
        missforest_max_iter: int = 10,
    ) -> None:
        self.random_state = random_state
        self.missforest_n_estimators = missforest_n_estimators
        self.missforest_max_iter = missforest_max_iter

    @staticmethod
    def _resolve_columns(data: pd.DataFrame, columns: list[str] | None) -> list[str]:
        if columns is None:
            numeric = data.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric:
                raise ValueError("No hay columnas numéricas para imputación")
            return numeric

        missing_columns = [column for column in columns if column not in data.columns]
        if missing_columns:
            raise ValueError(f"Columnas inexistentes en DataFrame: {missing_columns}")
        return columns

    @staticmethod
    def _to_numeric_frame(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        return data[columns].apply(pd.to_numeric, errors="coerce")

    def reconstruct_time_series_pchip(
        self,
        data: pd.DataFrame,
        columns: list[str] | None = None,
        time_column: str | None = None,
    ) -> pd.DataFrame:
        if data.empty:
            raise ValueError("El DataFrame de entrada está vacío")

        resolved_columns = self._resolve_columns(data, columns)
        reconstructed = data.copy()

        if time_column is not None:
            if time_column not in reconstructed.columns:
                raise ValueError(f"time_column '{time_column}' no existe en DataFrame")
            time_values = pd.to_numeric(reconstructed[time_column], errors="coerce")
            x_all = time_values.to_numpy(dtype=float)
        else:
            x_all = np.arange(len(reconstructed), dtype=float)

        for column in resolved_columns:
            y = pd.to_numeric(reconstructed[column], errors="coerce").to_numpy(dtype=float)
            valid_mask = ~np.isnan(y)

            if valid_mask.sum() < 2:
                continue

            x_valid = x_all[valid_mask]
            y_valid = y[valid_mask]

            order = np.argsort(x_valid)
            x_sorted = x_valid[order]
            y_sorted = y_valid[order]

            unique_x, unique_idx = np.unique(x_sorted, return_index=True)
            y_unique = y_sorted[unique_idx]

            if len(unique_x) < 2:
                continue

            interpolator = PchipInterpolator(unique_x, y_unique, extrapolate=False)
            missing_mask = np.isnan(y)
            if missing_mask.any():
                y_imputed = y.copy()
                y_imputed[missing_mask] = interpolator(x_all[missing_mask])
                reconstructed[column] = y_imputed

        return reconstructed

    def reconstruct_tabular_missforest(
        self,
        data: pd.DataFrame,
        columns: list[str] | None = None,
        categorical_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        if data.empty:
            raise ValueError("El DataFrame de entrada está vacío")
        if MissForest is None:
            raise ImportError(
                "missingpy no está instalado. Instala con: pip install missingpy"
            )

        resolved_columns = self._resolve_columns(data, columns)
        reconstructed = data.copy()
        numeric_frame = self._to_numeric_frame(reconstructed, resolved_columns)

        cat_vars: list[int] | None = None
        if categorical_columns:
            missing_categories = [column for column in categorical_columns if column not in resolved_columns]
            if missing_categories:
                raise ValueError(
                    "categorical_columns debe ser subconjunto de columns. Faltan: "
                    f"{missing_categories}"
                )
            cat_vars = [resolved_columns.index(column) for column in categorical_columns]

        model = MissForest(
            max_iter=self.missforest_max_iter,
            n_estimators=self.missforest_n_estimators,
            random_state=self.random_state,
        )

        imputed_array = model.fit_transform(numeric_frame.to_numpy(), cat_vars=cat_vars)
        imputed_frame = pd.DataFrame(imputed_array, columns=resolved_columns, index=reconstructed.index)
        reconstructed[resolved_columns] = imputed_frame[resolved_columns]

        return reconstructed

    def estimate_imputation_mse(
        self,
        data: pd.DataFrame,
        strategy: str,
        columns: list[str] | None = None,
        folds: int = 5,
        mask_fraction: float = 0.1,
        time_column: str | None = None,
        categorical_columns: list[str] | None = None,
    ) -> ImputationValidationReport:
        if folds < 2:
            raise ValueError("folds debe ser >= 2")
        if mask_fraction <= 0 or mask_fraction >= 1:
            raise ValueError("mask_fraction debe estar en el intervalo (0, 1)")

        resolved_columns = self._resolve_columns(data, columns)
        numeric_frame = self._to_numeric_frame(data, resolved_columns)

        observed_positions = np.argwhere(~numeric_frame.isna().to_numpy())
        if len(observed_positions) == 0:
            raise ValueError("No hay valores observados para evaluar la imputación")

        rng = np.random.default_rng(self.random_state)
        mse_values: list[float] = []

        for _ in range(folds):
            eval_frame = data.copy()
            eval_numeric = self._to_numeric_frame(eval_frame, resolved_columns)

            sample_size = max(1, int(len(observed_positions) * mask_fraction))
            sampled_idx = rng.choice(len(observed_positions), size=sample_size, replace=False)
            sampled_positions = observed_positions[sampled_idx]

            true_values: list[float] = []
            masked_positions: list[tuple[int, int]] = []

            for row_idx, col_idx in sampled_positions:
                row_label = eval_numeric.index[row_idx]
                col_name = resolved_columns[col_idx]
                true_values.append(float(eval_numeric.iloc[row_idx, col_idx]))
                eval_frame.at[row_label, col_name] = np.nan
                masked_positions.append((row_idx, col_idx))

            if strategy == "pchip":
                imputed = self.reconstruct_time_series_pchip(
                    eval_frame,
                    columns=resolved_columns,
                    time_column=time_column,
                )
            elif strategy == "missforest":
                imputed = self.reconstruct_tabular_missforest(
                    eval_frame,
                    columns=resolved_columns,
                    categorical_columns=categorical_columns,
                )
            else:
                raise ValueError("strategy debe ser 'pchip' o 'missforest'")

            pred_values: list[float] = []
            for row_idx, col_idx in masked_positions:
                pred_value = pd.to_numeric(
                    pd.Series([imputed.iloc[row_idx][resolved_columns[col_idx]]]),
                    errors="coerce",
                ).iloc[0]
                pred_values.append(float(pred_value) if not np.isnan(pred_value) else np.nan)

            valid_pairs = [
                (true_value, pred_value)
                for true_value, pred_value in zip(true_values, pred_values)
                if not np.isnan(pred_value)
            ]
            if not valid_pairs:
                mse_values.append(np.nan)
                continue

            y_true = [pair[0] for pair in valid_pairs]
            y_pred = [pair[1] for pair in valid_pairs]
            mse_values.append(float(mean_squared_error(y_true, y_pred)))

        valid_mse = [value for value in mse_values if not np.isnan(value)]
        if not valid_mse:
            raise ValueError("No se pudo calcular MSE válido; revise datos o estrategia")

        return ImputationValidationReport(
            strategy=strategy,
            folds=folds,
            mask_fraction=mask_fraction,
            mse_per_fold=mse_values,
            mse_mean=float(np.mean(valid_mse)),
            mse_std=float(np.std(valid_mse)),
        )

    def run(
        self,
        data: pd.DataFrame,
        strategy: str,
        columns: list[str] | None = None,
        time_column: str | None = None,
        categorical_columns: list[str] | None = None,
        estimate_mse: bool = True,
        folds: int = 5,
        mask_fraction: float = 0.1,
    ) -> dict[str, Any]:
        if strategy == "pchip":
            reconstructed = self.reconstruct_time_series_pchip(
                data,
                columns=columns,
                time_column=time_column,
            )
        elif strategy == "missforest":
            reconstructed = self.reconstruct_tabular_missforest(
                data,
                columns=columns,
                categorical_columns=categorical_columns,
            )
        else:
            raise ValueError("strategy debe ser 'pchip' o 'missforest'")

        report: ImputationValidationReport | None = None
        if estimate_mse:
            report = self.estimate_imputation_mse(
                data=data,
                strategy=strategy,
                columns=columns,
                folds=folds,
                mask_fraction=mask_fraction,
                time_column=time_column,
                categorical_columns=categorical_columns,
            )

        return {
            "reconstructed_data": reconstructed,
            "validation_report": report,
            "validation_report_dict": report.__dict__ if report else None,
        }
