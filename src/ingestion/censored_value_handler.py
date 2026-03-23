"""
Censored Value Handler for Environmental Data Analysis

Handles Bottom-of-Detection Limit (BDL/LOD) and Above-Quantification-Limit (AQL) values
with multiple imputation strategies compatible with machine learning and statistical analysis.

Implements:
- LOD/2 (simple, conservative)
- ROS (Regression on Order Statistics) - for larger datasets
- QMLE (Quantile Maximum Likelihood Estimation)
- Custom percentile imputations
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class CensoredValueMetadata:
    """Tracks censoring information for traceability"""
    column: str
    lod_value: float
    aloq_value: float | None  # Above Limit of Quantification
    n_bdl: int  # Count of Below Detection Limit values
    n_aloq: int  # Count of Above Limit of Quantification
    imputation_strategy: str
    imputation_params: dict


class CensoredValueError(Exception):
    pass


class CensoredValueHandler:
    """
    Intelligent handler for censored values in environmental datasets.
    
    For FRX data with notation like:
    - "< 17" (BDL - Below Detection Limit)
    - "> 12000" (AQL - Above Quantification Limit)
    - "< 17 (8.5)" (BDL with estimated value)
    """
    
    def __init__(
        self,
        imputation_strategy: Literal["lod_half", "ros", "qmle", "percentile"] = "lod_half",
        percentile_value: float = 0.5,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize the censored value handler.
        
        Args:
            imputation_strategy: Method for imputing BDL values
                - "lod_half": Use LOD/2 (simple, transparent, standard in environmental science)
                - "ros": Regression on Order Statistics (requires >10 uncensored values)
                - "qmle": Quantile Maximum Likelihood Estimation
                - "percentile": Custom percentile between LOD and min observed
            percentile_value: For "percentile" strategy, value between 0 and 1
            logger: Logger instance for tracking
        """
        self.imputation_strategy = imputation_strategy
        self.percentile_value = percentile_value
        self.logger = logger or logging.getLogger(__name__)
        self.metadata: list[CensoredValueMetadata] = []
        
    def handle_column(
        self,
        series: pd.Series,
        column_name: str,
        lod: float,
        aloq: float | None = None,
    ) -> pd.Series:
        """
        Process a column with censored values.
        
        Args:
            series: Original series with possible censored values
            column_name: Name of the column (for reporting)
            lod: Limit of Detection
            aloq: Limit of Quantification (optional)
            
        Returns:
            Imputed series with float values
        """
        result = series.copy()
        n_bdl = (result < lod).sum()
        n_aloq = (result > aloq).sum() if aloq else 0
        
        if n_bdl == 0 and n_aloq == 0:
            return result
        
        # Apply imputation for BDL
        if n_bdl > 0:
            if self.imputation_strategy == "lod_half":
                result.loc[result < lod] = lod / 2
            elif self.imputation_strategy == "ros":
                result = self._impute_ros(result, lod)
            elif self.imputation_strategy == "qmle":
                result = self._impute_qmle(result, lod)
            elif self.imputation_strategy == "percentile":
                result = self._impute_percentile(result, lod)
        
        # Handle AQL (typically use the limit or max value)
        if aloq and n_aloq > 0:
            result.loc[result > aloq] = aloq
        
        metadata = CensoredValueMetadata(
            column=column_name,
            lod_value=lod,
            aloq_value=aloq,
            n_bdl=n_bdl,
            n_aloq=n_aloq,
            imputation_strategy=self.imputation_strategy,
            imputation_params={
                "percentile_value": self.percentile_value if self.imputation_strategy == "percentile" else None
            }
        )
        self.metadata.append(metadata)
        
        self.logger.debug(
            f"{column_name}: {n_bdl} BDL @ LOD={lod}, {n_aloq} ALOQ @ {aloq}. "
            f"Strategy: {self.imputation_strategy}"
        )
        
        return result
    
    def _impute_ros(self, series: pd.Series, lod: float) -> pd.Series:
        """
        Regression on Order Statistics for BDL imputation.
        Robust for datasets with <50% censoring.
        
        Based on Helsel's ROS method (widely used in environmental science).
        """
        bdl_mask = series < lod
        n_bdl = bdl_mask.sum()
        n_total = len(series)
        
        if n_bdl / n_total > 0.5:
            self.logger.warning(
                f"ROS: >50% censoring detected ({n_bdl}/{n_total}). "
                f"Falling back to LOD/2. Consider larger sample size."
            )
            series.loc[bdl_mask] = lod / 2
            return series
        
        if n_bdl < 3:
            series.loc[bdl_mask] = lod / 2
            return series
        
        # Get uncensored values
        uncensored = series[~bdl_mask].sort_values().values
        
        if len(uncensored) < 3:
            series.loc[bdl_mask] = lod / 2
            return series
        
        # Fit regression on log-scale
        x = np.arange(1, len(uncensored) + 1) / (len(uncensored) + 1)
        y = np.log10(uncensored + 1e-10)
        
        coeffs = np.polyfit(x, y, 1)
        
        # Predict for censored values
        x_censored = np.arange(1, n_bdl + 1) / (len(uncensored) + 1)
        y_censored = coeffs[0] * x_censored + coeffs[1]
        imputed_values = np.power(10, y_censored)
        
        series.loc[bdl_mask] = np.clip(imputed_values, lod / 2, lod)
        
        return series
    
    def _impute_qmle(self, series: pd.Series, lod: float) -> pd.Series:
        """
        Quantile Maximum Likelihood Estimation.
        More sophisticated but requires >30 samples.
        """
        if len(series) < 30:
            self.logger.warning(
                f"QMLE requires >=30 samples, got {len(series)}. Falling back to LOD/2."
            )
            series.loc[series < lod] = lod / 2
            return series
        
        # Simple QMLE: use geometric mean of uncensored values
        uncensored = series[series >= lod].values
        if len(uncensored) > 0:
            geom_mean = np.exp(np.mean(np.log(uncensored + 1e-10)))
            bdl_mask = series < lod
            series.loc[bdl_mask] = np.maximum(lod / 2, geom_mean / 10)
        
        return series
    
    def _impute_percentile(self, series: pd.Series, lod: float) -> pd.Series:
        """
        Impute BDL values at specified percentile between LOD/2 and minimum observed.
        """
        bdl_mask = series < lod
        uncensored = series[~bdl_mask]
        
        if len(uncensored) == 0:
            series.loc[bdl_mask] = lod / 2
            return series
        
        min_observed = uncensored.min()
        impute_value = lod / 2 + self.percentile_value * (min_observed - lod / 2)
        series.loc[bdl_mask] = impute_value
        
        return series
    
    def get_quality_report(self) -> dict:
        """Generate a summarized report of censoring handled"""
        report = {
            "total_columns_processed": len(self.metadata),
            "total_bdl_values": sum(m.n_bdl for m in self.metadata),
            "total_aloq_values": sum(m.n_aloq for m in self.metadata),
            "strategy_used": self.imputation_strategy,
            "columns": [
                {
                    "name": m.column,
                    "lod": m.lod_value,
                    "n_bdl": m.n_bdl,
                    "n_aloq": m.n_aloq,
                }
                for m in self.metadata
            ]
        }
        return report
