"""
aeda.pipeline.runner
Orchestrator for the full AEDA pipeline.
Connects ingestion -> auto-selector -> engine -> report.
"""

import logging
import pandas as pd
from typing import Optional, Union
from pathlib import Path
from dataclasses import dataclass, field

from aeda.io.parsers import load, DatasetInfo
from aeda.io.validators import validate, ValidationReport
from aeda.io.preprocessor import preprocess, select_numeric, PreprocessingLog
from aeda.engine.auto_selector import auto_select, AnalysisPlan
from aeda.engine.dimensionality import reduce, DimReductionResult
from aeda.engine.clustering import cluster, ClusteringResult
from aeda.engine.anomalies import detect_anomalies, AnomalyResult
from aeda.engine.correlations import correlate, CorrelationResult
from aeda.engine.feature_analysis import analyze_features, FeatureImportanceResult


logger = logging.getLogger(__name__)


@dataclass
class AEDAResults:
    """Container for all pipeline results."""
    # Input
    raw_data: Optional[pd.DataFrame] = None
    dataset_info: Optional[DatasetInfo] = None

    # Validation
    validation: Optional[ValidationReport] = None

    # Preprocessing
    processed_data: Optional[pd.DataFrame] = None
    preprocessing_log: Optional[PreprocessingLog] = None

    # Analysis plan
    plan: Optional[AnalysisPlan] = None

    # Engine results
    dim_reduction: Optional[DimReductionResult] = None
    clustering: Optional[ClusteringResult] = None
    anomalies: Optional[AnomalyResult] = None
    correlations: Optional[Union[CorrelationResult, dict]] = None
    feature_importance: Optional[FeatureImportanceResult] = None

    def summary(self) -> str:
        lines = ["=" * 60, "AEDA-AI RESULTS", "=" * 60]

        if self.dataset_info:
            lines.append(f"\nDataset: {self.dataset_info.n_rows} × {self.dataset_info.n_cols}")
            lines.append(f"Measurement variables: {len(self.dataset_info.measurement_cols)}")

        if self.validation:
            lines.append(f"\nValidation: {len(self.validation.issues)} issues detected")
            lines.append(f"Completeness: {self.validation.completeness_pct:.1f}%")

        if self.preprocessing_log:
            lines.append(f"\nPreprocessing: {len(self.preprocessing_log.steps)} steps")

        if self.dim_reduction:
            lines.append(f"\nDimensionality reduction: {self.dim_reduction.method}")
            if self.dim_reduction.explained_variance is not None:
                total = self.dim_reduction.diagnostics.get("total_variance_explained", 0)
                lines.append(f"  Explained variance: {total:.1%}")
                lines.append(f"  Components: {self.dim_reduction.n_components_selected}")
        else:
            lines.append("\nDimensionality reduction: FAILED (check logs)")

        if self.clustering:
            lines.append(f"\nClustering: {self.clustering.method}")
            lines.append(f"  Number of clusters: {self.clustering.n_clusters}")
            sil = self.clustering.metrics.get("silhouette")
            if sil:
                lines.append(f"  Silhouette: {sil:.3f}")
        else:
            lines.append("\nClustering: FAILED (check logs)")

        if self.anomalies:
            lines.append(f"\nAnomalies: {self.anomalies.method}")
            lines.append(f"  Detected: {self.anomalies.n_anomalies}")
        else:
            lines.append("\nAnomalies: FAILED (check logs)")

        if self.correlations:
            if isinstance(self.correlations, dict):
                p = self.correlations.get("pearson")
                if p:
                    lines.append(f"\nCorrelations: {p.n_strong} strong, {p.n_moderate} moderate")
                nl = self.correlations.get("nonlinear_candidates", [])
                if nl:
                    lines.append(f"  Nonlinear candidates: {len(nl)}")
            elif isinstance(self.correlations, CorrelationResult):
                lines.append(f"\nCorrelations ({self.correlations.method}): "
                             f"{self.correlations.n_strong} strong, {self.correlations.n_moderate} moderate")
        else:
            lines.append("\nCorrelations: FAILED (check logs)")

        if self.feature_importance:
            lines.append(f"\nFeature importance ({self.feature_importance.method}):")
            top5 = self.feature_importance.top_n(5)
            for var, imp in top5.items():
                lines.append(f"  {var}: {imp:.4f}")
        else:
            lines.append("\nFeature importance: FAILED (check logs)")

        lines.append("=" * 60)
        return "\n".join(lines)


class AEDAPipeline:
    """
    Main AEDA-AI pipeline.

    Basic usage:
        pipeline = AEDAPipeline()
        results = pipeline.run("environmental_data.xlsx")
        print(results.summary())

    Advanced usage:
        pipeline = AEDAPipeline(
            scale_method="robust",
            clustering_method="dbscan",
            dim_method="pca",
        )
        results = pipeline.run("data.csv", exclude_cols=["No", "Code"])
    """

    def __init__(
        self,
        scale_method: str = "auto",
        impute_strategy: str = "auto",
        dim_method: str = "auto",
        clustering_method: str = "auto",
        anomaly_method: str = "auto",
        correlation_method: str = "compare",
        apply_clr: bool | str | None = False,
        contamination: float = 0.05,
    ):
        self.scale_method = scale_method
        self.impute_strategy = impute_strategy
        self.dim_method = dim_method
        self.clustering_method = clustering_method
        self.anomaly_method = anomaly_method
        self.correlation_method = correlation_method
        self.apply_clr = apply_clr
        self.contamination = contamination

    def run(
        self,
        filepath: Union[str, Path],
        exclude_cols: Optional[list[str]] = None,
        sheet_name: Optional[str] = None,
    ) -> AEDAResults:
        """
        Run the full pipeline.

        Parameters
        ----------
        filepath : str or Path
            Path to the data file.
        exclude_cols : list[str], optional
            Columns to exclude from analysis (IDs, codes, etc.).
        sheet_name : str, optional
            Excel sheet name to use.

        Returns
        -------
        AEDAResults
            Object with all analysis results.
        """
        results = AEDAResults()

        # 1. INGESTION
        df, info = load(filepath, sheet_name=sheet_name)
        results.raw_data = df
        results.dataset_info = info

        # 2. VALIDATION
        results.validation = validate(df, measurement_cols=info.measurement_cols)

        # 3. AUTO-SELECTOR
        numeric_df = select_numeric(df, exclude_cols=exclude_cols)
        plan = auto_select(
            numeric_df,
            has_coordinates=len(info.coordinate_cols) > 0,
            has_depth=info.depth_col is not None,
            depth_col=info.depth_col,
            has_sites=info.site_col is not None,
            site_col=info.site_col,
            n_sites=df[info.site_col].nunique() if info.site_col else 0,
            original_df=df,
        )
        results.plan = plan

        # Resolve "auto" parameters from the plan
        VALID_IMPUTE_STRATEGIES = {"drop_rows", "drop_cols", "mean", "median", "knn"}
        VALID_SCALE_METHODS = {"standard", "minmax", "robust"}

        preproc_recs = [r for r in plan.recommendations if r.category == "preprocessing"]
        effective_scale = self.scale_method
        effective_impute = self.impute_strategy
        effective_clr = self.apply_clr

        for rec in preproc_recs:
            if rec.priority == 1:
                if "scale_method" in rec.params and effective_scale == "auto":
                    candidate = rec.params["scale_method"]
                    if candidate in VALID_SCALE_METHODS:
                        effective_scale = candidate
                if "impute_strategy" in rec.params and effective_impute == "auto":
                    candidate = rec.params["impute_strategy"]
                    # Map unsupported recommendations to safe defaults
                    if candidate in VALID_IMPUTE_STRATEGIES:
                        effective_impute = candidate
                    elif candidate == "subset_analysis":
                        # Structured-missing data: subset analysis is not implemented yet.
                        # Fall back to median, a safe default for environmental datasets.
                        effective_impute = "median"
                if "apply_clr" in rec.params:
                    # Only override user choice if the user didn't set an explicit boolean.
                    if self.apply_clr is None or self.apply_clr == "auto":
                        effective_clr = rec.params["apply_clr"]

        # Final fallbacks for any remaining "auto" value
        if effective_scale == "auto":
            effective_scale = "standard"
        if effective_impute == "auto":
            effective_impute = "median"
        if effective_clr == "auto":
            effective_clr = False

        # 4. PREPROCESSING
        processed, proc_log, scaler = preprocess(
            df,
            exclude_cols=exclude_cols,
            impute_strategy=effective_impute,
            scale_method=effective_scale,
            apply_clr=effective_clr,
        )
        results.processed_data = processed
        results.preprocessing_log = proc_log

        # 5. DIMENSIONALITY REDUCTION
        try:
            results.dim_reduction = reduce(processed, method=self.dim_method)
        except Exception as e:
            logger.warning(f"Dimensionality reduction failed: {type(e).__name__}: {e}")
            results.dim_reduction = None

        # 6. CLUSTERING
        try:
            results.clustering = cluster(processed, method=self.clustering_method)
        except Exception as e:
            logger.warning(f"Clustering failed: {type(e).__name__}: {e}")
            results.clustering = None

        # 7. ANOMALY DETECTION
        try:
            results.anomalies = detect_anomalies(
                processed,
                method=self.anomaly_method,
                contamination=self.contamination,
            )
        except Exception as e:
            logger.warning(f"Anomaly detection failed: {type(e).__name__}: {e}")
            results.anomalies = None

        # 8. CORRELATIONS
        try:
            results.correlations = correlate(processed, method=self.correlation_method)
        except Exception as e:
            logger.warning(f"Correlation analysis failed: {type(e).__name__}: {e}")
            results.correlations = None

        # 9. FEATURE IMPORTANCE (if clusters are available)
        try:
            if results.clustering is not None:
                results.feature_importance = analyze_features(
                    processed,
                    cluster_labels=results.clustering.labels,
                )
            else:
                results.feature_importance = analyze_features(processed)
        except Exception as e:
            logger.warning(f"Feature importance failed: {type(e).__name__}: {e}")
            results.feature_importance = None

        return results
