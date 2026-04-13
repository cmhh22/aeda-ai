"""
aeda.pipeline.runner
Orquestador del pipeline AEDA completo.
Conecta ingesta → auto-selector → engine → reporte.
"""

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


@dataclass
class AEDAResults:
    """Contenedor de todos los resultados del pipeline."""
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
        lines = ["=" * 60, "RESULTADOS AEDA-AI", "=" * 60]

        if self.dataset_info:
            lines.append(f"\nDataset: {self.dataset_info.n_rows} × {self.dataset_info.n_cols}")
            lines.append(f"Variables de medición: {len(self.dataset_info.measurement_cols)}")

        if self.validation:
            lines.append(f"\nValidación: {len(self.validation.issues)} problemas detectados")
            lines.append(f"Completitud: {self.validation.completeness_pct:.1f}%")

        if self.preprocessing_log:
            lines.append(f"\nPreprocesamiento: {len(self.preprocessing_log.steps)} pasos")

        if self.dim_reduction:
            lines.append(f"\nReducción dimensional: {self.dim_reduction.method}")
            if self.dim_reduction.explained_variance is not None:
                total = self.dim_reduction.diagnostics.get("total_variance_explained", 0)
                lines.append(f"  Varianza explicada: {total:.1%}")
                lines.append(f"  Componentes: {self.dim_reduction.n_components_selected}")

        if self.clustering:
            lines.append(f"\nClustering: {self.clustering.method}")
            lines.append(f"  Clusters: {self.clustering.n_clusters}")
            sil = self.clustering.metrics.get("silhouette")
            if sil:
                lines.append(f"  Silhouette: {sil:.3f}")

        if self.anomalies:
            lines.append(f"\nAnomalías: {self.anomalies.method}")
            lines.append(f"  Detectadas: {self.anomalies.n_anomalies}")

        if self.correlations:
            if isinstance(self.correlations, dict):
                p = self.correlations.get("pearson")
                if p:
                    lines.append(f"\nCorrelaciones: {p.n_strong} fuertes, {p.n_moderate} moderadas")
                nl = self.correlations.get("nonlinear_candidates", [])
                if nl:
                    lines.append(f"  Candidatos no-lineales: {len(nl)}")
            elif isinstance(self.correlations, CorrelationResult):
                lines.append(f"\nCorrelaciones ({self.correlations.method}): "
                             f"{self.correlations.n_strong} fuertes, {self.correlations.n_moderate} moderadas")

        if self.feature_importance:
            lines.append(f"\nFeature importance ({self.feature_importance.method}):")
            top5 = self.feature_importance.top_n(5)
            for var, imp in top5.items():
                lines.append(f"  {var}: {imp:.4f}")

        lines.append("=" * 60)
        return "\n".join(lines)


class AEDAPipeline:
    """
    Pipeline principal de AEDA-AI.

    Uso básico:
        pipeline = AEDAPipeline()
        results = pipeline.run("datos_ambientales.xlsx")
        print(results.summary())

    Uso avanzado:
        pipeline = AEDAPipeline(
            scale_method="robust",
            clustering_method="dbscan",
            dim_method="pca",
        )
        results = pipeline.run("datos.csv", exclude_cols=["No", "Code"])
    """

    def __init__(
        self,
        scale_method: str = "auto",
        impute_strategy: str = "auto",
        dim_method: str = "auto",
        clustering_method: str = "auto",
        anomaly_method: str = "auto",
        correlation_method: str = "compare",
        apply_clr: bool = False,
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
        Ejecuta el pipeline completo.

        Parameters
        ----------
        filepath : Ruta al archivo de datos
        exclude_cols : Columnas a excluir del análisis (IDs, códigos, etc.)
        sheet_name : Hoja de Excel a usar

        Returns
        -------
        AEDAResults con todos los resultados del análisis.
        """
        results = AEDAResults()

        # 1. INGESTA
        df, info = load(filepath, sheet_name=sheet_name)
        results.raw_data = df
        results.dataset_info = info

        # 2. VALIDACIÓN
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

        # Resolver parámetros "auto" desde el plan
        preproc_recs = [r for r in plan.recommendations if r.category == "preprocessing"]
        effective_scale = self.scale_method
        effective_impute = self.impute_strategy
        effective_clr = self.apply_clr

        for rec in preproc_recs:
            if rec.priority == 1:
                if "scale_method" in rec.params and effective_scale == "auto":
                    effective_scale = rec.params["scale_method"]
                if "impute_strategy" in rec.params and effective_impute == "auto":
                    effective_impute = rec.params["impute_strategy"]
                if "apply_clr" in rec.params:
                    effective_clr = effective_clr or rec.params["apply_clr"]

        if effective_scale == "auto":
            effective_scale = "standard"
        if effective_impute == "auto":
            effective_impute = "median"

        # 4. PREPROCESAMIENTO
        processed, proc_log, scaler = preprocess(
            df,
            exclude_cols=exclude_cols,
            impute_strategy=effective_impute,
            scale_method=effective_scale,
            apply_clr=effective_clr,
        )
        results.processed_data = processed
        results.preprocessing_log = proc_log

        # 5. REDUCCIÓN DIMENSIONAL
        try:
            results.dim_reduction = reduce(processed, method=self.dim_method)
        except Exception as e:
            results.dim_reduction = None

        # 6. CLUSTERING
        try:
            results.clustering = cluster(processed, method=self.clustering_method)
        except Exception as e:
            results.clustering = None

        # 7. DETECCIÓN DE ANOMALÍAS
        try:
            results.anomalies = detect_anomalies(
                processed,
                method=self.anomaly_method,
                contamination=self.contamination,
            )
        except Exception as e:
            results.anomalies = None

        # 8. CORRELACIONES
        try:
            results.correlations = correlate(processed, method=self.correlation_method)
        except Exception as e:
            results.correlations = None

        # 9. FEATURE IMPORTANCE (si hay clusters)
        try:
            if results.clustering is not None:
                results.feature_importance = analyze_features(
                    processed,
                    cluster_labels=results.clustering.labels,
                )
            else:
                results.feature_importance = analyze_features(processed)
        except Exception as e:
            results.feature_importance = None

        return results
