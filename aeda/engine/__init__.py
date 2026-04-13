"""Core ML engine modules for AEDA-AI."""

from aeda.engine.auto_selector import (
    AnalysisPlan,
    AnalysisScale,
    Confidence,
    DataProfile,
    MethodRecommendation,
    auto_select,
)
from aeda.engine.dimensionality import DimReductionResult, reduce
from aeda.engine.clustering import ClusteringResult, cluster
from aeda.engine.anomalies import AnomalyResult, detect_anomalies
from aeda.engine.correlations import CorrelationResult, correlate
from aeda.engine.feature_analysis import FeatureImportanceResult, analyze_features

__all__ = [
    "AnalysisPlan",
    "AnalysisScale",
    "Confidence",
    "DataProfile",
    "MethodRecommendation",
    "auto_select",
    "DimReductionResult",
    "reduce",
    "ClusteringResult",
    "cluster",
    "AnomalyResult",
    "detect_anomalies",
    "CorrelationResult",
    "correlate",
    "FeatureImportanceResult",
    "analyze_features",
]
