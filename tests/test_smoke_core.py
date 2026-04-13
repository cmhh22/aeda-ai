"""Smoke tests for the packaged AEDA-AI core modules."""

from __future__ import annotations

import numpy as np
import pandas as pd

from aeda.engine.anomalies import detect_anomalies
from aeda.engine.auto_selector import auto_select
from aeda.engine.clustering import cluster
from aeda.engine.correlations import correlate
from aeda.engine.dimensionality import reduce
from aeda.engine.feature_analysis import analyze_features
from aeda.io.preprocessor import preprocess


def _build_df(n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "Na": rng.normal(2.0, 0.3, size=n),
            "Mg": rng.normal(1.5, 0.2, size=n),
            "Cr": rng.normal(60, 10, size=n),
            "Pb": rng.normal(35, 7, size=n),
            "clay": rng.uniform(20, 45, size=n),
            "silt": rng.uniform(25, 55, size=n),
            "sand": rng.uniform(5, 35, size=n),
        }
    )


def test_core_smoke() -> None:
    raw = _build_df()

    processed, _, _ = preprocess(raw, scale_method="standard", impute_strategy="median")
    plan = auto_select(processed)

    dim = reduce(processed, method="auto")
    clu = cluster(processed, method="auto")
    anom = detect_anomalies(processed, method="auto")
    corr = correlate(processed, method="auto")
    feat = analyze_features(processed, method="auto", cluster_labels=clu.labels)

    assert plan.profile.n_samples == len(raw)
    assert dim.components.shape[0] == len(raw)
    assert len(clu.labels) == len(raw)
    assert anom.n_anomalies >= 0
    assert isinstance(corr, dict)
    assert len(feat.importances) == processed.shape[1]
