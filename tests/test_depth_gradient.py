"""Regression tests for _detect_depth_gradient.

Covers the failure modes that crashed the function on the real ISOVIDA dataset:
- Duplicate column names in the source DataFrame.
- Zero-variance measurement columns.
- NaN values in measurement columns.
- Missing depth column.
- Happy path with a real depth gradient.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aeda.engine.auto_selector import _detect_depth_gradient


def test_detect_depth_gradient_with_duplicate_column_names():
    """Duplicate column names used to make spearmanr return matrices and crash."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        rng.standard_normal((50, 3)),
        columns=["Profundidad", "Pb", "Profundidad"],
    )
    # Must not raise; with random data the gradient should not be flagged.
    result = _detect_depth_gradient(df, "Profundidad", ["Pb"])
    assert result is False


def test_detect_depth_gradient_with_zero_variance_column():
    """Zero-variance columns must be skipped, not crash."""
    depth = np.linspace(0, 10, 50)
    df = pd.DataFrame({
        "Profundidad": depth,
        "ConstantMetal": np.zeros(50),       # zero variance
        "RealMetal": depth * 1.5 + 0.1,      # genuine gradient
    })
    # With 1/2 columns showing a gradient (ConstantMetal skipped, RealMetal counted)
    # and threshold_pct=0.3, the function should still flag the gradient.
    result = _detect_depth_gradient(df, "Profundidad", ["ConstantMetal", "RealMetal"])
    assert result is True


def test_detect_depth_gradient_handles_nans():
    """Pairwise NaN rows must be dropped, not propagated into the correlation."""
    depth = np.linspace(0, 20, 50)
    df = pd.DataFrame({"Profundidad": depth, "Pb": depth * 2.0})
    df.loc[::3, "Pb"] = np.nan  # introduce ~33% NaN
    result = _detect_depth_gradient(df, "Profundidad", ["Pb"])
    assert result is True


def test_detect_depth_gradient_returns_false_when_depth_missing():
    """Missing depth column must return False, not raise."""
    df = pd.DataFrame({"Pb": np.linspace(0, 10, 50), "Zn": np.linspace(0, 5, 50)})
    assert _detect_depth_gradient(df, "Profundidad", ["Pb", "Zn"]) is False


def test_detect_depth_gradient_happy_path():
    """When most columns correlate with depth, the function must flag the gradient."""
    rng = np.random.default_rng(42)
    depth = np.linspace(0, 20, 80)
    df = pd.DataFrame({
        "Profundidad": depth,
        "Pb": depth * 2.0 + rng.standard_normal(80) * 0.5,
        "Zn": -1.5 * depth + rng.standard_normal(80) * 0.5,
        "Cu": depth * 0.8 + rng.standard_normal(80) * 0.3,
    })
    assert _detect_depth_gradient(df, "Profundidad", ["Pb", "Zn", "Cu"]) is True
