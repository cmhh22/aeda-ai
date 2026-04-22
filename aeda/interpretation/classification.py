"""Classification of samples by TEL/PEL thresholds and EF bands."""

from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd

from aeda.interpretation.thresholds import get_thresholds


class TELPELClass(str, Enum):
    """Toxicological classification per Buchman (2008) / Long & MacDonald (1998)."""

    BELOW_TEL = "below_TEL"
    BETWEEN_TEL_PEL = "TEL_to_PEL"
    ABOVE_PEL = "above_PEL"
    NO_THRESHOLDS = "no_thresholds"


class EFClass(str, Enum):
    """Enrichment classification per Birch (2003)."""

    NO_ENRICHMENT = "no_enrichment"
    MINOR = "minor"
    MODERATE = "moderate"
    MODERATELY_SEVERE = "moderately_severe"
    SEVERE = "severe"
    VERY_SEVERE = "very_severe"
    EXTREMELY_SEVERE = "extremely_severe"


def classify_tel_pel(
    concentrations: pd.Series,
    metal: str,
) -> pd.Series:
    """Classify each concentration against TEL/PEL for one metal."""
    try:
        t = get_thresholds(metal)
    except KeyError:
        return pd.Series(
            [TELPELClass.NO_THRESHOLDS.value] * len(concentrations),
            index=concentrations.index,
        )

    if t.tel is None or t.pel is None:
        return pd.Series(
            [TELPELClass.NO_THRESHOLDS.value] * len(concentrations),
            index=concentrations.index,
        )

    out = pd.Series(index=concentrations.index, dtype=object)
    for idx, val in concentrations.items():
        if pd.isna(val):
            out[idx] = np.nan
        elif val < t.tel:
            out[idx] = TELPELClass.BELOW_TEL.value
        elif val < t.pel:
            out[idx] = TELPELClass.BETWEEN_TEL_PEL.value
        else:
            out[idx] = TELPELClass.ABOVE_PEL.value
    return out


def classify_ef_birch(ef_values: pd.Series) -> pd.Series:
    """Classify EF values using Birch (2003) bands."""

    def _band(v):
        if pd.isna(v):
            return np.nan
        if v <= 2:
            return EFClass.NO_ENRICHMENT.value
        if v <= 3:
            return EFClass.MINOR.value
        if v <= 5:
            return EFClass.MODERATE.value
        if v <= 10:
            return EFClass.MODERATELY_SEVERE.value
        if v <= 25:
            return EFClass.SEVERE.value
        if v <= 50:
            return EFClass.VERY_SEVERE.value
        return EFClass.EXTREMELY_SEVERE.value

    return ef_values.apply(_band)
