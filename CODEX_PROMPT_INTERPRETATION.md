# MÓDULO DE INTERPRETACIÓN AMBIENTAL — AEDA-AI

## Contexto

El framework AEDA-AI actualmente tiene capas de ingesta, motor ML, orquestación y visualización. Falta la capa de **interpretación ambiental específica del dominio**: aplicar el Factor de Enriquecimiento (EF), comparar contra umbrales TEL/PEL, y clasificar muestras por nivel de contaminación.

Esta capa es conceptualmente distinta del motor ML (que es exploratorio y genérico) — aquí entra el conocimiento toxicológico específico para sedimentos.

## Referencias científicas implementadas

Este módulo implementa los criterios discutidos con el tutor Yoelvis y respaldados por tres papers de referencia:

1. **Succop et al. (2004)** — imputación de valores bajo el Límite de Detección del Método (LDM).
   - Criterio simple aceptado para la tesis: si la variable es Normal → reemplazar <LDM por LDM/2; si no es Normal → reemplazar por LDM/√2.

2. **Buchman (2008) — NOAA Screening Quick Reference Tables.** Valores TEL/PEL/ERL/ERM para sedimentos marinos.

3. **Bolaños-Alvarez et al. (2024) — Sci. Total Environ. 920.** Metodología para cálculo de Enrichment Factor y clasificación de Birch (2003).

## Arquitectura a construir

```
aeda/interpretation/
├── __init__.py
├── lod.py               # Simple LOD imputation (Yoelvis criterion)
├── thresholds.py        # TEL/PEL/ERL/ERM tables (NOAA Buchman 2008)
├── normalization.py     # Enrichment Factor computation
├── classification.py    # Classify samples vs TEL/PEL and EF bands
└── reporter.py          # InterpretationReport dataclass + build function

aeda/viz/
├── interpretation.py    # NEW FILE: ef_depth_plot, contamination_bar_plot
└── profiles.py          # EXTEND: depth_profile_with_thresholds

tests/
└── test_interpretation.py  # NEW FILE: 8+ regression tests
```

---

## FILE 1 — `aeda/interpretation/__init__.py`

```python
"""Environmental interpretation layer for AEDA-AI.

This module applies domain-specific toxicological and geochemical analysis
on top of the exploratory results produced by the ML engine. It computes
enrichment factors, applies regulatory thresholds (TEL/PEL), and classifies
samples by contamination level.

References
----------
Buchman, M. F. (2008). NOAA Screening Quick Reference Tables. NOAA OR&R Report 08-1.
Bolaños-Alvarez et al. (2024). Sci. Total Environ. 920, 170609.
Succop et al. (2004). J. Occup. Environ. Hyg. 1(7), 436-441.
Birch, G. (2003). A scheme for assessing human impacts on coastal aquatic environments.
"""

from aeda.interpretation.lod import handle_lod_values, LODImputationLog
from aeda.interpretation.thresholds import (
    TEL_PEL_MARINE_SEDIMENT,
    get_thresholds,
)
from aeda.interpretation.normalization import (
    compute_enrichment_factor,
    EnrichmentFactorResult,
)
from aeda.interpretation.classification import (
    classify_tel_pel,
    classify_ef_birch,
    TELPELClass,
    EFClass,
)
from aeda.interpretation.reporter import (
    InterpretationReport,
    build_interpretation_report,
)

__all__ = [
    "handle_lod_values",
    "LODImputationLog",
    "TEL_PEL_MARINE_SEDIMENT",
    "get_thresholds",
    "compute_enrichment_factor",
    "EnrichmentFactorResult",
    "classify_tel_pel",
    "classify_ef_birch",
    "TELPELClass",
    "EFClass",
    "InterpretationReport",
    "build_interpretation_report",
]
```

---

## FILE 2 — `aeda/interpretation/lod.py`

Implementa el criterio simple de Yoelvis para imputación de valores bajo LDM.

```python
"""Handle values below the limit of detection (LOD) using Succop et al. (2004) simple criterion.

This module implements the criterion recommended by the project's scientific
advisor (Yoelvis Bolaños-Alvarez):

- If the variable follows a Normal distribution: replace <LOD values by LOD/2
- If the variable does NOT follow a Normal distribution: replace by LOD/sqrt(2)

Normality is tested per-variable using the Shapiro-Wilk test at alpha=0.05.

References
----------
Succop, P. A., Clark, S., Chen, M., & Galke, W. (2004). Imputation of data
values that are less than a detection limit. J. Occupational and Environmental
Hygiene, 1(7), 436-441.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class LODImputationLog:
    """Record of LOD imputation decisions for audit and reporting."""
    variable_decisions: dict = field(default_factory=dict)  # {var: {"normal": bool, "p_value": float, "method": str, "n_imputed": int}}
    alpha: float = 0.05

    def summary(self) -> str:
        lines = ["=" * 60, "LOD IMPUTATION LOG", "=" * 60]
        if not self.variable_decisions:
            lines.append("No variables had LOD values to impute.")
            return "\n".join(lines)
        for var, info in self.variable_decisions.items():
            lines.append(
                f"  {var}: n_imputed={info['n_imputed']}, "
                f"normal={info['normal']} (p={info['p_value']:.4f}), "
                f"method={info['method']}"
            )
        return "\n".join(lines)


def handle_lod_values(
    df: pd.DataFrame,
    lod_values: dict[str, float],
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, LODImputationLog]:
    """Replace values below the limit of detection (LOD) using Yoelvis's criterion.

    For each variable in `lod_values`:
    1. Test whether the non-missing, non-below-LOD values are Normal (Shapiro-Wilk).
    2. If Normal (p > alpha): replace values flagged as <LOD by LOD/2.
    3. If not Normal: replace by LOD/sqrt(2).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with numeric columns. Values below LOD must be pre-marked
        as NaN, or alternatively, values less than the LOD itself are treated
        as below-LOD (the function detects both cases).
    lod_values : dict[str, float]
        Mapping from variable name to its limit of detection (in the same
        units as the data). Only variables present both in `df` and in this
        dict are processed; others are left untouched.
    alpha : float
        Significance level for the Shapiro-Wilk normality test. Default 0.05.

    Returns
    -------
    tuple[pd.DataFrame, LODImputationLog]
        (df with imputed values, log with per-variable decisions)

    Notes
    -----
    This is the "simple" method per the project tutor. For a more rigorous
    "most probable value" method (Cohen 1961 + Succop median-below-LOD),
    see references in Succop et al. 2004.

    Shapiro-Wilk requires at least 3 samples; fewer samples are treated as
    non-Normal by default (defensive choice).
    """
    out = df.copy()
    log = LODImputationLog(alpha=alpha)

    for var, lod in lod_values.items():
        if var not in out.columns:
            continue
        if lod <= 0:
            raise ValueError(f"LOD for variable '{var}' must be positive, got {lod}")

        col = out[var]
        # Mark values below LOD: either already NaN, or numerically below the LOD
        below_lod_mask = col.isna() | (col < lod)
        valid_mask = ~below_lod_mask
        valid_values = col[valid_mask].dropna()

        if len(valid_values) < 3:
            # Shapiro-Wilk requires n >= 3; default to non-Normal (safer for
            # skewed environmental data)
            is_normal = False
            p_value = float("nan")
        else:
            # Shapiro-Wilk has an upper limit ~5000 samples; subsample if needed
            if len(valid_values) > 5000:
                rng = np.random.default_rng(42)
                sample = rng.choice(valid_values.values, size=5000, replace=False)
            else:
                sample = valid_values.values
            try:
                _, p_value = stats.shapiro(sample)
                is_normal = p_value > alpha
            except Exception:
                is_normal = False
                p_value = float("nan")

        if is_normal:
            replacement = lod / 2.0
            method = "LOD/2 (Normal)"
        else:
            replacement = lod / np.sqrt(2.0)
            method = "LOD/sqrt(2) (non-Normal)"

        n_imputed = int(below_lod_mask.sum())
        if n_imputed > 0:
            out.loc[below_lod_mask, var] = replacement

        log.variable_decisions[var] = {
            "normal": is_normal,
            "p_value": float(p_value),
            "method": method,
            "n_imputed": n_imputed,
            "replacement_value": float(replacement),
            "lod": float(lod),
        }

    return out, log
```

---

## FILE 3 — `aeda/interpretation/thresholds.py`

Tabla hardcodeada de umbrales TEL/PEL/ERL/ERM para sedimentos marinos, extraída directamente del reporte NOAA Buchman (2008), página 2 (Marine Sediment section). Todos los valores están convertidos a **mg/kg** (ppm) para consistencia con la unidad típica de las mediciones FRX para metales traza.

```python
"""Regulatory thresholds for sediment contamination (Buchman 2008 - NOAA SQuiRTs).

All concentrations are expressed in mg/kg (equivalent to ppm or µg/g) on a
dry weight basis.

Sources
-------
Buchman, M. F. (2008). NOAA Screening Quick Reference Tables, NOAA OR&R
Report 08-1. Marine Sediment section (page 2 of the report).

Threshold definitions
---------------------
- TEL (Threshold Effect Level):    concentration below which adverse biological
  effects are expected only rarely.
- PEL (Probable Effect Level):     concentration above which adverse biological
  effects are expected frequently.
- ERL (Effects Range - Low):       10th percentile of effects dataset.
- ERM (Effects Range - Median):    50th percentile of effects dataset.

Interpretation (per Buchman 2008 and Long & MacDonald 1998):
- concentration < TEL:             rarely associated with adverse effects
- TEL <= concentration < PEL:      occasional adverse effects
- concentration >= PEL:            frequent adverse effects
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class MetalThresholds:
    """Regulatory thresholds for a single metal in marine sediment (mg/kg)."""
    tel: Optional[float] = None
    pel: Optional[float] = None
    erl: Optional[float] = None
    erm: Optional[float] = None


# Values from Buchman (2008) NOAA SQuiRTs, MARINE SEDIMENT section.
# Original units in the document: µg/kg (ppb). Converted to mg/kg for
# consistency with FRX trace-element reporting conventions.
TEL_PEL_MARINE_SEDIMENT: dict[str, MetalThresholds] = {
    "As": MetalThresholds(tel=7.24, pel=41.6, erl=8.2, erm=70.0),
    "Cd": MetalThresholds(tel=0.68, pel=4.21, erl=1.2, erm=9.6),
    "Cr": MetalThresholds(tel=52.3, pel=160.0, erl=81.0, erm=370.0),
    "Cu": MetalThresholds(tel=18.7, pel=108.0, erl=34.0, erm=270.0),
    "Hg": MetalThresholds(tel=0.13, pel=0.70, erl=0.15, erm=0.71),
    "Ni": MetalThresholds(tel=15.9, pel=42.8, erl=20.9, erm=51.6),
    "Pb": MetalThresholds(tel=30.24, pel=112.0, erl=46.7, erm=218.0),
    "Zn": MetalThresholds(tel=124.0, pel=271.0, erl=150.0, erm=410.0),
    "Ag": MetalThresholds(tel=0.73, pel=1.77, erl=1.0, erm=3.7),
    "Sb": MetalThresholds(tel=None, pel=None, erl=2.0, erm=25.0),
}


def get_thresholds(metal: str) -> MetalThresholds:
    """Retrieve the marine-sediment thresholds for a given metal.

    Parameters
    ----------
    metal : str
        Symbol of the metal (e.g., 'Pb', 'Hg', 'Zn'). Case-sensitive.

    Returns
    -------
    MetalThresholds
        Dataclass with .tel, .pel, .erl, .erm attributes (any may be None
        if not available in the NOAA table).

    Raises
    ------
    KeyError
        If the metal is not in the NOAA marine sediment table.
    """
    if metal not in TEL_PEL_MARINE_SEDIMENT:
        raise KeyError(
            f"Metal '{metal}' not found in NOAA marine-sediment thresholds. "
            f"Available: {sorted(TEL_PEL_MARINE_SEDIMENT.keys())}"
        )
    return TEL_PEL_MARINE_SEDIMENT[metal]
```

---

## FILE 4 — `aeda/interpretation/normalization.py`

```python
"""Enrichment Factor (EF) computation for geochemical normalization.

The Enrichment Factor quantifies the degree to which a metal concentration
in a given sediment sample exceeds a natural reference level, after
normalizing against a conservative lithogenic element (commonly Al).

Formula (Buat-Menard & Chesselet 1979, applied in Bolaños-Alvarez 2024):

    EF = (Metal / Reference)_sample / (Metal / Reference)_baseline

Interpretation:
- EF ~ 1:    no enrichment, natural origin
- EF > 1:    increasing contribution of anthropogenic sources
- See aeda.interpretation.classification.classify_ef_birch for bands.

Baseline strategy
-----------------
In absence of radiometric dating (e.g. Pb-210) to identify pre-anthropogenic
sections (>100 years old), AEDA-AI uses the DEEPEST core section per
sampling site as an approximation of the pre-industrial baseline. This is
a documented limitation for the thesis (Chapter 3 discussion) but follows
common practice for shorter cores without dating.

References
----------
Buat-Menard, P., & Chesselet, R. (1979). Earth Planet. Sci. Lett. 42(3), 399-411.
Bolaños-Alvarez et al. (2024). Sci. Total Environ. 920, 170609.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import pandas as pd


@dataclass
class EnrichmentFactorResult:
    """Result of an EF computation for one or more metals."""
    ef_values: pd.DataFrame  # index: sample index; columns: metal names
    reference_element: str
    baseline_concentrations: dict  # {site: {metal+"/"+reference: value}}
    baseline_strategy: str
    diagnostics: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = ["=" * 60, "ENRICHMENT FACTOR RESULT", "=" * 60]
        lines.append(f"Reference element: {self.reference_element}")
        lines.append(f"Baseline strategy: {self.baseline_strategy}")
        lines.append(f"Metals analyzed: {list(self.ef_values.columns)}")
        lines.append(f"Samples: {len(self.ef_values)}")
        lines.append("\nEF descriptive statistics per metal:")
        lines.append(self.ef_values.describe().T.to_string())
        return "\n".join(lines)


def compute_enrichment_factor(
    df: pd.DataFrame,
    metals: list[str],
    reference_element: str = "Al",
    site_col: Optional[str] = None,
    depth_col: Optional[str] = None,
    baseline_strategy: str = "deepest",
    custom_baseline: Optional[dict] = None,
) -> EnrichmentFactorResult:
    """Compute EF = (Metal/Ref)_sample / (Metal/Ref)_baseline per sample.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data including metal concentrations and (optionally) site/depth columns.
    metals : list[str]
        Names of metal columns to compute EF for. The reference element must
        NOT be in this list.
    reference_element : str
        Name of the conservative lithogenic element used for normalization.
        Default "Al". Common alternatives: "Fe", "Ti", "Sc", "Li".
    site_col : str, optional
        Name of the column that identifies the sampling site. If None, the
        baseline is computed globally (a single baseline for the entire
        dataset). If provided, one baseline per site is computed.
    depth_col : str, optional
        Name of the depth column (required for strategy="deepest").
    baseline_strategy : str
        How to select the baseline:
        - "deepest": use the deepest-depth sample per site (default; suitable
          for sediment cores without radiometric dating).
        - "user": use concentrations provided via `custom_baseline`.
        - "global_min_depth": use deepest sample across the entire dataset
          (ignoring sites).
    custom_baseline : dict, optional
        Only used when `baseline_strategy="user"`. Expected structure:
        {site_name: {metal: concentration, reference_element: concentration}}
        or for global baseline: {metal: concentration, reference: concentration}.

    Returns
    -------
    EnrichmentFactorResult
        Object with per-sample EF values and baseline diagnostic info.

    Raises
    ------
    ValueError
        If the reference element is not in df, if metals overlap with the
        reference, if strategy="deepest" without depth_col, or if
        strategy="user" without custom_baseline.
    """
    if reference_element not in df.columns:
        raise ValueError(
            f"Reference element '{reference_element}' not found in DataFrame. "
            f"Available numeric columns: {list(df.select_dtypes(include='number').columns)}"
        )
    if reference_element in metals:
        raise ValueError(
            f"Reference element '{reference_element}' cannot be in the metals list."
        )
    missing = [m for m in metals if m not in df.columns]
    if missing:
        raise ValueError(f"Metal columns not found in DataFrame: {missing}")

    if baseline_strategy == "deepest" and depth_col is None:
        raise ValueError(
            "baseline_strategy='deepest' requires depth_col to identify deepest samples"
        )
    if baseline_strategy == "user" and custom_baseline is None:
        raise ValueError(
            "baseline_strategy='user' requires custom_baseline dictionary"
        )

    # Compute baseline concentrations per site (or globally)
    baseline_concs = {}

    if baseline_strategy == "deepest":
        if site_col is not None:
            for site, sub in df.groupby(site_col):
                deepest_idx = sub[depth_col].idxmax()
                site_baseline = {}
                site_baseline[reference_element] = float(df.at[deepest_idx, reference_element])
                for m in metals:
                    site_baseline[m] = float(df.at[deepest_idx, m])
                baseline_concs[site] = site_baseline
        else:
            deepest_idx = df[depth_col].idxmax()
            global_baseline = {reference_element: float(df.at[deepest_idx, reference_element])}
            for m in metals:
                global_baseline[m] = float(df.at[deepest_idx, m])
            baseline_concs["__global__"] = global_baseline

    elif baseline_strategy == "global_min_depth":
        if depth_col is None:
            raise ValueError("baseline_strategy='global_min_depth' requires depth_col")
        deepest_idx = df[depth_col].idxmax()
        global_baseline = {reference_element: float(df.at[deepest_idx, reference_element])}
        for m in metals:
            global_baseline[m] = float(df.at[deepest_idx, m])
        baseline_concs["__global__"] = global_baseline

    elif baseline_strategy == "user":
        baseline_concs = custom_baseline
    else:
        raise ValueError(
            f"Unknown baseline_strategy '{baseline_strategy}'. "
            f"Valid: 'deepest', 'global_min_depth', 'user'"
        )

    # Compute EF per sample
    ef_df = pd.DataFrame(index=df.index, columns=metals, dtype=float)

    for idx, row in df.iterrows():
        ref_val_sample = row[reference_element]
        if pd.isna(ref_val_sample) or ref_val_sample == 0:
            ef_df.loc[idx, :] = np.nan
            continue

        # Determine which baseline to use
        if site_col is not None and baseline_strategy == "deepest":
            site = row[site_col]
            if site not in baseline_concs:
                ef_df.loc[idx, :] = np.nan
                continue
            base = baseline_concs[site]
        else:
            base = baseline_concs.get("__global__", next(iter(baseline_concs.values())))

        ref_val_base = base.get(reference_element)
        if ref_val_base is None or ref_val_base == 0:
            ef_df.loc[idx, :] = np.nan
            continue

        for m in metals:
            metal_val_sample = row[m]
            metal_val_base = base.get(m)
            if (
                pd.isna(metal_val_sample)
                or metal_val_base is None
                or metal_val_base == 0
            ):
                ef_df.at[idx, m] = np.nan
            else:
                ef_df.at[idx, m] = (metal_val_sample / ref_val_sample) / (
                    metal_val_base / ref_val_base
                )

    return EnrichmentFactorResult(
        ef_values=ef_df,
        reference_element=reference_element,
        baseline_concentrations=baseline_concs,
        baseline_strategy=baseline_strategy,
        diagnostics={
            "n_samples": len(df),
            "n_metals": len(metals),
            "n_sites": len(baseline_concs) if site_col else 1,
        },
    )
```

---

## FILE 5 — `aeda/interpretation/classification.py`

```python
"""Classification of samples by TEL/PEL thresholds and EF bands."""

from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from aeda.interpretation.thresholds import (
    TEL_PEL_MARINE_SEDIMENT,
    get_thresholds,
)


class TELPELClass(str, Enum):
    """Toxicological classification per Buchman (2008) / Long & MacDonald (1998)."""
    BELOW_TEL = "below_TEL"        # No adverse effects expected
    BETWEEN_TEL_PEL = "TEL_to_PEL"  # Occasional adverse effects
    ABOVE_PEL = "above_PEL"         # Frequent adverse effects
    NO_THRESHOLDS = "no_thresholds" # Metal not in NOAA table


class EFClass(str, Enum):
    """Enrichment classification per Birch (2003)."""
    NO_ENRICHMENT = "no_enrichment"              # EF <= 2
    MINOR = "minor"                              # 2 < EF <= 3
    MODERATE = "moderate"                        # 3 < EF <= 5
    MODERATELY_SEVERE = "moderately_severe"      # 5 < EF <= 10
    SEVERE = "severe"                            # 10 < EF <= 25
    VERY_SEVERE = "very_severe"                  # 25 < EF <= 50
    EXTREMELY_SEVERE = "extremely_severe"        # EF > 50


def classify_tel_pel(
    concentrations: pd.Series,
    metal: str,
) -> pd.Series:
    """Classify each concentration against TEL/PEL for the given metal.

    Parameters
    ----------
    concentrations : pd.Series
        Concentrations in mg/kg (dry weight) for the metal.
    metal : str
        Metal symbol (e.g., 'Pb'). Must exist in TEL_PEL_MARINE_SEDIMENT.

    Returns
    -------
    pd.Series
        Series of TELPELClass values (as string enum values), same index as input.
    """
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
    """Classify EF values using Birch (2003) bands.

    Parameters
    ----------
    ef_values : pd.Series
        Enrichment factor values (dimensionless).

    Returns
    -------
    pd.Series
        Series of EFClass values (as string enum values), same index as input.
    """
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
```

---

## FILE 6 — `aeda/interpretation/reporter.py`

```python
"""Build a consolidated interpretation report combining TEL/PEL and EF analyses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from aeda.interpretation.classification import (
    classify_tel_pel,
    classify_ef_birch,
    TELPELClass,
    EFClass,
)
from aeda.interpretation.normalization import (
    compute_enrichment_factor,
    EnrichmentFactorResult,
)


@dataclass
class InterpretationReport:
    """Consolidated interpretation of contamination levels."""
    tel_pel_classifications: pd.DataFrame  # samples × metals, string enum
    ef_result: Optional[EnrichmentFactorResult] = None
    ef_classifications: Optional[pd.DataFrame] = None  # samples × metals
    metals_analyzed: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = ["=" * 60, "INTERPRETATION REPORT", "=" * 60]
        lines.append(f"\nMetals analyzed: {self.metals_analyzed}")
        lines.append(f"Samples: {len(self.tel_pel_classifications)}")

        lines.append("\n--- TEL/PEL Classification Counts (per metal) ---")
        for metal in self.metals_analyzed:
            if metal in self.tel_pel_classifications.columns:
                counts = self.tel_pel_classifications[metal].value_counts(dropna=False)
                lines.append(f"\n  {metal}:")
                for label, n in counts.items():
                    lines.append(f"    {label}: {n}")

        if self.ef_result is not None:
            lines.append("\n--- Enrichment Factor Summary ---")
            lines.append(self.ef_result.summary())

            if self.ef_classifications is not None:
                lines.append("\n--- EF Classification Counts (per metal) ---")
                for metal in self.metals_analyzed:
                    if metal in self.ef_classifications.columns:
                        counts = self.ef_classifications[metal].value_counts(dropna=False)
                        lines.append(f"\n  {metal}:")
                        for label, n in counts.items():
                            lines.append(f"    {label}: {n}")

        return "\n".join(lines)


def build_interpretation_report(
    df: pd.DataFrame,
    metals: list[str],
    reference_element: str = "Al",
    site_col: Optional[str] = None,
    depth_col: Optional[str] = None,
    baseline_strategy: str = "deepest",
    custom_baseline: Optional[dict] = None,
    compute_ef: bool = True,
) -> InterpretationReport:
    """Run the full interpretation pipeline and return a consolidated report.

    Steps:
    1. Classify each metal concentration vs TEL/PEL (Buchman 2008).
    2. If `compute_ef=True` and depth_col/site_col are provided: compute EF
       using the selected baseline strategy and classify per Birch (2003).

    Parameters
    ----------
    df : pd.DataFrame
        Raw concentration data. Metal columns must be in mg/kg (dry weight).
    metals : list[str]
        Metal symbols to analyze.
    reference_element : str
        For EF normalization. Default "Al".
    site_col, depth_col : str, optional
        Required for EF with site-specific baselines.
    baseline_strategy : str
        See compute_enrichment_factor documentation.
    custom_baseline : dict, optional
        See compute_enrichment_factor documentation.
    compute_ef : bool
        Whether to compute the EF section of the report. Disable if no
        reference element is available in the dataset.

    Returns
    -------
    InterpretationReport
    """
    # TEL/PEL classification (always)
    tel_pel_df = pd.DataFrame(index=df.index, columns=metals, dtype=object)
    for metal in metals:
        if metal in df.columns:
            tel_pel_df[metal] = classify_tel_pel(df[metal], metal)

    ef_result = None
    ef_class_df = None

    if compute_ef and reference_element in df.columns:
        try:
            ef_result = compute_enrichment_factor(
                df,
                metals=metals,
                reference_element=reference_element,
                site_col=site_col,
                depth_col=depth_col,
                baseline_strategy=baseline_strategy,
                custom_baseline=custom_baseline,
            )
            ef_class_df = pd.DataFrame(index=df.index, columns=metals, dtype=object)
            for metal in metals:
                ef_class_df[metal] = classify_ef_birch(ef_result.ef_values[metal])
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                f"EF computation failed: {type(e).__name__}: {e}"
            )

    return InterpretationReport(
        tel_pel_classifications=tel_pel_df,
        ef_result=ef_result,
        ef_classifications=ef_class_df,
        metals_analyzed=metals,
        diagnostics={
            "n_samples": len(df),
            "reference_element": reference_element,
            "baseline_strategy": baseline_strategy if ef_result else None,
        },
    )
```

---

## FILE 7 — `aeda/viz/interpretation.py` (NEW)

```python
"""Visualization functions for environmental interpretation results."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from aeda.viz.base import (
    CATEGORICAL_PALETTE,
    apply_default_layout,
    get_categorical_colors,
)
from aeda.interpretation.normalization import EnrichmentFactorResult


# Colors for EF classification bands (based on Birch 2003)
EF_BAND_COLORS = {
    "no_enrichment": "#2ca02c",         # green
    "minor": "#bcbd22",                 # olive
    "moderate": "#ff7f0e",              # orange
    "moderately_severe": "#d62728",     # red
    "severe": "#8c564b",                # brown
    "very_severe": "#9467bd",           # purple
    "extremely_severe": "#1f1f1f",      # near black
}

# Colors for TEL/PEL classification
TEL_PEL_COLORS = {
    "below_TEL": "#2ca02c",        # green - safe
    "TEL_to_PEL": "#ff7f0e",       # orange - caution
    "above_PEL": "#d62728",        # red - toxic
    "no_thresholds": "#999999",    # gray - unknown
}


def enrichment_factor_depth_plot(
    ef_result: EnrichmentFactorResult,
    df: pd.DataFrame,
    depth_col: str,
    site_col: Optional[str] = None,
    metals: Optional[list[str]] = None,
    n_cols: int = 3,
    height_per_row: int = 260,
) -> go.Figure:
    """Plot EF vs depth for multiple metals, with Birch (2003) threshold bands.

    Parameters
    ----------
    ef_result : EnrichmentFactorResult
        Result from compute_enrichment_factor().
    df : pd.DataFrame
        Raw data containing depth and (optionally) site columns. Must share
        index with ef_result.ef_values.
    depth_col : str
        Name of the depth column in df.
    site_col : str, optional
        If provided, lines are colored by site.
    metals : list[str], optional
        Metals to plot. Default: all metals in ef_result.
    n_cols : int
        Grid columns.

    Returns
    -------
    go.Figure
        Plotly subplot figure with one panel per metal.
    """
    if metals is None:
        metals = list(ef_result.ef_values.columns)

    n_metals = len(metals)
    n_rows = (n_metals + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=metals,
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    # Reference Birch bands (dashed horizontal lines at key EF thresholds)
    bands = [2, 3, 5, 10, 25, 50]

    site_values = df[site_col].unique() if site_col else [None]
    color_map = (
        dict(zip(site_values, get_categorical_colors(len(site_values))))
        if site_col else {None: CATEGORICAL_PALETTE[0]}
    )

    for i, metal in enumerate(metals):
        row = i // n_cols + 1
        col = i % n_cols + 1

        merged = df[[depth_col]].copy()
        if site_col:
            merged[site_col] = df[site_col]
        merged["ef"] = ef_result.ef_values[metal]
        merged = merged.dropna(subset=["ef", depth_col])

        if site_col:
            show_legend_for_this_subplot = (i == 0)
            for site in site_values:
                site_data = merged[merged[site_col] == site].sort_values(depth_col)
                if len(site_data) == 0:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=site_data["ef"],
                        y=site_data[depth_col],
                        mode="lines+markers",
                        name=str(site),
                        line=dict(color=color_map[site]),
                        marker=dict(size=5),
                        showlegend=show_legend_for_this_subplot,
                        legendgroup=str(site),
                    ),
                    row=row, col=col,
                )
        else:
            site_data = merged.sort_values(depth_col)
            fig.add_trace(
                go.Scatter(
                    x=site_data["ef"],
                    y=site_data[depth_col],
                    mode="lines+markers",
                    line=dict(color=CATEGORICAL_PALETTE[0]),
                    showlegend=False,
                ),
                row=row, col=col,
            )

        # Birch threshold bands (vertical dashed lines)
        for b in bands:
            fig.add_vline(
                x=b,
                line=dict(color="gray", dash="dot", width=0.8),
                row=row, col=col,
            )

        fig.update_xaxes(title_text="EF", type="log", row=row, col=col)
        fig.update_yaxes(title_text=depth_col, autorange="reversed", row=row, col=col)

    fig.update_layout(
        height=height_per_row * n_rows,
        title_text=f"Enrichment Factor vs {depth_col} (reference: {ef_result.reference_element})",
    )
    apply_default_layout(fig)
    return fig


def contamination_classification_plot(
    classifications: pd.DataFrame,
    kind: str = "tel_pel",
    title: Optional[str] = None,
) -> go.Figure:
    """Stacked bar chart of classification counts per metal.

    Parameters
    ----------
    classifications : pd.DataFrame
        Samples × metals, with string enum values (from classify_tel_pel or
        classify_ef_birch).
    kind : str
        Either "tel_pel" or "ef" (controls the color palette).
    title : str, optional

    Returns
    -------
    go.Figure
    """
    if kind == "tel_pel":
        color_map = TEL_PEL_COLORS
        default_title = "TEL/PEL Classification by Metal"
        order = ["below_TEL", "TEL_to_PEL", "above_PEL", "no_thresholds"]
    elif kind == "ef":
        color_map = EF_BAND_COLORS
        default_title = "Enrichment Factor Classification by Metal (Birch 2003)"
        order = [
            "no_enrichment", "minor", "moderate", "moderately_severe",
            "severe", "very_severe", "extremely_severe",
        ]
    else:
        raise ValueError(f"Unknown kind '{kind}'. Use 'tel_pel' or 'ef'.")

    counts = pd.DataFrame(index=classifications.columns, columns=order, dtype=int).fillna(0)
    for metal in classifications.columns:
        vc = classifications[metal].value_counts(dropna=True)
        for label in order:
            if label in vc.index:
                counts.at[metal, label] = int(vc[label])

    fig = go.Figure()
    for label in order:
        if counts[label].sum() == 0:
            continue
        fig.add_trace(
            go.Bar(
                name=label.replace("_", " "),
                x=counts.index,
                y=counts[label],
                marker_color=color_map.get(label, "#999999"),
            )
        )

    fig.update_layout(
        barmode="stack",
        title=title or default_title,
        xaxis_title="Metal",
        yaxis_title="Number of samples",
    )
    apply_default_layout(fig)
    return fig
```

---

## FILE 8 — Extend `aeda/viz/profiles.py`

Agregar al final del archivo existente (no reemplazar — añadir):

```python
def depth_profile_with_thresholds(
    df: pd.DataFrame,
    metal: str,
    depth_col: str,
    site_col: Optional[str] = None,
    tel: Optional[float] = None,
    pel: Optional[float] = None,
    use_noaa_defaults: bool = True,
    log_scale: bool = True,
) -> go.Figure:
    """Depth profile with TEL/PEL horizontal reference lines.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data including metal concentration and depth.
    metal : str
        Name of the metal column.
    depth_col : str
        Name of the depth column.
    site_col : str, optional
        If provided, one line per site.
    tel, pel : float, optional
        Custom thresholds in mg/kg. If both None and use_noaa_defaults=True,
        loads NOAA marine-sediment thresholds for the metal.
    use_noaa_defaults : bool
        Whether to auto-load thresholds from Buchman (2008) table.
    log_scale : bool
        Use log scale for concentration axis (common for environmental data).

    Returns
    -------
    go.Figure
    """
    from aeda.interpretation.thresholds import get_thresholds

    if use_noaa_defaults and tel is None and pel is None:
        try:
            t = get_thresholds(metal)
            tel = t.tel
            pel = t.pel
        except KeyError:
            pass

    fig = go.Figure()

    if site_col and site_col in df.columns:
        colors = get_categorical_colors(df[site_col].nunique())
        for i, (site, sub) in enumerate(df.groupby(site_col)):
            sub = sub.sort_values(depth_col).dropna(subset=[metal, depth_col])
            fig.add_trace(
                go.Scatter(
                    x=sub[metal],
                    y=sub[depth_col],
                    mode="lines+markers",
                    name=str(site),
                    line=dict(color=colors[i % len(colors)]),
                    marker=dict(size=5),
                )
            )
    else:
        sub = df.sort_values(depth_col).dropna(subset=[metal, depth_col])
        fig.add_trace(
            go.Scatter(
                x=sub[metal],
                y=sub[depth_col],
                mode="lines+markers",
                line=dict(color=CATEGORICAL_PALETTE[0]),
                showlegend=False,
            )
        )

    # Add vertical threshold lines for TEL and PEL
    if tel is not None:
        fig.add_vline(
            x=tel, line=dict(color="#ff7f0e", dash="dash", width=1.5),
            annotation_text=f"TEL={tel}", annotation_position="top",
        )
    if pel is not None:
        fig.add_vline(
            x=pel, line=dict(color="#d62728", dash="dash", width=1.5),
            annotation_text=f"PEL={pel}", annotation_position="top",
        )

    fig.update_layout(
        title=f"{metal} — depth profile with TEL/PEL",
        xaxis_title=f"{metal} (mg/kg)",
        yaxis_title=depth_col,
        xaxis=dict(type="log" if log_scale else "linear"),
        yaxis=dict(autorange="reversed"),
    )
    apply_default_layout(fig)
    return fig
```

Actualizar `aeda/viz/__init__.py` para exportar las nuevas funciones:

```python
# Agregar a los imports existentes:
from aeda.viz.profiles import depth_profile, depth_profile_grid, depth_profile_with_thresholds
from aeda.viz.interpretation import (
    enrichment_factor_depth_plot,
    contamination_classification_plot,
)

# Agregar a __all__:
"depth_profile_with_thresholds",
"enrichment_factor_depth_plot",
"contamination_classification_plot",
```

---

## FILE 9 — `tests/test_interpretation.py` (NEW)

```python
"""Regression tests for the environmental interpretation module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aeda.interpretation import (
    handle_lod_values,
    compute_enrichment_factor,
    classify_tel_pel,
    classify_ef_birch,
    build_interpretation_report,
    get_thresholds,
    TEL_PEL_MARINE_SEDIMENT,
    TELPELClass,
    EFClass,
)


def _synthetic_sediment_data(n_per_site: int = 20, n_sites: int = 3):
    """Build synthetic sediment data: 3 sites, depth gradient, Pb contamination."""
    rng = np.random.default_rng(42)
    rows = []
    for site_idx in range(n_sites):
        site_name = f"Site_{site_idx}"
        for depth in np.linspace(0, 50, n_per_site):
            # Background-to-surface contamination gradient
            pb_surface_factor = 1 + 4 * (1 - depth / 50)  # more Pb near surface
            rows.append({
                "Site_Name": site_name,
                "Depth": depth,
                "Al": 7.0 + rng.normal(0, 0.5),        # %
                "Fe": 4.5 + rng.normal(0, 0.3),        # %
                "Pb": 30 * pb_surface_factor + rng.normal(0, 3),   # mg/kg
                "Cr": 55 + rng.normal(0, 5),                        # mg/kg
                "Cu": 15 + rng.normal(0, 2),                        # mg/kg
                "Zn": 100 * pb_surface_factor ** 0.5 + rng.normal(0, 5),
            })
    return pd.DataFrame(rows)


# ---- LOD handling ----

def test_lod_imputation_normal_uses_half_lod():
    rng = np.random.default_rng(0)
    # Normal distribution, clearly normal
    df = pd.DataFrame({"X": rng.normal(10, 1, size=200)})
    df.loc[:10, "X"] = np.nan  # mark some as below LOD
    out, log = handle_lod_values(df, lod_values={"X": 5.0})
    # For Normal data, the imputed value should be LOD/2 = 2.5
    assert log.variable_decisions["X"]["method"].startswith("LOD/2")
    # All previously NaN should now be 2.5
    imputed_vals = out.loc[:10, "X"].unique()
    assert len(imputed_vals) == 1
    assert abs(imputed_vals[0] - 2.5) < 1e-9


def test_lod_imputation_lognormal_uses_sqrt_lod():
    rng = np.random.default_rng(0)
    # Strongly skewed (log-normal) data → not Normal
    df = pd.DataFrame({"X": rng.lognormal(0, 1.5, size=200)})
    df.loc[:10, "X"] = np.nan
    out, log = handle_lod_values(df, lod_values={"X": 4.0})
    assert "sqrt" in log.variable_decisions["X"]["method"]
    expected = 4.0 / np.sqrt(2.0)
    imputed_vals = out.loc[:10, "X"].unique()
    assert abs(imputed_vals[0] - expected) < 1e-9


def test_lod_raises_on_negative_lod():
    df = pd.DataFrame({"X": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="must be positive"):
        handle_lod_values(df, lod_values={"X": -1.0})


# ---- TEL/PEL classification ----

def test_tel_pel_classification_bands():
    # Pb: TEL=30.24, PEL=112
    concentrations = pd.Series([10, 50, 200, np.nan], name="Pb")
    result = classify_tel_pel(concentrations, "Pb")
    assert result.iloc[0] == TELPELClass.BELOW_TEL.value
    assert result.iloc[1] == TELPELClass.BETWEEN_TEL_PEL.value
    assert result.iloc[2] == TELPELClass.ABOVE_PEL.value
    assert pd.isna(result.iloc[3])


def test_tel_pel_unknown_metal():
    result = classify_tel_pel(pd.Series([10, 20]), "Unobtanium")
    assert all(v == TELPELClass.NO_THRESHOLDS.value for v in result)


def test_tel_pel_has_required_metals():
    required = {"As", "Cd", "Cr", "Cu", "Hg", "Ni", "Pb", "Zn"}
    assert required.issubset(TEL_PEL_MARINE_SEDIMENT.keys())


# ---- EF classification ----

def test_ef_classification_birch_bands():
    ef = pd.Series([1.5, 2.5, 4.0, 8.0, 20.0, 40.0, 100.0])
    result = classify_ef_birch(ef)
    assert result.iloc[0] == EFClass.NO_ENRICHMENT.value
    assert result.iloc[1] == EFClass.MINOR.value
    assert result.iloc[2] == EFClass.MODERATE.value
    assert result.iloc[3] == EFClass.MODERATELY_SEVERE.value
    assert result.iloc[4] == EFClass.SEVERE.value
    assert result.iloc[5] == EFClass.VERY_SEVERE.value
    assert result.iloc[6] == EFClass.EXTREMELY_SEVERE.value


# ---- EF computation ----

def test_ef_uses_deepest_section_as_baseline():
    df = _synthetic_sediment_data()
    result = compute_enrichment_factor(
        df,
        metals=["Pb", "Cr", "Cu", "Zn"],
        reference_element="Al",
        site_col="Site_Name",
        depth_col="Depth",
        baseline_strategy="deepest",
    )
    # Surface Pb should be enriched (gradient: surface_factor=5, deep_factor=1)
    surface = df[df["Depth"] < 3].index
    deep = df[df["Depth"] > 45].index

    # EF at the deepest (baseline) sample should be ~1 by construction
    for site_idx in range(3):
        site_df = df[df["Site_Name"] == f"Site_{site_idx}"]
        deepest_row = site_df.loc[site_df["Depth"].idxmax()]
        ef_at_baseline = result.ef_values.loc[deepest_row.name, "Pb"]
        assert abs(ef_at_baseline - 1.0) < 1e-9

    # Surface EF should be greater than deep EF for Pb
    surface_pb_ef_mean = result.ef_values.loc[surface, "Pb"].mean()
    deep_pb_ef_mean = result.ef_values.loc[deep, "Pb"].mean()
    assert surface_pb_ef_mean > deep_pb_ef_mean
    assert surface_pb_ef_mean > 2.5  # meaningful enrichment


def test_ef_rejects_reference_in_metals():
    df = _synthetic_sediment_data()
    with pytest.raises(ValueError, match="cannot be in the metals list"):
        compute_enrichment_factor(
            df, metals=["Pb", "Al"],
            reference_element="Al",
            site_col="Site_Name", depth_col="Depth",
        )


def test_ef_rejects_missing_reference():
    df = _synthetic_sediment_data()
    with pytest.raises(ValueError, match="not found"):
        compute_enrichment_factor(
            df, metals=["Pb"],
            reference_element="Scandium",
            site_col="Site_Name", depth_col="Depth",
        )


def test_ef_strategy_deepest_requires_depth_col():
    df = _synthetic_sediment_data()
    with pytest.raises(ValueError, match="deepest.*depth_col"):
        compute_enrichment_factor(
            df, metals=["Pb"],
            reference_element="Al",
            baseline_strategy="deepest",
            depth_col=None,
        )


def test_ef_custom_baseline():
    df = _synthetic_sediment_data()
    custom = {
        "__global__": {"Al": 7.0, "Pb": 30.0}
    }
    result = compute_enrichment_factor(
        df, metals=["Pb"],
        reference_element="Al",
        baseline_strategy="user",
        custom_baseline=custom,
    )
    # EF = (Pb/Al)_sample / (Pb/Al)_baseline; with baseline=(30/7), middle-depth
    # Pb ~ 3x baseline, so EF ~ 3
    surface_ef = result.ef_values.loc[df[df["Depth"] < 3].index, "Pb"].mean()
    assert 3 < surface_ef < 7


# ---- Integration: full report ----

def test_interpretation_report_end_to_end():
    df = _synthetic_sediment_data()
    report = build_interpretation_report(
        df,
        metals=["Pb", "Cr", "Cu", "Zn"],
        reference_element="Al",
        site_col="Site_Name",
        depth_col="Depth",
        baseline_strategy="deepest",
    )
    assert report.ef_result is not None
    assert report.ef_classifications is not None
    assert "Pb" in report.tel_pel_classifications.columns
    assert len(report.summary()) > 0

    # Surface Pb should be classified above TEL in synthetic data
    surface_idx = df[df["Depth"] < 3].index
    pb_classes = report.tel_pel_classifications.loc[surface_idx, "Pb"]
    # At least some surface samples should exceed TEL given the 5x gradient
    assert (pb_classes != TELPELClass.BELOW_TEL.value).sum() > 0
```

---

## Orden de trabajo

1. Crear la carpeta `aeda/interpretation/` con los 6 archivos (FILE 1–6).
2. Crear `aeda/viz/interpretation.py` (FILE 7).
3. Extender `aeda/viz/profiles.py` con `depth_profile_with_thresholds` (FILE 8).
4. Actualizar `aeda/viz/__init__.py` para exportar las nuevas funciones.
5. Crear `tests/test_interpretation.py` (FILE 9).
6. Ejecutar `pytest tests/ -v`. Todos los tests previos (integration + smoke) más los 12 nuevos deben pasar.
7. Ejecutar manualmente contra ISOVIDA (ver script de verificación al final).
8. Commit con mensaje: `feat: add environmental interpretation module (EF, TEL/PEL, LOD handling)`.

---

## Script de verificación manual con ISOVIDA

Después de aplicar todo, ejecutar para comprobar que el módulo funciona con datos reales:

```python
from aeda.pipeline.runner import AEDAPipeline
from aeda.interpretation import build_interpretation_report
from aeda.viz import (
    enrichment_factor_depth_plot,
    contamination_classification_plot,
    depth_profile_with_thresholds,
    save_figure,
)
import os

EXCLUDE = ["No", "Code", "Site_Name", "Pret_Code", "Código_muestra",
    "Sitio_muestreo", "Fecha_muestreo", "Core", "Latitud", "Longitud", "Profundidad"]

pipeline = AEDAPipeline(impute_strategy="median")
r = pipeline.run("data/BD_ISOVIDA_MANGLARES2023_rectificadaYBA_230326.xlsx",
                 exclude_cols=EXCLUDE, sheet_name="DATA")

METALS = ["Pb", "Cr", "Cu", "Zn", "Ni", "As"]

report = build_interpretation_report(
    r.raw_data,
    metals=METALS,
    reference_element="Al",
    site_col="Sitio_muestreo",
    depth_col="Profundidad",
    baseline_strategy="deepest",
)

print(report.summary())

os.makedirs("/tmp/interpretation_check", exist_ok=True)

# Visualizations
save_figure(
    enrichment_factor_depth_plot(
        report.ef_result, r.raw_data,
        depth_col="Profundidad", site_col="Sitio_muestreo",
    ),
    "/tmp/interpretation_check/ef_by_depth.html",
)

save_figure(
    contamination_classification_plot(report.tel_pel_classifications, kind="tel_pel"),
    "/tmp/interpretation_check/tel_pel_bar.html",
)

save_figure(
    contamination_classification_plot(report.ef_classifications, kind="ef"),
    "/tmp/interpretation_check/ef_bar.html",
)

save_figure(
    depth_profile_with_thresholds(
        r.raw_data, metal="Pb",
        depth_col="Profundidad", site_col="Sitio_muestreo",
    ),
    "/tmp/interpretation_check/pb_profile_thresholds.html",
)

print("\nAll plots saved to /tmp/interpretation_check/")
```

Los resultados esperados (si el código está correcto):
- El summary debe mostrar clasificaciones por metal tanto para TEL/PEL como para EF.
- Para ISOVIDA (mayormente manglares de Cienfuegos), muchos sedimentos deberían clasificar como "below_TEL" o "TEL_to_PEL" — es un entorno moderadamente contaminado, no hotspot extremo.
- Los plots deben generarse sin errores.
