"""
aeda.engine.auto_selector
Automatic analysis-method selector.

This is the core module of the thesis project: given an environmental dataset,
it profiles data characteristics and recommends an optimal combination
of preprocessing and analysis techniques.

v0.2 refinements:
- Mixed-unit detection (%, mg/kg, ppm) and normalization recommendations
- Geochemical heuristics: identifies major vs trace elements
- Compositional subgroup detection (FRX vs granulometry)
- Hierarchical structure analysis (site -> core -> depth)
- Confidence scoring for each recommendation
- Multi-scale analysis support (global, by site, by profile)
- Depth-gradient detection
- Rules for small vs large datasets
"""

import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass, field
from scipy import stats
from enum import Enum


# ============================================================
#  GEOCHEMICAL CONSTANTS
# ============================================================

# Geochemical element classification (aligned with NOAA Buchman 2008 thresholds
# and reviewed with the project's scientific tutor).
# These constants are also exposed for external alignment with the
# interpretation module thresholds table.
MAJOR_ELEMENTS = {"Na", "Mg", "Al", "Si", "K", "Ca", "Fe", "Ti", "Mn", "P"}
TRACE_ELEMENTS = {
    "V", "Cr", "Co", "Ni", "Cu", "Zn", "Ga", "As", "Br", "Rb",
    "Sr", "Y", "Zr", "Nb", "Mo", "Ba", "Pb", "Sc", "S", "Cl",
    "Cd", "Hg", "Ag", "Sb", "Se",
}
# Regulated heavy metals with TEL/PEL thresholds in NOAA Buchman (2008).
# This list MUST stay in sync with aeda.interpretation.thresholds.TEL_PEL_MARINE_SEDIMENT.
HEAVY_METALS = {"As", "Cd", "Cr", "Cu", "Hg", "Ni", "Pb", "Zn", "Ag", "Sb"}
SEDIMENT_INDICATORS = {"PPI550", "PPI950", "HC"}
GRANULOMETRY_PATTERNS = [
    ("< 2", "2 < G < 63", "> 63"),
    ("clay", "silt", "sand"),
    ("arcilla", "limo", "arena"),
]


# Sanity check: HEAVY_METALS must stay aligned with the NOAA thresholds table
# in the interpretation module. Drift between these two lists creates the bug
# where the brain "detects" metals that the interpretation module cannot classify.
def _check_heavy_metals_alignment() -> None:
    """Verify HEAVY_METALS matches NOAA TEL/PEL thresholds at import time."""
    try:
        from aeda.interpretation.thresholds import TEL_PEL_MARINE_SEDIMENT
    except ImportError:  # pragma: no cover
        # interpretation module not available at import time — skip silently.
        return
    noaa_metals = set(TEL_PEL_MARINE_SEDIMENT.keys())
    if HEAVY_METALS != noaa_metals:
        # Use a lazy assertion so the import does not fail in production but
        # surfaces the issue clearly during development.
        import warnings
        warnings.warn(
            f"HEAVY_METALS in auto_selector ({sorted(HEAVY_METALS)}) does not match "
            f"NOAA TEL/PEL table ({sorted(noaa_metals)}). "
            f"This will cause inconsistencies between the brain's recommendations "
            f"and the interpretation module.",
            UserWarning,
        )


_check_heavy_metals_alignment()


class Confidence(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ============================================================
#  DATACLASSES
# ============================================================

@dataclass
class DataProfile:
    """Comprehensive dataset profile used to drive method selection."""
    n_samples: int = 0
    n_features: int = 0
    ratio_samples_features: float = 0.0
    is_small_dataset: bool = False
    is_wide_dataset: bool = False

    skewed_features: list = field(default_factory=list)
    normal_features: list = field(default_factory=list)
    pct_skewed: float = 0.0
    heavy_tail_features: list = field(default_factory=list)
    bimodal_features: list = field(default_factory=list)

    high_correlation_pairs: int = 0
    mean_abs_correlation: float = 0.0
    is_multicollinear: bool = False
    correlation_blocks: list = field(default_factory=list)

    pct_missing: float = 0.0
    missing_pattern: str = ""
    missing_groups: list = field(default_factory=list)

    is_compositional: bool = False
    compositional_subgroups: list = field(default_factory=list)

    has_major_elements: bool = False
    has_trace_elements: bool = False
    has_heavy_metals: bool = False
    has_granulometry: bool = False
    has_sediment_indicators: bool = False
    major_element_cols: list = field(default_factory=list)
    trace_element_cols: list = field(default_factory=list)
    heavy_metal_cols: list = field(default_factory=list)
    granulometry_cols: list = field(default_factory=list)
    sediment_indicator_cols: list = field(default_factory=list)
    mixed_units_detected: bool = False

    has_coordinates: bool = False
    has_depth: bool = False
    has_sites: bool = False
    n_sites: int = 0
    has_depth_gradient: bool = False
    depth_range: tuple = (0, 0)
    samples_per_site: dict = field(default_factory=dict)
    unbalanced_sites: bool = False

    pct_outliers_iqr: float = 0.0
    outlier_columns: list = field(default_factory=list)
    pct_outliers_zscore: float = 0.0

    low_variance_features: list = field(default_factory=list)
    high_variance_features: list = field(default_factory=list)
    effective_dimensionality: int = 0


@dataclass
class MethodRecommendation:
    """Recommendation with rationale and confidence score."""
    category: str
    method: str
    params: dict = field(default_factory=dict)
    reason: str = ""
    priority: int = 1
    confidence: Confidence = Confidence.MEDIUM
    evidence: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "method": self.method,
            "params": self.params,
            "reason": self.reason,
            "priority": self.priority,
            "confidence": self.confidence.value,
            "evidence": self.evidence,
        }


@dataclass
class AnalysisScale:
    """Defines an analysis scale (global, by site, by profile)."""
    name: str
    description: str
    filter_col: Optional[str] = None
    filter_values: Optional[list] = None
    recommended: bool = True
    reason: str = ""


@dataclass
class AnalysisPlan:
    """Full analysis plan generated by the auto-selector."""
    profile: DataProfile
    recommendations: list[MethodRecommendation] = field(default_factory=list)
    analysis_scales: list[AnalysisScale] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def get_by_category(self, category: str) -> list[MethodRecommendation]:
        return [r for r in self.recommendations if r.category == category]

    def get_primary(self, category: str) -> Optional[MethodRecommendation]:
        recs = [r for r in self.recommendations if r.category == category and r.priority == 1]
        return recs[0] if recs else None

    def summary(self) -> str:
        p = self.profile
        lines = [
            "=" * 70,
            "  ANALYSIS PLAN GENERATED BY AEDA-AI AUTO-SELECTOR v0.2",
            "=" * 70,
            "",
            "DATASET PROFILE",
            "-" * 40,
            f"  Dimensions: {p.n_samples} samples × {p.n_features} variables",
            f"  Sample/feature ratio: {p.ratio_samples_features:.1f}",
            f"  Effective dimensionality (PCA 90%): ~{p.effective_dimensionality} components",
        ]

        if p.is_small_dataset:
            lines.append("  ⚠ Small dataset (<50 samples)")
        if p.is_wide_dataset:
            lines.append("  ⚠ Wide dataset (more variables than samples)")

        lines.extend([
            "",
            "  Distribution:",
            f"    Skewed variables: {p.pct_skewed:.0f}%",
            f"    Bimodal variables: {len(p.bimodal_features)}",
            f"    Heavy-tail variables: {len(p.heavy_tail_features)}",
            "",
            "  Structure:",
            f"    Missing data: {p.pct_missing:.1f}% ({p.missing_pattern})",
            f"    Multicollinearity: {'Yes' if p.is_multicollinear else 'No'}"
            f" ({p.high_correlation_pairs} pairs with |r|>0.7)",
            f"    Compositional: {'Yes' if p.is_compositional else 'No'}",
            f"    Outliers (IQR): {p.pct_outliers_iqr:.0f}% of samples",
        ])

        if p.has_major_elements or p.has_trace_elements:
            lines.extend(["", "  Detected geochemistry:"])
            if p.major_element_cols:
                lines.append(f"    Major elements ({len(p.major_element_cols)}): "
                             f"{', '.join(p.major_element_cols[:8])}")
            if p.trace_element_cols:
                lines.append(f"    Trace elements ({len(p.trace_element_cols)}): "
                             f"{', '.join(p.trace_element_cols[:8])}")
            if p.heavy_metal_cols:
                lines.append(f"    Heavy metals: {', '.join(p.heavy_metal_cols)}")
            if p.has_granulometry:
                lines.append(f"    Granulometry: {', '.join(p.granulometry_cols)}")
            if p.mixed_units_detected:
                lines.append("    ⚠ Mixed units detected (% and mg/kg)")

        if p.has_sites or p.has_depth:
            lines.extend(["", "  Spatial/hierarchical structure:"])
            if p.has_sites:
                lines.append(f"    Sites: {p.n_sites}")
                if p.unbalanced_sites:
                    lines.append("    ⚠ Unbalanced samples across sites")
            if p.has_depth:
                lines.append(f"    Depth: {p.depth_range[0]}-{p.depth_range[1]} cm")
                if p.has_depth_gradient:
                    lines.append("    Depth gradient detected in variables")

        if self.warnings:
            lines.extend(["", "WARNINGS", "-" * 40])
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")

        if self.analysis_scales:
            lines.extend(["", "RECOMMENDED ANALYSIS SCALES", "-" * 40])
            for scale in self.analysis_scales:
                marker = "★" if scale.recommended else "○"
                lines.append(f"  {marker} {scale.name}: {scale.description}")
                if scale.reason:
                    lines.append(f"    → {scale.reason}")

        categories = [
            ("preprocessing", "PREPROCESSING"),
            ("dimensionality", "DIMENSIONALITY REDUCTION"),
            ("clustering", "CLUSTERING"),
            ("anomaly", "ANOMALY DETECTION"),
            ("correlation", "CORRELATIONS"),
            ("feature_analysis", "FEATURE ANALYSIS"),
        ]

        lines.extend(["", "METHOD RECOMMENDATIONS", "-" * 40])
        for cat_key, cat_name in categories:
            recs = self.get_by_category(cat_key)
            if recs:
                lines.append(f"\n  [{cat_name}]")
                for r in recs:
                    prio = "★" if r.priority == 1 else "○"
                    conf = f"[{r.confidence.value}]"
                    lines.append(f"    {prio} {r.method} {conf}")
                    lines.append(f"      {r.reason}")
                    if r.evidence:
                        for ev in r.evidence:
                            lines.append(f"      · {ev}")
                    if r.params:
                        lines.append(f"      Params: {r.params}")

        lines.extend(["", "=" * 70])
        return "\n".join(lines)


# ============================================================
#  PROFILING FUNCTIONS
# ============================================================

def _profile_distributions(df: pd.DataFrame) -> dict:
    skewed, normal, heavy_tail, bimodal = [], [], [], []
    for col in df.columns:
        s = df[col].dropna()
        if len(s) < 20:
            continue
        skewness = s.skew()
        kurtosis = s.kurtosis()
        sample = s.sample(min(len(s), 500), random_state=42)
        _, p_shapiro = stats.shapiro(sample)
        if abs(skewness) > 1.5 or p_shapiro < 0.01:
            skewed.append(col)
        else:
            normal.append(col)
        if kurtosis > 5.0:
            heavy_tail.append(col)
        if kurtosis < -1.0:
            bimodal.append(col)
    return {"skewed": skewed, "normal": normal, "heavy_tail": heavy_tail, "bimodal": bimodal}


def _detect_compositional_subgroups(df: pd.DataFrame) -> list[dict]:
    subgroups = []
    for patterns in GRANULOMETRY_PATTERNS:
        matched = []
        for p in patterns:
            for col in df.columns:
                if p.lower() in col.lower() and "u_" not in col.lower():
                    matched.append(col)
                    break
        if len(matched) == 3:
            valid = df[matched].dropna()
            if len(valid) > 0:
                sums = valid.sum(axis=1)
                cv = sums.std() / sums.mean() if sums.mean() > 0 else 999
                if cv < 0.10:
                    subgroups.append({"name": "Granulometry", "columns": matched,
                                      "mean_sum": float(sums.mean()), "cv": float(cv)})
            break

    major_in_data = [c for c in df.columns if c in MAJOR_ELEMENTS]
    if len(major_in_data) >= 4:
        valid = df[major_in_data].dropna()
        if len(valid) > 0:
            sums = valid.sum(axis=1)
            cv = sums.std() / sums.mean() if sums.mean() > 0 else 999
            if cv < 0.25:
                subgroups.append({"name": "FRX major elements", "columns": major_in_data,
                                  "mean_sum": float(sums.mean()), "cv": float(cv)})
    return subgroups


def _detect_geochemistry(df: pd.DataFrame) -> dict:
    cols = set(df.columns)
    major = sorted(cols & MAJOR_ELEMENTS)
    trace = sorted(cols & TRACE_ELEMENTS)
    heavy = sorted(cols & HEAVY_METALS)
    sediment = sorted(cols & SEDIMENT_INDICATORS)

    gran_cols = []
    for patterns in GRANULOMETRY_PATTERNS:
        matched = []
        for p in patterns:
            for col in df.columns:
                if p.lower() in col.lower() and "u_" not in col.lower():
                    matched.append(col)
                    break
        if len(matched) == 3:
            gran_cols = matched
            break

    mixed_units = False
    if major and trace:
        major_medians = df[major].median()
        trace_medians = df[trace].median()
        if major_medians.median() < 20 and trace_medians.median() > 20:
            mixed_units = True

    return {"major": major, "trace": trace, "heavy": heavy,
            "granulometry": gran_cols, "sediment": sediment, "mixed_units": mixed_units}


def _detect_correlation_blocks(corr_matrix: pd.DataFrame, threshold: float = 0.7) -> list[list[str]]:
    cols = corr_matrix.columns.tolist()
    visited = set()
    blocks = []
    for i, col in enumerate(cols):
        if col in visited:
            continue
        block = [col]
        visited.add(col)
        for j in range(i + 1, len(cols)):
            other = cols[j]
            if other in visited:
                continue
            if abs(corr_matrix.loc[col, other]) >= threshold:
                all_corr = all(abs(corr_matrix.loc[b, other]) >= threshold * 0.85 for b in block)
                if all_corr:
                    block.append(other)
                    visited.add(other)
        if len(block) >= 2:
            blocks.append(block)
    return blocks


def _detect_depth_gradient(df: pd.DataFrame, depth_col: str, measurement_cols: list[str],
                           threshold_pct: float = 0.3) -> bool:
    if depth_col not in df.columns:
        return False
    n_significant = 0
    for col in measurement_cols:
        if col not in df.columns:
            continue
        valid = df[[depth_col, col]].dropna()
        if len(valid) < 10:
            continue
        r, p = stats.spearmanr(valid[depth_col], valid[col])
        if p < 0.05 and abs(r) > 0.3:
            n_significant += 1
    return n_significant / max(len(measurement_cols), 1) > threshold_pct


def _estimate_effective_dimensionality(df: pd.DataFrame, threshold: float = 0.90) -> int:
    from sklearn.decomposition import PCA
    clean = df.dropna()
    if len(clean) < 2 or clean.shape[1] < 2:
        return clean.shape[1]
    centered = (clean - clean.mean()) / clean.std().replace(0, 1)
    pca = PCA()
    pca.fit(centered)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    return max(2, int(np.argmax(cumvar >= threshold) + 1))


def _analyze_missing_groups(df: pd.DataFrame) -> list[dict]:
    null_cols = [c for c in df.columns if df[c].isnull().any()]
    if not null_cols:
        return []
    null_mask = df[null_cols].isnull()
    patterns = null_mask.drop_duplicates()
    groups = []
    for _, pattern in patterns.iterrows():
        missing_cols = [c for c in null_cols if pattern[c]]
        if missing_cols:
            n_rows = null_mask[null_mask[missing_cols].all(axis=1)].shape[0]
            groups.append({"missing_columns": missing_cols, "n_rows": n_rows,
                           "pct_rows": n_rows / len(df) * 100})
    return groups


def _detect_outlier_details(df: pd.DataFrame) -> dict:
    q1, q3 = df.quantile(0.25), df.quantile(0.75)
    iqr = q3 - q1
    outlier_mask = (df < (q1 - 1.5 * iqr)) | (df > (q3 + 1.5 * iqr))
    outlier_cols = []
    for col in df.columns:
        n_out = outlier_mask[col].sum()
        if n_out > 0:
            outlier_cols.append({"column": col, "n_outliers": int(n_out), "pct": n_out / len(df) * 100})
    outlier_cols.sort(key=lambda x: x["n_outliers"], reverse=True)
    z = np.abs((df - df.mean()) / df.std().replace(0, 1))
    pct_zscore = float((z > 3).any(axis=1).mean() * 100)
    return {"pct_iqr": float(outlier_mask.any(axis=1).mean() * 100),
            "pct_zscore": pct_zscore, "columns": outlier_cols}


# ============================================================
#  MAIN PROFILING
# ============================================================

def profile_dataset(
    df: pd.DataFrame,
    has_coordinates: bool = False,
    has_depth: bool = False,
    depth_col: Optional[str] = None,
    has_sites: bool = False,
    site_col: Optional[str] = None,
    n_sites: int = 0,
    original_df: Optional[pd.DataFrame] = None,
) -> DataProfile:
    n_samples, n_features = df.shape
    profile = DataProfile(
        n_samples=n_samples, n_features=n_features,
        ratio_samples_features=n_samples / max(n_features, 1),
        is_small_dataset=n_samples < 50,
        is_wide_dataset=n_features > n_samples,
    )

    dist = _profile_distributions(df)
    profile.skewed_features = dist["skewed"]
    profile.normal_features = dist["normal"]
    profile.heavy_tail_features = dist["heavy_tail"]
    profile.bimodal_features = dist["bimodal"]
    profile.pct_skewed = len(dist["skewed"]) / max(len(dist["skewed"]) + len(dist["normal"]), 1) * 100

    if n_features > 1:
        corr = df.corr(method="spearman").abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        profile.high_correlation_pairs = int((upper > 0.7).sum().sum())
        profile.mean_abs_correlation = float(upper.stack().mean())
        profile.is_multicollinear = profile.high_correlation_pairs > n_features * 0.3
        profile.correlation_blocks = _detect_correlation_blocks(df.corr(method="spearman"))

    profile.pct_missing = df.isnull().mean().mean() * 100
    null_cols = df.columns[df.isnull().any()]
    if len(null_cols) == 0:
        profile.missing_pattern = "none"
    else:
        n_patterns = df[null_cols].isnull().drop_duplicates().shape[0]
        profile.missing_pattern = "structured" if n_patterns <= 3 else "random"
    profile.missing_groups = _analyze_missing_groups(df)

    comp_sub = _detect_compositional_subgroups(df)
    profile.compositional_subgroups = comp_sub
    profile.is_compositional = len(comp_sub) > 0

    geo = _detect_geochemistry(df)
    profile.major_element_cols = geo["major"]
    profile.trace_element_cols = geo["trace"]
    profile.heavy_metal_cols = geo["heavy"]
    profile.granulometry_cols = geo["granulometry"]
    profile.sediment_indicator_cols = geo["sediment"]
    profile.has_major_elements = len(geo["major"]) > 0
    profile.has_trace_elements = len(geo["trace"]) > 0
    profile.has_heavy_metals = len(geo["heavy"]) > 0
    profile.has_granulometry = len(geo["granulometry"]) > 0
    profile.has_sediment_indicators = len(geo["sediment"]) > 0
    profile.mixed_units_detected = geo["mixed_units"]

    profile.has_coordinates = has_coordinates
    profile.has_depth = has_depth
    profile.has_sites = has_sites
    profile.n_sites = n_sites

    if original_df is not None and site_col and site_col in original_df.columns:
        counts = original_df[site_col].value_counts()
        profile.samples_per_site = counts.to_dict()
        cv_sites = counts.std() / counts.mean() if counts.mean() > 0 else 0
        profile.unbalanced_sites = cv_sites > 0.4

    if original_df is not None and depth_col and depth_col in original_df.columns:
        d = original_df[depth_col].dropna()
        profile.depth_range = (float(d.min()), float(d.max()))
        profile.has_depth_gradient = _detect_depth_gradient(original_df, depth_col, df.columns.tolist())

    outlier_info = _detect_outlier_details(df)
    profile.pct_outliers_iqr = outlier_info["pct_iqr"]
    profile.pct_outliers_zscore = outlier_info["pct_zscore"]
    profile.outlier_columns = outlier_info["columns"]

    cv = df.std() / df.mean().abs().replace(0, np.nan)
    profile.low_variance_features = cv[cv < 0.1].index.tolist()
    profile.high_variance_features = cv[cv > 2.0].index.tolist()

    try:
        profile.effective_dimensionality = _estimate_effective_dimensionality(df)
    except Exception:
        profile.effective_dimensionality = min(n_features, n_samples)

    return profile


# ============================================================
#  RECOMMENDATION ENGINE
# ============================================================

def _recommend_preprocessing(p: DataProfile) -> list[MethodRecommendation]:
    recs = []

    if p.is_compositional and p.compositional_subgroups:
        names = [s["name"] for s in p.compositional_subgroups]
        recs.append(MethodRecommendation(
            category="preprocessing", method="CLR Transform (by subgroup)",
            params={"apply_clr": True, "subgroups": p.compositional_subgroups},
            reason="Compositional data detected. CLR avoids spurious correlations in PCA.",
            priority=1, confidence=Confidence.HIGH,
            evidence=[f"Subgrupos: {', '.join(names)}",
                      "Without CLR, PCA and correlations produce artifacts in closed data."],
        ))

    if p.pct_skewed > 60:
        recs.append(MethodRecommendation(
            category="preprocessing", method="Log Transform",
            params={"apply_log": True, "auto_detect": True},
            reason=f"{p.pct_skewed:.0f}% of variables are skewed.",
            priority=1 if not p.is_compositional else 2,
            confidence=Confidence.HIGH if p.pct_skewed > 80 else Confidence.MEDIUM,
            evidence=[f"Skewed variables: {', '.join(p.skewed_features[:5])}{'...' if len(p.skewed_features) > 5 else ''}",
                      "Log-transform normalizes distributions and stabilizes variance."],
        ))
    elif p.pct_skewed > 30:
        recs.append(MethodRecommendation(
            category="preprocessing", method="Log Transform (selectivo)",
            params={"apply_log": True, "cols": p.skewed_features},
            reason=f"{p.pct_skewed:.0f}% of variables are skewed. Apply only to affected variables.",
            priority=2, confidence=Confidence.MEDIUM,
            evidence=[f"Features: {', '.join(p.skewed_features[:8])}"],
        ))

    if p.mixed_units_detected:
        recs.append(MethodRecommendation(
            category="preprocessing",
            method="StandardScaler (required for mixed units)",
            params={"scale_method": "standard"},
            reason="Mixed units (% and mg/kg). Scaling is mandatory.",
            priority=1, confidence=Confidence.HIGH,
            evidence=[f"Major elements in %: {', '.join(p.major_element_cols[:5])}",
                      f"Trace elements in mg/kg: {', '.join(p.trace_element_cols[:5])}",
                      "Without scaling, mg/kg variables would dominate PCA and clustering."],
        ))
    elif p.pct_outliers_iqr > 20:
        recs.append(MethodRecommendation(
            category="preprocessing", method="RobustScaler",
            params={"scale_method": "robust"},
            reason=f"{p.pct_outliers_iqr:.0f}% of samples contain outliers. RobustScaler is resilient.",
            priority=1, confidence=Confidence.HIGH,
            evidence=[f"Top outliers: " + ", ".join(f"{o['column']}({o['n_outliers']})" for o in p.outlier_columns[:5])],
        ))
    else:
        recs.append(MethodRecommendation(
            category="preprocessing", method="StandardScaler",
            params={"scale_method": "standard"},
            reason="Distribution is acceptable. Standard scaling is sufficient.",
            priority=1, confidence=Confidence.MEDIUM,
        ))

    if p.missing_pattern == "structured":
        recs.append(MethodRecommendation(
            category="preprocessing", method="Subset Analysis",
            params={"impute_strategy": "subset_analysis"},
            reason="Structured missing data (by design). Do not impute; analyze subsets.",
            priority=1, confidence=Confidence.HIGH,
            evidence=[f"Groups: {len(p.missing_groups)}"] +
                     [f"Group: {g['n_rows']} rows missing {', '.join(g['missing_columns'][:3])}"
                      for g in p.missing_groups[:3]],
        ))
    elif p.pct_missing > 10:
        recs.append(MethodRecommendation(
            category="preprocessing", method="KNN Imputation",
            params={"impute_strategy": "knn", "n_neighbors": 5},
            reason=f"{p.pct_missing:.1f}% random missingness. KNN preserves local structure.",
            priority=1, confidence=Confidence.MEDIUM,
        ))
    elif p.pct_missing > 0:
        recs.append(MethodRecommendation(
            category="preprocessing", method="Median Imputation",
            params={"impute_strategy": "median"},
            reason=f"Few missing values ({p.pct_missing:.1f}%). Median is robust and simple.",
            priority=1, confidence=Confidence.HIGH,
        ))

    if p.low_variance_features:
        recs.append(MethodRecommendation(
            category="preprocessing", method="Drop Constant Variables",
            params={"drop_low_variance": True, "columns": p.low_variance_features},
            reason=f"{len(p.low_variance_features)} variables with near-zero variance.",
            priority=1, confidence=Confidence.HIGH,
            evidence=[f"Features: {', '.join(p.low_variance_features)}"],
        ))

    return recs


def _recommend_dimensionality(p: DataProfile) -> list[MethodRecommendation]:
    recs = []

    if p.is_multicollinear or p.n_features > 10:
        evidence = []
        if p.is_multicollinear:
            evidence.append(f"{p.high_correlation_pairs} pairs with |r|>0.7")
        if p.correlation_blocks:
            for block in p.correlation_blocks[:3]:
                evidence.append(f"Correlated block: {', '.join(block[:4])}")
        evidence.append(f"~{p.effective_dimensionality} components for 90% explained variance")

        recs.append(MethodRecommendation(
            category="dimensionality", method="PCA",
            params={"method": "pca", "variance_threshold": 0.85},
            reason="Multicollinearity and high dimensionality. PCA reduces redundancy.",
            priority=1, confidence=Confidence.HIGH, evidence=evidence,
        ))

    if not p.is_small_dataset and p.n_features > 5:
        recs.append(MethodRecommendation(
            category="dimensionality", method="UMAP",
            params={"method": "umap", "n_components": 2, "n_neighbors": 15},
            reason="For 2D visualization of nonlinear structure.",
            priority=2, confidence=Confidence.MEDIUM,
            evidence=[f"n={p.n_samples} is sufficient for nonlinear embedding.",
                      "UMAP typically preserves global structure better than t-SNE."],
        ))
        recs.append(MethodRecommendation(
            category="dimensionality", method="t-SNE",
            params={"method": "tsne", "n_components": 2, "perplexity": 30},
            reason="Alternative to UMAP, often better for separating local clusters.",
            priority=3, confidence=Confidence.MEDIUM,
        ))
    elif p.is_small_dataset:
        recs.append(MethodRecommendation(
            category="dimensionality", method="PCA (only)",
            params={"method": "pca", "n_components": 2},
            reason="Small dataset. UMAP/t-SNE can be unstable with few samples.",
            priority=1 if not p.is_multicollinear else 2, confidence=Confidence.HIGH,
            evidence=[f"Only {p.n_samples} samples. Nonlinear methods usually need >50."],
        ))

    return recs


def _recommend_clustering(p: DataProfile) -> list[MethodRecommendation]:
    recs = []

    if p.has_sites and p.n_sites >= 2:
        recs.append(MethodRecommendation(
            category="clustering",
            method=f"K-Means (K={p.n_sites}, geographic validation)",
            params={"method": "kmeans", "n_clusters": p.n_sites},
            reason=f"Check whether {p.n_sites} chemical clusters match known sites.",
            priority=1, confidence=Confidence.HIGH,
            evidence=["If they match: contamination likely drives spatial grouping.",
                      "If not: other factors dominate variability.",
                      f"Sites: {', '.join(list(p.samples_per_site.keys())[:5]) if p.samples_per_site else 'N/A'}"],
        ))

    recs.append(MethodRecommendation(
        category="clustering", method="K-Means (automatic K, silhouette)",
        params={"method": "kmeans", "n_clusters": None, "k_range": (2, 10)},
        reason="Search for optimal K without strong prior assumptions.",
        priority=2 if p.has_sites else 1, confidence=Confidence.MEDIUM,
        evidence=["Evaluates silhouette for K=2..10 and selects the best."],
    ))

    recs.append(MethodRecommendation(
        category="clustering", method="Hierarchical Clustering (Ward)",
        params={"method": "hierarchical", "linkage": "ward"},
        reason="Produces a dendrogram useful for visualizing sample/site relationships.",
        priority=2, confidence=Confidence.MEDIUM,
        evidence=["The dendrogram is a strong visual deliverable for the thesis."],
    ))

    if p.pct_outliers_iqr > 15:
        recs.append(MethodRecommendation(
            category="clustering", method="DBSCAN",
            params={"method": "dbscan", "min_samples": 5},
            reason="Significant outliers detected. DBSCAN can label them as noise.",
            priority=2, confidence=Confidence.MEDIUM,
            evidence=[f"{p.pct_outliers_iqr:.0f}% of samples have IQR outliers.",
                      "DBSCAN does not require specifying K a priori."],
        ))

    return recs


def _recommend_anomaly(p: DataProfile) -> list[MethodRecommendation]:
    recs = []
    contamination = min(max(p.pct_outliers_zscore / 100, 0.02), 0.15) if p.pct_outliers_zscore > 0 else 0.05

    recs.append(MethodRecommendation(
        category="anomaly", method="Isolation Forest",
        params={"method": "isolation_forest", "contamination": round(contamination, 3)},
        reason="Primary method. Scales well with high dimensionality.",
        priority=1, confidence=Confidence.HIGH,
        evidence=[f"Estimated contamination: {contamination:.1%} (based on z-score outliers)."],
    ))

    recs.append(MethodRecommendation(
        category="anomaly", method="LOF (Local Outlier Factor)",
        params={"method": "lof", "n_neighbors": 20, "contamination": round(contamination, 3)},
        reason="Local-density complement. Detects contextual anomalies.",
        priority=2, confidence=Confidence.MEDIUM,
    ))

    if p.has_heavy_metals:
        recs.append(MethodRecommendation(
            category="anomaly", method="Z-score (heavy metals)",
            params={"method": "zscore", "threshold": 3.0, "cols": p.heavy_metal_cols},
            reason="Univariate heavy-metal detection to identify hotspots.",
            priority=2, confidence=Confidence.HIGH,
            evidence=[f"Heavy metals: {', '.join(p.heavy_metal_cols)}",
                      "Z-score indicates WHICH metal is anomalous in each sample."],
        ))

    return recs


def _recommend_correlation(p: DataProfile) -> list[MethodRecommendation]:
    recs = []

    recs.append(MethodRecommendation(
        category="correlation", method="Pearson + Spearman comparison",
        params={"method": "compare"},
        reason="Compare linear vs monotonic behavior to detect nonlinear relationships.",
        priority=1, confidence=Confidence.HIGH,
        evidence=["Large differences indicate nonlinearity.",
              "Both matrices are key thesis deliverables."],
    ))

    if p.has_heavy_metals and p.has_granulometry:
        recs.append(MethodRecommendation(
            category="correlation",
            method="Metal vs granulometry correlation",
            params={"method": "spearman", "subset_a": p.heavy_metal_cols, "subset_b": p.granulometry_cols},
            reason="Assess whether metal accumulation depends on grain size.",
            priority=1, confidence=Confidence.HIGH,
            evidence=["Clays and silts usually retain more metals than sands.",
                      "Positive metal-clay correlation supports adsorption mechanisms."],
        ))

    return recs


def _recommend_feature_analysis(p: DataProfile) -> list[MethodRecommendation]:
    recs = []

    recs.append(MethodRecommendation(
        category="feature_analysis", method="RF Feature Importance (by cluster)",
        params={"method": "rf_cluster_discrimination"},
        reason="Identify which variables best discriminate clusters.",
        priority=1, confidence=Confidence.HIGH,
        evidence=["Random Forest + permutation importance for robustness."],
    ))

    if p.has_heavy_metals and p.has_sites:
        recs.append(MethodRecommendation(
            category="feature_analysis", method="RF Feature Importance (by site)",
            params={"method": "rf_site_discrimination", "target": "site"},
            reason="Identify which metals most differentiate contaminated sites.",
            priority=1, confidence=Confidence.HIGH,
            evidence=["Use site as target. Most important metals = key contaminants."],
        ))

    return recs


def _recommend_analysis_scales(p: DataProfile) -> list[AnalysisScale]:
    scales = [AnalysisScale(name="Global", description="All samples.",
                            recommended=True, reason="Global overview and inter-site relationships.")]

    if p.has_sites and p.n_sites >= 2:
        scales.append(AnalysisScale(
            name="By site", description=f"Separate analysis for each of the {p.n_sites} sites.",
            filter_col="site", recommended=True,
            reason="Intra-site patterns and contamination-profile comparison."))

    if p.has_depth and p.has_depth_gradient:
        scales.append(AnalysisScale(
            name="Depth profile", description="Vertical variation by profile.",
            filter_col="depth", recommended=True,
            reason="Gradients detected. Analyze temporal trends in sediments."))

    if p.has_major_elements and p.has_trace_elements:
        scales.append(AnalysisScale(
            name="Major vs Trace", description="Separate analysis by element type.",
            recommended=True, reason="Different scales and geochemical meaning."))

    return scales


def _generate_warnings(p: DataProfile) -> list[str]:
    warnings = []
    if p.is_wide_dataset:
        warnings.append(f"More variables ({p.n_features}) than samples ({p.n_samples}). "
                        "PCA may overfit.")
    if p.ratio_samples_features < 5:
        warnings.append(f"Low sample/feature ratio ({p.ratio_samples_features:.1f}). "
                        "ML results may be unstable.")
    if p.unbalanced_sites:
        warnings.append("Unbalanced samples across sites. Clustering may be biased.")
    if p.mixed_units_detected:
        warnings.append("Variables are in different units (% and mg/kg). "
                        "Scaling is MANDATORY before multivariate analysis.")
    if p.pct_skewed > 80:
        warnings.append("Almost all variables are skewed. Consider log-transform before scaling.")
    if len(p.bimodal_features) > p.n_features * 0.3:
        warnings.append(f"{len(p.bimodal_features)} bimodal variables. "
                        "Possible indicator of two distinct populations.")
    return warnings


# ============================================================
#  MAIN INTERFACE
# ============================================================

def recommend(profile: DataProfile) -> AnalysisPlan:
    """Generate the full analysis plan."""
    recs = []
    recs.extend(_recommend_preprocessing(profile))
    recs.extend(_recommend_dimensionality(profile))
    recs.extend(_recommend_clustering(profile))
    recs.extend(_recommend_anomaly(profile))
    recs.extend(_recommend_correlation(profile))
    recs.extend(_recommend_feature_analysis(profile))
    return AnalysisPlan(
        profile=profile, recommendations=recs,
        analysis_scales=_recommend_analysis_scales(profile),
        warnings=_generate_warnings(profile),
    )


def auto_select(
    df: pd.DataFrame,
    has_coordinates: bool = False,
    has_depth: bool = False,
    depth_col: Optional[str] = None,
    has_sites: bool = False,
    site_col: Optional[str] = None,
    n_sites: int = 0,
    original_df: Optional[pd.DataFrame] = None,
) -> AnalysisPlan:
    """
    Main interface: profile the dataset and generate a full analysis plan.
    """
    profile = profile_dataset(
        df, has_coordinates=has_coordinates, has_depth=has_depth,
        depth_col=depth_col, has_sites=has_sites, site_col=site_col,
        n_sites=n_sites, original_df=original_df,
    )
    return recommend(profile)
