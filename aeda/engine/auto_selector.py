"""
aeda.engine.auto_selector
Selector automático de métodos de análisis.

Este es el módulo central de la tesis: dado un dataset ambiental,
analiza sus características y recomienda la combinación óptima
de técnicas de preprocesamiento y análisis.

Refinamientos v0.2:
- Detección de unidades mixtas (%, mg/kg, ppm) y recomendación de normalización
- Heurísticas geoquímicas: identifica elementos mayoritarios vs traza
- Detección de composicionalidad por subgrupos (FRX vs granulometría)
- Análisis de estructura jerárquica (sitio → core → profundidad)
- Scoring de confianza para cada recomendación
- Soporte para análisis multi-escala (global, por sitio, por perfil)
- Detección de gradientes de profundidad
- Reglas para datasets pequeños vs grandes
"""

import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass, field
from scipy import stats
from enum import Enum


# ============================================================
#  CONSTANTES GEOQUÍMICAS
# ============================================================

MAJOR_ELEMENTS = {"Na", "Mg", "Al", "Si", "K", "Ca", "Fe", "Ti", "Mn", "P"}
TRACE_ELEMENTS = {
    "V", "Cr", "Co", "Ni", "Cu", "Zn", "Ga", "As", "Br", "Rb",
    "Sr", "Y", "Zr", "Nb", "Mo", "Ba", "Pb", "Sc", "S", "Cl",
}
HEAVY_METALS = {"Cr", "Mn", "Co", "Ni", "Cu", "Zn", "As", "Pb", "Mo"}
SEDIMENT_INDICATORS = {"PPI550", "PPI950", "HC"}
GRANULOMETRY_PATTERNS = [
    ("< 2", "2 < G < 63", "> 63"),
    ("clay", "silt", "sand"),
    ("arcilla", "limo", "arena"),
]


class Confidence(Enum):
    HIGH = "alta"
    MEDIUM = "media"
    LOW = "baja"


# ============================================================
#  DATACLASSES
# ============================================================

@dataclass
class DataProfile:
    """Perfil exhaustivo del dataset que guía la selección de métodos."""
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
    """Recomendación con justificación y scoring de confianza."""
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
    """Define una escala de análisis (global, por sitio, por perfil)."""
    name: str
    description: str
    filter_col: Optional[str] = None
    filter_values: Optional[list] = None
    recommended: bool = True
    reason: str = ""


@dataclass
class AnalysisPlan:
    """Plan completo de análisis generado por el auto-selector."""
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
            "  PLAN DE ANÁLISIS GENERADO POR AEDA-AI AUTO-SELECTOR v0.2",
            "=" * 70,
            "",
            "PERFIL DEL DATASET",
            "-" * 40,
            f"  Dimensiones: {p.n_samples} muestras × {p.n_features} variables",
            f"  Ratio muestras/features: {p.ratio_samples_features:.1f}",
            f"  Dimensionalidad efectiva (PCA 90%): ~{p.effective_dimensionality} componentes",
        ]

        if p.is_small_dataset:
            lines.append("  ⚠ Dataset pequeño (<50 muestras)")
        if p.is_wide_dataset:
            lines.append("  ⚠ Dataset ancho (más variables que muestras)")

        lines.extend([
            "",
            "  Distribución:",
            f"    Variables sesgadas: {p.pct_skewed:.0f}%",
            f"    Variables bimodales: {len(p.bimodal_features)}",
            f"    Variables con colas pesadas: {len(p.heavy_tail_features)}",
            "",
            "  Estructura:",
            f"    Datos faltantes: {p.pct_missing:.1f}% ({p.missing_pattern})",
            f"    Multicolinealidad: {'Sí' if p.is_multicollinear else 'No'}"
            f" ({p.high_correlation_pairs} pares con |r|>0.7)",
            f"    Composicional: {'Sí' if p.is_compositional else 'No'}",
            f"    Outliers (IQR): {p.pct_outliers_iqr:.0f}% de muestras",
        ])

        if p.has_major_elements or p.has_trace_elements:
            lines.extend(["", "  Geoquímica detectada:"])
            if p.major_element_cols:
                lines.append(f"    Elem. mayoritarios ({len(p.major_element_cols)}): "
                             f"{', '.join(p.major_element_cols[:8])}")
            if p.trace_element_cols:
                lines.append(f"    Elem. traza ({len(p.trace_element_cols)}): "
                             f"{', '.join(p.trace_element_cols[:8])}")
            if p.heavy_metal_cols:
                lines.append(f"    Metales pesados: {', '.join(p.heavy_metal_cols)}")
            if p.has_granulometry:
                lines.append(f"    Granulometría: {', '.join(p.granulometry_cols)}")
            if p.mixed_units_detected:
                lines.append("    ⚠ Unidades mixtas detectadas (% y mg/kg)")

        if p.has_sites or p.has_depth:
            lines.extend(["", "  Estructura espacial/jerárquica:"])
            if p.has_sites:
                lines.append(f"    Sitios: {p.n_sites}")
                if p.unbalanced_sites:
                    lines.append("    ⚠ Muestras desbalanceadas entre sitios")
            if p.has_depth:
                lines.append(f"    Profundidad: {p.depth_range[0]}-{p.depth_range[1]} cm")
                if p.has_depth_gradient:
                    lines.append("    Gradiente de profundidad detectado en variables")

        if self.warnings:
            lines.extend(["", "ADVERTENCIAS", "-" * 40])
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")

        if self.analysis_scales:
            lines.extend(["", "ESCALAS DE ANÁLISIS RECOMENDADAS", "-" * 40])
            for scale in self.analysis_scales:
                marker = "★" if scale.recommended else "○"
                lines.append(f"  {marker} {scale.name}: {scale.description}")
                if scale.reason:
                    lines.append(f"    → {scale.reason}")

        categories = [
            ("preprocessing", "PREPROCESAMIENTO"),
            ("dimensionality", "REDUCCIÓN DIMENSIONAL"),
            ("clustering", "CLUSTERING"),
            ("anomaly", "DETECCIÓN DE ANOMALÍAS"),
            ("correlation", "CORRELACIONES"),
            ("feature_analysis", "ANÁLISIS DE VARIABLES"),
        ]

        lines.extend(["", "RECOMENDACIONES DE MÉTODOS", "-" * 40])
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
                    subgroups.append({"name": "Granulometría", "columns": matched,
                                      "mean_sum": float(sums.mean()), "cv": float(cv)})
            break

    major_in_data = [c for c in df.columns if c in MAJOR_ELEMENTS]
    if len(major_in_data) >= 4:
        valid = df[major_in_data].dropna()
        if len(valid) > 0:
            sums = valid.sum(axis=1)
            cv = sums.std() / sums.mean() if sums.mean() > 0 else 999
            if cv < 0.25:
                subgroups.append({"name": "Elementos mayoritarios FRX", "columns": major_in_data,
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
            category="preprocessing", method="CLR Transform (por subgrupo)",
            params={"apply_clr": True, "subgroups": p.compositional_subgroups},
            reason="Datos composicionales detectados. CLR evita correlaciones espurias en PCA.",
            priority=1, confidence=Confidence.HIGH,
            evidence=[f"Subgrupos: {', '.join(names)}",
                      "Sin CLR, PCA y correlaciones producen artefactos en datos cerrados."],
        ))

    if p.pct_skewed > 60:
        recs.append(MethodRecommendation(
            category="preprocessing", method="Log Transform",
            params={"apply_log": True, "auto_detect": True},
            reason=f"{p.pct_skewed:.0f}% de variables con distribución sesgada.",
            priority=1 if not p.is_compositional else 2,
            confidence=Confidence.HIGH if p.pct_skewed > 80 else Confidence.MEDIUM,
            evidence=[f"Variables sesgadas: {', '.join(p.skewed_features[:5])}{'...' if len(p.skewed_features) > 5 else ''}",
                      "Log-transform normaliza distribuciones y estabiliza varianza."],
        ))
    elif p.pct_skewed > 30:
        recs.append(MethodRecommendation(
            category="preprocessing", method="Log Transform (selectivo)",
            params={"apply_log": True, "cols": p.skewed_features},
            reason=f"{p.pct_skewed:.0f}% de variables sesgadas. Aplicar solo a las afectadas.",
            priority=2, confidence=Confidence.MEDIUM,
            evidence=[f"Variables: {', '.join(p.skewed_features[:8])}"],
        ))

    if p.mixed_units_detected:
        recs.append(MethodRecommendation(
            category="preprocessing",
            method="StandardScaler (obligatorio por unidades mixtas)",
            params={"scale_method": "standard"},
            reason="Unidades mixtas (% y mg/kg). Escalado es imprescindible.",
            priority=1, confidence=Confidence.HIGH,
            evidence=[f"Mayoritarios en %: {', '.join(p.major_element_cols[:5])}",
                      f"Traza en mg/kg: {', '.join(p.trace_element_cols[:5])}",
                      "Sin escalar, variables en mg/kg dominarían PCA y clustering."],
        ))
    elif p.pct_outliers_iqr > 20:
        recs.append(MethodRecommendation(
            category="preprocessing", method="RobustScaler",
            params={"scale_method": "robust"},
            reason=f"{p.pct_outliers_iqr:.0f}% de muestras con outliers. RobustScaler es resistente.",
            priority=1, confidence=Confidence.HIGH,
            evidence=[f"Top outliers: " + ", ".join(f"{o['column']}({o['n_outliers']})" for o in p.outlier_columns[:5])],
        ))
    else:
        recs.append(MethodRecommendation(
            category="preprocessing", method="StandardScaler",
            params={"scale_method": "standard"},
            reason="Distribución aceptable. Escalado estándar es suficiente.",
            priority=1, confidence=Confidence.MEDIUM,
        ))

    if p.missing_pattern == "structured":
        recs.append(MethodRecommendation(
            category="preprocessing", method="Análisis por subconjuntos",
            params={"impute_strategy": "subset_analysis"},
            reason="Datos faltantes estructurados (por diseño). No imputar, analizar subconjuntos.",
            priority=1, confidence=Confidence.HIGH,
            evidence=[f"Grupos: {len(p.missing_groups)}"] +
                     [f"Grupo: {g['n_rows']} filas sin {', '.join(g['missing_columns'][:3])}"
                      for g in p.missing_groups[:3]],
        ))
    elif p.pct_missing > 10:
        recs.append(MethodRecommendation(
            category="preprocessing", method="KNN Imputation",
            params={"impute_strategy": "knn", "n_neighbors": 5},
            reason=f"{p.pct_missing:.1f}% faltantes aleatorios. KNN preserva estructura local.",
            priority=1, confidence=Confidence.MEDIUM,
        ))
    elif p.pct_missing > 0:
        recs.append(MethodRecommendation(
            category="preprocessing", method="Imputación por mediana",
            params={"impute_strategy": "median"},
            reason=f"Pocos faltantes ({p.pct_missing:.1f}%). Mediana es robusta y simple.",
            priority=1, confidence=Confidence.HIGH,
        ))

    if p.low_variance_features:
        recs.append(MethodRecommendation(
            category="preprocessing", method="Eliminar variables constantes",
            params={"drop_low_variance": True, "columns": p.low_variance_features},
            reason=f"{len(p.low_variance_features)} variables con varianza casi nula.",
            priority=1, confidence=Confidence.HIGH,
            evidence=[f"Variables: {', '.join(p.low_variance_features)}"],
        ))

    return recs


def _recommend_dimensionality(p: DataProfile) -> list[MethodRecommendation]:
    recs = []

    if p.is_multicollinear or p.n_features > 10:
        evidence = []
        if p.is_multicollinear:
            evidence.append(f"{p.high_correlation_pairs} pares con |r|>0.7")
        if p.correlation_blocks:
            for block in p.correlation_blocks[:3]:
                evidence.append(f"Bloque correlacionado: {', '.join(block[:4])}")
        evidence.append(f"~{p.effective_dimensionality} componentes para 90% de varianza")

        recs.append(MethodRecommendation(
            category="dimensionality", method="PCA",
            params={"method": "pca", "variance_threshold": 0.85},
            reason="Multicolinealidad y alta dimensionalidad. PCA reduce redundancia.",
            priority=1, confidence=Confidence.HIGH, evidence=evidence,
        ))

    if not p.is_small_dataset and p.n_features > 5:
        recs.append(MethodRecommendation(
            category="dimensionality", method="UMAP",
            params={"method": "umap", "n_components": 2, "n_neighbors": 15},
            reason="Para visualización 2D de estructura no-lineal.",
            priority=2, confidence=Confidence.MEDIUM,
            evidence=[f"n={p.n_samples} suficiente para embedding no-lineal.",
                      "UMAP preserva mejor la estructura global que t-SNE."],
        ))
        recs.append(MethodRecommendation(
            category="dimensionality", method="t-SNE",
            params={"method": "tsne", "n_components": 2, "perplexity": 30},
            reason="Alternativa a UMAP, mejor para separar clusters locales.",
            priority=3, confidence=Confidence.MEDIUM,
        ))
    elif p.is_small_dataset:
        recs.append(MethodRecommendation(
            category="dimensionality", method="PCA (solo)",
            params={"method": "pca", "n_components": 2},
            reason="Dataset pequeño. UMAP/t-SNE inestables con pocas muestras.",
            priority=1 if not p.is_multicollinear else 2, confidence=Confidence.HIGH,
            evidence=[f"Solo {p.n_samples} muestras. Métodos no-lineales necesitan >50."],
        ))

    return recs


def _recommend_clustering(p: DataProfile) -> list[MethodRecommendation]:
    recs = []

    if p.has_sites and p.n_sites >= 2:
        recs.append(MethodRecommendation(
            category="clustering",
            method=f"K-Means (K={p.n_sites}, validación geográfica)",
            params={"method": "kmeans", "n_clusters": p.n_sites},
            reason=f"Verificar si {p.n_sites} clusters químicos coinciden con los sitios.",
            priority=1, confidence=Confidence.HIGH,
            evidence=["Si coinciden: contaminación determina agrupación espacial.",
                      "Si no: otros factores dominan la variabilidad.",
                      f"Sitios: {', '.join(list(p.samples_per_site.keys())[:5]) if p.samples_per_site else 'N/A'}"],
        ))

    recs.append(MethodRecommendation(
        category="clustering", method="K-Means (K automático, silhouette)",
        params={"method": "kmeans", "n_clusters": None, "k_range": (2, 10)},
        reason="Búsqueda del K óptimo sin supuestos a priori.",
        priority=2 if p.has_sites else 1, confidence=Confidence.MEDIUM,
        evidence=["Evalúa silhouette para K=2..10 y selecciona el mejor."],
    ))

    recs.append(MethodRecommendation(
        category="clustering", method="Clustering Jerárquico (Ward)",
        params={"method": "hierarchical", "linkage": "ward"},
        reason="Genera dendrograma útil para visualizar relaciones entre muestras/sitios.",
        priority=2, confidence=Confidence.MEDIUM,
        evidence=["El dendrograma es un entregable visual fuerte para la tesis."],
    ))

    if p.pct_outliers_iqr > 15:
        recs.append(MethodRecommendation(
            category="clustering", method="DBSCAN",
            params={"method": "dbscan", "min_samples": 5},
            reason="Outliers significativos. DBSCAN los clasifica como ruido.",
            priority=2, confidence=Confidence.MEDIUM,
            evidence=[f"{p.pct_outliers_iqr:.0f}% de muestras con outliers IQR.",
                      "DBSCAN no requiere especificar K a priori."],
        ))

    return recs


def _recommend_anomaly(p: DataProfile) -> list[MethodRecommendation]:
    recs = []
    contamination = min(max(p.pct_outliers_zscore / 100, 0.02), 0.15) if p.pct_outliers_zscore > 0 else 0.05

    recs.append(MethodRecommendation(
        category="anomaly", method="Isolation Forest",
        params={"method": "isolation_forest", "contamination": round(contamination, 3)},
        reason="Método principal. Escala bien con alta dimensionalidad.",
        priority=1, confidence=Confidence.HIGH,
        evidence=[f"Contamination estimada: {contamination:.1%} (basada en outliers z-score)."],
    ))

    recs.append(MethodRecommendation(
        category="anomaly", method="LOF (Local Outlier Factor)",
        params={"method": "lof", "n_neighbors": 20, "contamination": round(contamination, 3)},
        reason="Complemento basado en densidad local. Detecta anomalías contextuales.",
        priority=2, confidence=Confidence.MEDIUM,
    ))

    if p.has_heavy_metals:
        recs.append(MethodRecommendation(
            category="anomaly", method="Z-score (metales pesados)",
            params={"method": "zscore", "threshold": 3.0, "cols": p.heavy_metal_cols},
            reason="Detección univariada en metales pesados para identificar hotspots.",
            priority=2, confidence=Confidence.HIGH,
            evidence=[f"Metales pesados: {', '.join(p.heavy_metal_cols)}",
                      "Z-score identifica QUÉ metal es anómalo en cada muestra."],
        ))

    return recs


def _recommend_correlation(p: DataProfile) -> list[MethodRecommendation]:
    recs = []

    recs.append(MethodRecommendation(
        category="correlation", method="Pearson + Spearman comparativo",
        params={"method": "compare"},
        reason="Comparar lineal vs. monotónica para detectar relaciones no-lineales.",
        priority=1, confidence=Confidence.HIGH,
        evidence=["Diferencias grandes revelan no-linealidad.",
                  "Ambas matrices son entregables clave para la tesis."],
    ))

    if p.has_heavy_metals and p.has_granulometry:
        recs.append(MethodRecommendation(
            category="correlation",
            method="Correlación metales vs. granulometría",
            params={"method": "spearman", "subset_a": p.heavy_metal_cols, "subset_b": p.granulometry_cols},
            reason="Evaluar si la acumulación de metales depende del tamaño de grano.",
            priority=1, confidence=Confidence.HIGH,
            evidence=["Arcillas y limos suelen retener más metales que arenas.",
                      "Correlación positiva metal-arcilla confirmaría mecanismo de adsorción."],
        ))

    return recs


def _recommend_feature_analysis(p: DataProfile) -> list[MethodRecommendation]:
    recs = []

    recs.append(MethodRecommendation(
        category="feature_analysis", method="RF Feature Importance (por cluster)",
        params={"method": "rf_cluster_discrimination"},
        reason="Identifica qué variables discriminan más entre clusters.",
        priority=1, confidence=Confidence.HIGH,
        evidence=["Random Forest + Permutation importance para robustez."],
    ))

    if p.has_heavy_metals and p.has_sites:
        recs.append(MethodRecommendation(
            category="feature_analysis", method="RF Feature Importance (por sitio)",
            params={"method": "rf_site_discrimination", "target": "site"},
            reason="Identifica qué metales diferencian más los sitios contaminados.",
            priority=1, confidence=Confidence.HIGH,
            evidence=["Usar sitio como target. Metales más importantes = contaminantes clave."],
        ))

    return recs


def _recommend_analysis_scales(p: DataProfile) -> list[AnalysisScale]:
    scales = [AnalysisScale(name="Global", description="Todas las muestras.",
                            recommended=True, reason="Visión general y relaciones entre sitios.")]

    if p.has_sites and p.n_sites >= 2:
        scales.append(AnalysisScale(
            name="Por sitio", description=f"Separado para cada uno de los {p.n_sites} sitios.",
            filter_col="site", recommended=True,
            reason="Patrones intra-sitio y comparación de perfiles de contaminación."))

    if p.has_depth and p.has_depth_gradient:
        scales.append(AnalysisScale(
            name="Perfil de profundidad", description="Variación vertical por perfil.",
            filter_col="depth", recommended=True,
            reason="Gradientes detectados. Analizar tendencias temporales en sedimentos."))

    if p.has_major_elements and p.has_trace_elements:
        scales.append(AnalysisScale(
            name="Mayoritarios vs. Traza", description="Análisis separado por tipo de elemento.",
            recommended=True, reason="Diferentes escalas y significado geoquímico."))

    return scales


def _generate_warnings(p: DataProfile) -> list[str]:
    warnings = []
    if p.is_wide_dataset:
        warnings.append(f"Más variables ({p.n_features}) que muestras ({p.n_samples}). "
                        "PCA puede sobreajustar.")
    if p.ratio_samples_features < 5:
        warnings.append(f"Ratio muestras/features bajo ({p.ratio_samples_features:.1f}). "
                        "Resultados de ML podrían ser inestables.")
    if p.unbalanced_sites:
        warnings.append("Muestras desbalanceadas entre sitios. Clustering puede sesgar.")
    if p.mixed_units_detected:
        warnings.append("Variables en diferentes unidades (% y mg/kg). "
                        "Escalado OBLIGATORIO antes de análisis multivariado.")
    if p.pct_skewed > 80:
        warnings.append("Casi todas las variables sesgadas. Considerar log-transform antes del escalado.")
    if len(p.bimodal_features) > p.n_features * 0.3:
        warnings.append(f"{len(p.bimodal_features)} variables bimodales. "
                        "Posible indicador de dos poblaciones distintas.")
    return warnings


# ============================================================
#  MAIN INTERFACE
# ============================================================

def recommend(profile: DataProfile) -> AnalysisPlan:
    """Genera el plan completo de análisis."""
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
    Interfaz principal: analiza el dataset y genera un plan de análisis completo.
    """
    profile = profile_dataset(
        df, has_coordinates=has_coordinates, has_depth=has_depth,
        depth_col=depth_col, has_sites=has_sites, site_col=site_col,
        n_sites=n_sites, original_df=original_df,
    )
    return recommend(profile)
