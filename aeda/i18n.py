"""
Lightweight internationalization for the engine layer.

The analysis engine generates user-facing messages (recommendation reasons,
evidence, plan warnings, analysis-scale descriptions, validation messages).
To keep the engine framework-agnostic, this module holds the active language
in a module-level variable that the UI sets via ``set_lang()``. The engine
calls ``t()`` to translate its own messages.

Defaults to English so the engine behaves identically to its original form when
used standalone (e.g. in tests) and degrades gracefully: any string without a
translation is returned unchanged.
"""

DEFAULT_LANG = "en"
_lang = DEFAULT_LANG


def set_lang(lang: str) -> None:
    """Set the active language for engine-generated messages ('es' or 'en')."""
    global _lang
    if lang in ("es", "en"):
        _lang = lang


def get_lang() -> str:
    return _lang


def t(s: str) -> str:
    """Translate an engine message. Returns English (the key) if lang is 'en'
    or if no translation exists."""
    if _lang == "en":
        return s
    return ES.get(s, s)


# Engine message catalog (English key -> Spanish). Grown as engine modules are
# migrated. Templates keep their {placeholders} so callers do t(...).format(...).
ES: dict[str, str] = {
    # --- validators (aeda/io/validators.py) ---
    "Structured missing-data pattern detected: "
    "{n_rows} rows with these columns empty simultaneously. "
    "Likely unmeasured values by experimental design.":
        "Patrón estructurado de datos faltantes detectado: "
        "{n_rows} filas con estas columnas vacías simultáneamente. "
        "Probablemente valores no medidos por diseño experimental.",
    "{n_null} missing values ({pct:.1f}%)": "{n_null} valores faltantes ({pct:.1f}%)",
    "{n_neg} negative concentration values detected":
        "{n_neg} valores de concentración negativos detectados",
    "Granulometric fractions do not sum to ~100% in {n_bad} rows. "
    "Range: {smin:.1f}% - {smax:.1f}%":
        "Las fracciones granulométricas no suman ~100% en {n_bad} filas. "
        "Rango: {smin:.1f}% - {smax:.1f}%",
    "{n_out} extreme outliers (IQR×{factor})":
        "{n_out} valores atípicos extremos (IQR×{factor})",
    "Constant column (variance = 0), will be excluded from analysis":
        "Columna constante (varianza = 0), se excluirá del análisis",
    "Column with very low variability ({nunique} unique values)":
        "Columna con muy baja variabilidad ({nunique} valores únicos)",
    # --- auto_selector: reasons ---
    "Compositional data detected (sum ~100%, low CV). "
    "CLR transformation is recommended before multivariate analysis. "
    "Per scientific tutor decision, this transformation must be "
    "explicitly enabled by the user (apply_clr=True). "
    "It will NOT be applied automatically.":
        "Datos composicionales detectados (suma ~100%, CV bajo). "
        "Se recomienda la transformación CLR antes del análisis multivariado. "
        "Por decisión del tutor científico, esta transformación debe ser "
        "habilitada explícitamente por el usuario (apply_clr=True). "
        "NO se aplicará automáticamente.",
    "{pct:.0f}% of variables are skewed.": "{pct:.0f}% de las variables son asimétricas.",
    "{pct:.0f}% of variables are skewed. Apply only to affected variables.":
        "{pct:.0f}% de las variables son asimétricas. Aplicar solo a las variables afectadas.",
    "Mixed units (% and mg/kg). Scaling is mandatory.":
        "Unidades mixtas (% y mg/kg). El escalado es obligatorio.",
    "{pct:.0f}% of samples contain outliers. RobustScaler is resilient.":
        "{pct:.0f}% de las muestras contienen valores atípicos. RobustScaler es resistente.",
    "Distribution is acceptable. Standard scaling is sufficient.":
        "La distribución es aceptable. El escalado estándar es suficiente.",
    "Structured missing data (by design). Do not impute; analyze subsets.":
        "Datos faltantes estructurados (por diseño). No imputar; analizar subconjuntos.",
    "{pct:.1f}% random missingness. KNN preserves local structure.":
        "{pct:.1f}% de datos faltantes aleatorios. KNN preserva la estructura local.",
    "Few missing values ({pct:.1f}%). Median is robust and simple.":
        "Pocos valores faltantes ({pct:.1f}%). La mediana es robusta y simple.",
    "{n} variables with near-zero variance.": "{n} variables con varianza casi nula.",
    "Multicollinearity and high dimensionality. PCA reduces redundancy.":
        "Multicolinealidad y alta dimensionalidad. El PCA reduce la redundancia.",
    "For 2D visualization of nonlinear structure.":
        "Para visualización 2D de estructura no lineal.",
    "Alternative to UMAP, often better for separating local clusters.":
        "Alternativa a UMAP, a menudo mejor para separar grupos locales.",
    "Small dataset. UMAP/t-SNE can be unstable with few samples.":
        "Conjunto pequeño. UMAP/t-SNE pueden ser inestables con pocas muestras.",
    "Check whether {n} chemical clusters match known sites.":
        "Comprobar si {n} grupos químicos coinciden con los sitios conocidos.",
    "Search for optimal K without strong prior assumptions.":
        "Buscar el K óptimo sin supuestos previos fuertes.",
    "Produces a dendrogram useful for visualizing sample/site relationships.":
        "Produce un dendrograma útil para visualizar relaciones entre muestras/sitios.",
    "Significant outliers detected. DBSCAN can label them as noise.":
        "Valores atípicos significativos detectados. DBSCAN puede etiquetarlos como ruido.",
    "Multi-site dataset with depth: compare sites using only their "
    "surface (recent) sediment to avoid mixing historical periods.":
        "Conjunto multi-sitio con profundidad: comparar sitios usando solo su "
        "sedimento superficial (reciente) para evitar mezclar períodos históricos.",
    "Primary method. Scales well with high dimensionality.":
        "Método primario. Escala bien con alta dimensionalidad.",
    "Local-density complement. Detects contextual anomalies.":
        "Complemento de densidad local. Detecta anomalías contextuales.",
    "Univariate heavy-metal detection to identify hotspots.":
        "Detección univariada de metales pesados para identificar focos.",
    "Compare linear vs monotonic behavior to detect nonlinear relationships.":
        "Comparar comportamiento lineal vs. monótono para detectar relaciones no lineales.",
    "Assess whether metal accumulation depends on grain size.":
        "Evaluar si la acumulación de metales depende del tamaño de grano.",
    "Identify which variables best discriminate clusters.":
        "Identificar qué variables discriminan mejor los grupos.",
    "Identify which metals most differentiate contaminated sites.":
        "Identificar qué metales diferencian más los sitios contaminados.",
    # --- auto_selector: evidence ---
    "Subgroups: {names}": "Subgrupos: {names}",
    "Without CLR, PCA and correlations produce artifacts in closed data.":
        "Sin CLR, el PCA y las correlaciones producen artefactos en datos cerrados.",
    "Skewed variables: {vars}": "Variables asimétricas: {vars}",
    "Log-transform normalizes distributions and stabilizes variance.":
        "La transformación logarítmica normaliza las distribuciones y estabiliza la varianza.",
    "Features: {vars}": "Variables: {vars}",
    "Major elements in %: {cols}": "Elementos mayoritarios en %: {cols}",
    "Trace elements in mg/kg: {cols}": "Elementos traza en mg/kg: {cols}",
    "Without scaling, mg/kg variables would dominate PCA and clustering.":
        "Sin escalado, las variables en mg/kg dominarían el PCA y el agrupamiento.",
    "Top outliers:": "Atípicos principales:",
    "Groups: {n}": "Grupos: {n}",
    "Group: {rows} rows missing {cols}": "Grupo: {rows} filas sin {cols}",
    "{n} pairs with |r|>{thr}": "{n} pares con |r|>{thr}",
    "Correlated block: {cols}": "Bloque correlacionado: {cols}",
    "~{n} components for 90% explained variance":
        "~{n} componentes para el 90% de varianza explicada",
    "n={n} is sufficient for nonlinear embedding.":
        "n={n} es suficiente para una proyección no lineal.",
    "UMAP typically preserves global structure better than t-SNE.":
        "UMAP suele preservar mejor la estructura global que t-SNE.",
    "Only {n} samples. Nonlinear methods usually need >50.":
        "Solo {n} muestras. Los métodos no lineales suelen necesitar >50.",
    "If they match: contamination likely drives spatial grouping.":
        "Si coinciden: la contaminación probablemente explica la agrupación espacial.",
    "If not: other factors dominate variability.":
        "Si no: otros factores dominan la variabilidad.",
    "Sites: {sites}": "Sitios: {sites}",
    "Evaluates silhouette for K=2..10 and selects the best.":
        "Evalúa la silueta para K=2..10 y selecciona el mejor.",
    "The dendrogram is a strong visual deliverable for the thesis.":
        "El dendrograma es un entregable visual fuerte para la tesis.",
    "{pct:.0f}% of samples have IQR outliers.":
        "{pct:.0f}% de las muestras tienen atípicos IQR.",
    "DBSCAN does not require specifying K a priori.":
        "DBSCAN no requiere especificar K a priori.",
    "{n} sites available for inter-site comparison.":
        "{n} sitios disponibles para comparación entre sitios.",
    "Surface layer default is 0-10 cm (Yoelvis 2026, LEA-CEAC).":
        "La capa superficial por defecto es 0-10 cm (Yoelvis 2026, LEA-CEAC).",
    "Aggregates each site to its mean to avoid bias from cores with more samples.":
        "Agrega cada sitio a su media para evitar el sesgo de núcleos con más muestras.",
    "Standard approach: Birch (2003), Buchman (2008).":
        "Enfoque estándar: Birch (2003), Buchman (2008).",
    "Estimated contamination: {c:.1%} (based on z-score outliers).":
        "Contaminación estimada: {c:.1%} (según atípicos por z-score).",
    "Heavy metals: {cols}": "Metales pesados: {cols}",
    "Z-score indicates WHICH metal is anomalous in each sample.":
        "El z-score indica QUÉ metal es anómalo en cada muestra.",
    "Large differences indicate nonlinearity.":
        "Las diferencias grandes indican no linealidad.",
    "Both matrices are key thesis deliverables.":
        "Ambas matrices son entregables clave de la tesis.",
    "Clays and silts usually retain more metals than sands.":
        "Las arcillas y los limos suelen retener más metales que las arenas.",
    "Positive metal-clay correlation supports adsorption mechanisms.":
        "Una correlación positiva metal-arcilla respalda mecanismos de adsorción.",
    "Random Forest + permutation importance for robustness.":
        "Random Forest + importancia por permutación para robustez.",
    "Use site as target. Most important metals = key contaminants.":
        "Usar el sitio como objetivo. Los metales más importantes = contaminantes clave.",
    # --- auto_selector: analysis scales ---
    "All samples.": "Todas las muestras.",
    "Global overview and inter-site relationships.": "Visión global y relaciones entre sitios.",
    "By site": "Por sitio",
    "Separate analysis for each of the {n} sites.":
        "Análisis separado para cada uno de los {n} sitios.",
    "Intra-site patterns and contamination-profile comparison.":
        "Patrones intra-sitio y comparación de perfiles de contaminación.",
    "Depth profile": "Perfil de profundidad",
    "Vertical variation by profile.": "Variación vertical por perfil.",
    "Gradients detected. Analyze temporal trends in sediments.":
        "Gradientes detectados. Analizar tendencias temporales en los sedimentos.",
    "Major vs Trace": "Mayoritarios vs. traza",
    "Separate analysis by element type.": "Análisis separado por tipo de elemento.",
    "Different scales and geochemical meaning.": "Escalas y significado geoquímico diferentes.",
    # --- auto_selector: warnings ---
    "More variables ({nf}) than samples ({ns}). PCA may overfit.":
        "Más variables ({nf}) que muestras ({ns}). El PCA puede sobreajustar.",
    "Low sample/feature ratio ({r:.1f}). ML results may be unstable.":
        "Razón muestras/variables baja ({r:.1f}). Los resultados de ML pueden ser inestables.",
    "Unbalanced samples across sites. Clustering may be biased.":
        "Muestras desbalanceadas entre sitios. El agrupamiento puede estar sesgado.",
    "Variables are in different units (% and mg/kg). Scaling is MANDATORY before multivariate analysis.":
        "Las variables están en unidades distintas (% y mg/kg). El escalado es OBLIGATORIO antes del análisis multivariado.",
    "Almost all variables are skewed. Consider log-transform before scaling.":
        "Casi todas las variables son asimétricas. Considere una transformación logarítmica antes del escalado.",
    "{n} bimodal variables. Possible indicator of two distinct populations.":
        "{n} variables bimodales. Posible indicador de dos poblaciones distintas.",
}

