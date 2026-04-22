"""Environmental interpretation layer for AEDA-AI.

This module applies domain-specific toxicological and geochemical analysis
on top of the exploratory results produced by the ML engine. It computes
enrichment factors, applies regulatory thresholds (TEL/PEL), and classifies
samples by contamination level.

References
----------
Buchman, M. F. (2008). NOAA Screening Quick Reference Tables. NOAA OR&R Report 08-1.
Bolanos-Alvarez et al. (2024). Sci. Total Environ. 920, 170609.
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
