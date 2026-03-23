"""
Matrix Type Detection Engine for Environmental Samples

Automatically identifies the type of environmental matrix (sediment, soil, water, air, biota)
based on analytical signatures, element profiles, and physical properties.

This enables the universal ingestion module to apply appropriate preprocessing rules
for each matrix type (e.g., granulometry is specific to solid matrices).
"""

from __future__ import annotations

import logging
from typing import TypedDict

import pandas as pd


class MatrixProfile(TypedDict, total=False):
    """Detection profile for a matrix type"""
    name: str
    confidence: float
    indicators: dict[str, int]  # indicator_name -> count
    reasoning: list[str]


class MatrixTypeDetectionError(Exception):
    pass


class MatrixTypeDetector:
    """
    Detects environmental matrix type using multi-criteria analysis:
    1. Physical properties (granulometry, density, porosity)
    2. Geochemical signature (element ratios, normalization base)
    3. Analytical characteristics (concentration ranges, units)
    4. Metadata patterns (column names, keywords)
    """
    
    # Keywords and their scores for each matrix type
    _MATRIX_PROFILES = {
        "sediment": {
            "keywords": {
                "depth": 3,
                "core": 3,
                "granulometry": 2,
                "grain": 2,
                "mud": 2,
                "silt": 2,
                "clay": 2,
                "sand": 1,
                "ppi": 2,  # Loss on Ignition specific to sediments
                "mangrove": 3,
                "estuary": 2,
                "coastal": 2,
                "pore": 1,
            },
            "elements": {
                # These elements are typically analyzed in sediment via FRX
                "high_priority": {"fe", "mn", "zn", "pb", "cr", "cu", "ni", "as", "v", "co", "ba"},
                "typical_range": (10, 1e6),  # ppm typical for trace elements
            },
            "physical_props": {
                "has_granulometry": 2,
                "has_organic_matter": 2,
                "has_humidity": 1,
            }
        },
        "soil": {
            "keywords": {
                "soil": 3,
                "horizon": 3,
                "texture": 2,
                "organic": 2,
                "clay": 2,
                "silt": 2,
                "sand": 1,
                "cec": 2,  # Cation Exchange Capacity
                "topsoil": 2,
                "subsoil": 2,
            },
            "elements": {
                "high_priority": {"fe", "mn", "zn", "pb", "cr", "cu", "ni", "as"},
                "typical_range": (1, 1e5),
            },
            "physical_props": {
                "has_granulometry": 2,
                "has_organic_matter": 2,
            }
        },
        "water": {
            "keywords": {
                "salinity": 3,
                "conductivity": 3,
                "turbidity": 2,
                "bod": 2,  # Biochemical Oxygen Demand
                "cod": 2,  # Chemical Oxygen Demand
                "do": 2,  # Dissolved Oxygen
                "ph": 1,
                "temp": 1,
                "dissolved": 2,
                "suspended": 2,
                "aquatic": 2,
                "river": 2,
                "sea": 2,
            },
            "elements": {
                "high_priority": {"na", "cl", "s", "br", "k", "mg", "ca"},
                "typical_range": (0.1, 1e4),
            },
            "physical_props": {
                "has_salinity": 3,
                "has_conductivity": 2,
            }
        },
        "air": {
            "keywords": {
                "pm2": 3,
                "pm10": 3,
                "no2": 3,
                "so2": 3,
                "o3": 3,
                "co": 2,
                "aqi": 2,
                "atmospheric": 2,
                "aerosol": 2,
                "particulate": 2,
            },
            "elements": {
                "high_priority": set(),  # Air particles are chemical compounds, not elemental
                "typical_range": (0.001, 1000),
            },
            "physical_props": {}
        },
        "biota": {
            "keywords": {
                "species": 3,
                "tissue": 3,
                "organism": 2,
                "taxon": 2,
                "biomass": 2,
                "fish": 2,
                "algae": 2,
                "plant": 1,
                "organism": 2,
                "bioaccumulation": 2,
            },
            "elements": {
                "high_priority": {"zn", "pb", "cd", "hg", "cu", "cr", "ni"},
                "typical_range": (0.01, 1e4),
            },
            "physical_props": {}
        },
    }
    
    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger(__name__)
    
    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize column names for keyword matching"""
        return str(text).strip().lower().replace(" ", "_").replace("-", "_")
    
    def detect(
        self,
        data: pd.DataFrame,
        matrix_type_hint: str | None = None,
        confidence_threshold: float = 0.5,
    ) -> tuple[str, MatrixProfile]:
        """
        Detect the matrix type of the dataset.
        
        Args:
            data: DataFrame with analytical data
            matrix_type_hint: If provided, validates this type. Otherwise auto-detects.
            confidence_threshold: Minimum confidence score to accept detection (0-1)
            
        Returns:
            Tuple of (detected_matrix_type, profile_dict)
            
        Raises:
            MatrixTypeDetectionError: If detection fails or matrix_type_hint is invalid
        """
        # Validate hint if provided
        if matrix_type_hint:
            hint_normalized = matrix_type_hint.strip().lower()
            valid_types = set(self._MATRIX_PROFILES.keys())
            if hint_normalized not in valid_types:
                raise MatrixTypeDetectionError(
                    f"Invalid matrix_type_hint '{matrix_type_hint}'. "
                    f"Valid options: {sorted(valid_types)}"
                )
            # If hint is provided, just validate and return it
            profile = self._score_matrix(data, hint_normalized)
            self.logger.info(
                f"Matrix type provided as hint: {hint_normalized} "
                f"(confidence: {profile['confidence']:.2f})"
            )
            return hint_normalized, profile
        
        # Auto-detect
        scores: dict[str, MatrixProfile] = {}
        for matrix_name in self._MATRIX_PROFILES.keys():
            scores[matrix_name] = self._score_matrix(data, matrix_name)
        
        # Find best match
        best_match = max(scores.items(), key=lambda x: x[1]["confidence"])
        best_type, best_profile = best_match
        
        if best_profile["confidence"] < confidence_threshold:
            self.logger.warning(
                f"Low confidence detection: {best_type} ({best_profile['confidence']:.2f}) "
                f"below threshold {confidence_threshold}. Defaulting to 'generic'."
            )
            return "generic", best_profile
        
        self.logger.info(
            f"Detected matrix type: {best_type} "
            f"(confidence: {best_profile['confidence']:.2f})"
        )
        
        return best_type, best_profile
    
    def _score_matrix(self, data: pd.DataFrame, matrix_type: str) -> MatrixProfile:
        """
        Score how well the data matches a given matrix type.
        
        Returns profile with confidence score (0-1).
        """
        profile: MatrixProfile = {
            "name": matrix_type,
            "confidence": 0.0,
            "indicators": {},
            "reasoning": [],
        }
        
        normalized_cols = {self._normalize_text(col) for col in data.columns}
        profile_spec = self._MATRIX_PROFILES[matrix_type]
        
        # Score 1: Keyword matching
        keyword_score = 0
        max_keyword_score = 0
        for keyword, weight in profile_spec["keywords"].items():
            max_keyword_score += weight
            if any(keyword in col for col in normalized_cols):
                keyword_score += weight
                profile["indicators"][f"keyword_{keyword}"] = weight
        
        keyword_normalized = keyword_score / max_keyword_score if max_keyword_score > 0 else 0
        
        # Score 2: Element signature
        element_score = 0
        high_priority_elements = profile_spec["elements"]["high_priority"]
        if high_priority_elements:
            found_elements = sum(
                1 for elem in high_priority_elements
                if any(elem in col.lower() for col in data.columns)
            )
            element_score = found_elements / len(high_priority_elements)
            profile["indicators"]["element_match"] = found_elements
        
        # Score 3: Concentration ranges (heuristic)
        range_score = 0
        if len(data.select_dtypes(include=["number"]).columns) > 0:
            numeric_df = data.select_dtypes(include=["number"])
            medians = numeric_df.median()
            for median_val in medians:
                if pd.notna(median_val):
                    if profile_spec["elements"]["typical_range"][0] <= abs(median_val) <= profile_spec["elements"]["typical_range"][1]:
                        range_score += 1
            
            range_score = range_score / len(medians) if len(medians) > 0 else 0
            profile["indicators"]["concentration_range_match"] = range_score
        
        # Score 4: Physical properties
        phys_score = 0
        max_phys_score = 0
        for prop_name, prop_weight in profile_spec["physical_props"].items():
            max_phys_score += prop_weight
            if self._has_property(data, prop_name):
                phys_score += prop_weight
                profile["indicators"][prop_name] = prop_weight
        
        phys_normalized = phys_score / max_phys_score if max_phys_score > 0 else 0
        
        # Combine scores with weights
        weights = {
            "keywords": 0.4,
            "elements": 0.35,
            "ranges": 0.15,
            "physical": 0.1,
        }
        
        total_confidence = (
            weights["keywords"] * keyword_normalized +
            weights["elements"] * element_score +
            weights["ranges"] * range_score +
            weights["physical"] * phys_normalized
        )
        
        profile["confidence"] = min(1.0, max(0.0, total_confidence))
        
        if keyword_normalized > 0.3:
            profile["reasoning"].append(f"Strong keyword match (score: {keyword_normalized:.2f})")
        if element_score > 0.3:
            profile["reasoning"].append(f"Expected elements detected (match: {element_score:.2f})")
        
        return profile
    
    @staticmethod
    def _has_property(data: pd.DataFrame, property_name: str) -> bool:
        """Check if data has expected physical properties"""
        normalized_cols = {col.lower().replace(" ", "_") for col in data.columns}
        
        property_checks = {
            "has_granulometry": {"granulometry", "grain_size", "particle_size", "texture"},
            "has_organic_matter": {"ppi", "loi", "organic", "carbon", "matter"},
            "has_humidity": {"humidity", "moisture", "water_content"},
            "has_salinity": {"salinity", "pss"},
            "has_conductivity": {"conductivity", "ec", "cond"},
        }
        
        if property_name not in property_checks:
            return False
        
        return any(
            keyword in col for keyword in property_checks[property_name]
            for col in normalized_cols
        )
