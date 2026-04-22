"""Build a consolidated interpretation report combining TEL/PEL and EF analyses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from aeda.interpretation.classification import classify_tel_pel, classify_ef_birch
from aeda.interpretation.normalization import compute_enrichment_factor, EnrichmentFactorResult


@dataclass
class InterpretationReport:
    """Consolidated interpretation of contamination levels."""

    tel_pel_classifications: pd.DataFrame
    ef_result: Optional[EnrichmentFactorResult] = None
    ef_classifications: Optional[pd.DataFrame] = None
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
    """Run full interpretation pipeline and return consolidated report."""
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
