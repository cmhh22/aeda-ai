"""
Data Quality Report Generator

Creates comprehensive quality reports for scientifically interpretable data transformation.
Shows exactly what the framework did to the data, enabling informed decisions.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd


class DataQualityReporter:
    """
    Generates human-readable quality reports of the data ingestion process.
    
    Reports include:
    - Overview metrics
    - Censored value handling (BDL/AQL)
    - Missing data analysis
    - Outlier detection summary
    - Schema transformations
    - Recommendations
    """
    
    def __init__(self, framework_version: str = "1.0") -> None:
        self.framework_version = framework_version
        self.report_data: dict[str, Any] = {}
    
    def generate_report(
        self,
        raw_data: pd.DataFrame,
        processed_data: pd.DataFrame,
        matrix_type: str,
        censored_metadata: list | None = None,
        quality_flags: dict | None = None,
    ) -> str:
        """
        Generate a comprehensive quality report in human-readable format.
        
        Args:
            raw_data: Original unprocessed data
            processed_data: After ingestion processing
            matrix_type: Detected matrix type
            censored_metadata: List of CensoredValueMetadata objects
            quality_flags: Dict of quality issues encountered
            
        Returns:
            Formatted report as string (ready for printing or saving to file)
        """
        report_lines = []
        
        # Header
        report_lines.append("=" * 80)
        report_lines.append("AEDA FRAMEWORK - DATA INGESTION QUALITY REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Framework Version: {self.framework_version}")
        report_lines.append("")
        
        # 1. Overview
        report_lines.append("1. DATASET OVERVIEW")
        report_lines.append("-" * 80)
        report_lines.append(f"  Matrix Type Detected: {matrix_type}")
        report_lines.append(f"  Samples (rows): {raw_data.shape[0]}")
        report_lines.append(f"  Variables (columns): {raw_data.shape[1]}")
        report_lines.append(f"  Processed Variables: {processed_data.shape[1]}")
        report_lines.append("")
        
        # 2. Data Completeness
        report_lines.extend(self._section_completeness(raw_data, processed_data))
        
        # 3. Censored Values
        if censored_metadata:
            report_lines.extend(self._section_censored_values(censored_metadata))
        
        # 4. Quality Flags
        if quality_flags:
            report_lines.extend(self._section_quality_flags(quality_flags))
        
        # 5. Numeric Summary Statistics
        report_lines.extend(self._section_statistics(processed_data))
        
        # 6. Recommendations
        report_lines.extend(self._section_recommendations(processed_data, censored_metadata, quality_flags))
        
        # Footer
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("End of Report")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    @staticmethod
    def _section_completeness(raw_data: pd.DataFrame, processed_data: pd.DataFrame) -> list[str]:
        """Data completeness analysis"""
        lines = ["2. DATA COMPLETENESS"]
        lines.append("-" * 80)
        
        missing_raw = raw_data.isnull().sum()
        missing_processed = processed_data.isnull().sum()
        
        total_missing_raw = missing_raw.sum()
        total_missing_processed = missing_processed.sum()
        
        lines.append(f"  Missing values in RAW: {total_missing_raw} ({100*total_missing_raw/(raw_data.shape[0]*raw_data.shape[1]):.1f}%)")
        lines.append(f"  Missing values in PROCESSED: {total_missing_processed} ({100*total_missing_processed/(processed_data.shape[0]*processed_data.shape[1]):.1f}%)")
        
        cols_with_missing = missing_processed[missing_processed > 0].sort_values(ascending=False)
        if len(cols_with_missing) > 0:
            lines.append(f"  Columns with missing values in PROCESSED ({len(cols_with_missing)}):")
            for col, count in cols_with_missing.head(10).items():
                pct = 100 * count / processed_data.shape[0]
                lines.append(f"    - {col}: {count} ({pct:.1f}%)")
            if len(cols_with_missing) > 10:
                lines.append(f"    ... and {len(cols_with_missing) - 10} more")
        
        lines.append("")
        return lines
    
    @staticmethod
    def _section_censored_values(censored_metadata: list) -> list[str]:
        """Censored values (BDL/AQL) analysis"""
        lines = ["3. CENSORED VALUES HANDLING (BDL/AQL)"]
        lines.append("-" * 80)
        
        total_bdl = sum(m.n_bdl for m in censored_metadata)
        total_aloq = sum(m.n_aloq for m in censored_metadata)
        
        lines.append(f"  Total BDL (Below Detection Limit) values handled: {total_bdl}")
        lines.append(f"  Total ALOQ (Above Limit of Quantification) values: {total_aloq}")
        
        if censored_metadata:
            strategy = censored_metadata[0].imputation_strategy
            lines.append(f"  Imputation Strategy: {strategy}")
            lines.append("")
            lines.append("  Columns with censored values:")
            for meta in sorted(censored_metadata, key=lambda x: x.n_bdl + x.n_aloq, reverse=True)[:15]:
                if meta.n_bdl + meta.n_aloq > 0:
                    lines.append(
                        f"    - {meta.column}: {meta.n_bdl} BDL (LOD={meta.lod_value:.2e}), "
                        f"{meta.n_aloq} ALOQ"
                    )
        
        lines.append("")
        return lines
    
    @staticmethod
    def _section_quality_flags(quality_flags: dict) -> list[str]:
        """Quality flags and parsing notes"""
        lines = ["4. PARSING QUALITY FLAGS"]
        lines.append("-" * 80)
        
        total_flags = {}
        for col, flags in quality_flags.items():
            if isinstance(flags, dict):
                for flag_type, count in flags.items():
                    total_flags[flag_type] = total_flags.get(flag_type, 0) + count
        
        if total_flags:
            for flag_type, count in sorted(total_flags.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"  {flag_type}: {count} occurrences")
        else:
            lines.append("  No parsing issues detected.")
        
        lines.append("")
        return lines
    
    @staticmethod
    def _section_statistics(processed_data: pd.DataFrame) -> list[str]:
        """Summary statistics of processed data"""
        lines = ["5. PROCESSED DATA STATISTICS"]
        lines.append("-" * 80)
        
        numeric_cols = processed_data.select_dtypes(include=["number"]).columns
        
        lines.append(f"  Numeric columns: {len(numeric_cols)}")
        
        if len(numeric_cols) > 0:
            lines.append("")
            lines.append("  Summary (median, Q1, Q3, range):")
            for col in numeric_cols[:10]:
                values = processed_data[col].dropna()
                if len(values) > 0:
                    q1 = values.quantile(0.25)
                    median = values.median()
                    q3 = values.quantile(0.75)
                    min_val = values.min()
                    max_val = values.max()
                    lines.append(
                        f"    {col}: median={median:.2e} [Q1={q1:.2e}, Q3={q3:.2e}] "
                        f"range=[{min_val:.2e}, {max_val:.2e}]"
                    )
            
            if len(numeric_cols) > 10:
                lines.append(f"    ... and {len(numeric_cols) - 10} more numeric columns")
        
        lines.append("")
        return lines
    
    @staticmethod
    def _section_recommendations(processed_data: pd.DataFrame, censored_metadata: list | None = None, quality_flags: dict | None = None) -> list[str]:
        """Generate actionable recommendations"""
        lines = ["6. RECOMMENDATIONS FOR ANALYSIS"]
        lines.append("-" * 80)
        
        recommendations = []
        
        # Check for high censoring
        if censored_metadata:
            high_censoring = [m for m in censored_metadata if (m.n_bdl + m.n_aloq) > 0]
            if high_censoring:
                recommendations.append(
                    "• Consider sensitivity analysis: compare results with alternative "
                    "imputation strategies (ROS, QMLE) for censored values"
                )
        
        # Check for missing data
        missing_pct = processed_data.isnull().sum().sum() / (processed_data.shape[0] * processed_data.shape[1])
        if missing_pct > 0.05:
            recommendations.append(
                f"• High proportion of missing values ({100*missing_pct:.1f}%). "
                f"Consider specialized imputation or feature selection."
            )
        
        # Check for skewness
        numeric_cols = processed_data.select_dtypes(include=["number"]).columns
        if len(numeric_cols) > 0:
            skewed_cols = [col for col in numeric_cols if processed_data[col].std() / (processed_data[col].mean() + 1e-10) > 1.5]
            if skewed_cols:
                recommendations.append(
                    f"• {len(skewed_cols)} columns show high skewness. "
                    f"Consider log-transformation for PCA/clustering."
                )
        
        # Censoring impact on ML
        if censored_metadata and len([m for m in censored_metadata if m.n_bdl > 0]) > 0.3 * len(processed_data):
            recommendations.append(
                "• Significant censoring detected. Tree-based models (Random Forest, XGBoost) "
                "are recommended for exploratory analysis as they handle censored data better than linear methods."
            )
        
        if not recommendations:
            recommendations.append("• Dataset quality is good. Ready for exploratory analysis.")
        
        for rec in recommendations:
            lines.append(f"  {rec}")
        
        lines.append("")
        return lines
    
    def save_report(self, report_text: str, output_path: str) -> None:
        """Save report to file"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_text)
