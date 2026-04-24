"""
aeda.io.validators
Automatic environmental data quality validation.
Detects common issues: out-of-range values, missing data
with structured patterns, compositions that do not sum to 100%, etc.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class Severity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class ValidationIssue:
    column: str
    severity: Severity
    message: str
    affected_rows: int = 0
    details: Optional[dict] = None


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)
    n_rows: int = 0
    n_cols: int = 0
    completeness_pct: float = 0.0

    @property
    def has_errors(self) -> bool:
        return any(i.severity == Severity.ERROR for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.severity == Severity.WARNING for i in self.issues)

    def summary(self) -> str:
        lines = [
            f"Validation report: {self.n_rows} rows × {self.n_cols} columns",
            f"Overall completeness: {self.completeness_pct:.1f}%",
            f"Issues found: {len(self.issues)} "
            f"({sum(1 for i in self.issues if i.severity == Severity.ERROR)} errors, "
            f"{sum(1 for i in self.issues if i.severity == Severity.WARNING)} warnings, "
            f"{sum(1 for i in self.issues if i.severity == Severity.INFO)} info)",
        ]
        for issue in self.issues:
            lines.append(f"  [{issue.severity.value.upper()}] {issue.column}: {issue.message}")
        return "\n".join(lines)


def _check_missing_pattern(df: pd.DataFrame) -> list[ValidationIssue]:
    """Detect whether missing values follow a structured (non-random) pattern."""
    issues = []
    null_cols = [c for c in df.columns if df[c].isnull().any()]
    if not null_cols:
        return issues

    null_mask = df[null_cols].isnull()
    null_patterns = null_mask.drop_duplicates()

    if len(null_patterns) <= 3 and len(null_cols) > 1:
        # Missing values are grouped in blocks, likely by experimental design
        for _, pattern in null_patterns.iterrows():
            missing_in = [c for c in null_cols if pattern[c]]
            if missing_in:
                n_rows = null_mask[null_mask[missing_in].all(axis=1)].shape[0]
                issues.append(ValidationIssue(
                    column=", ".join(missing_in),
                    severity=Severity.INFO,
                    message=(
                        f"Structured missing-data pattern detected: "
                        f"{n_rows} rows with these columns empty simultaneously. "
                        f"Likely unmeasured values by experimental design."
                    ),
                    affected_rows=n_rows,
                ))
    else:
        for col in null_cols:
            n_null = df[col].isnull().sum()
            pct = n_null / len(df) * 100
            severity = Severity.WARNING if pct > 20 else Severity.INFO
            issues.append(ValidationIssue(
                column=col,
                severity=severity,
                message=f"{n_null} missing values ({pct:.1f}%)",
                affected_rows=n_null,
            ))

    return issues


def _check_negative_concentrations(df: pd.DataFrame, measurement_cols: list[str]) -> list[ValidationIssue]:
    """Chemical concentrations should not be negative."""
    issues = []
    for col in measurement_cols:
        if col in df.columns and df[col].dtype in ("float64", "int64"):
            n_neg = (df[col] < 0).sum()
            if n_neg > 0:
                issues.append(ValidationIssue(
                    column=col,
                    severity=Severity.ERROR,
                    message=f"{n_neg} negative concentration values detected",
                    affected_rows=n_neg,
                ))
    return issues


def _check_composition_closure(df: pd.DataFrame) -> list[ValidationIssue]:
    """
    Check whether granulometric fractions sum to ~100%.
    In compositional datasets (FRX), this checks closure.
    """
    issues = []

    # Granulometry: look for fraction columns
    gran_patterns = [
        ("< 2", "2 < G < 63", "> 63"),       # ISOVIDA format
        ("clay", "silt", "sand"),
        ("arcilla", "limo", "arena"),
    ]

    for patterns in gran_patterns:
        matched = []
        for p in patterns:
            for col in df.columns:
                if p.lower() in col.lower() and "u_" not in col.lower():
                    matched.append(col)
                    break

        if len(matched) == 3:
            valid_rows = df[matched].dropna()
            if len(valid_rows) > 0:
                sums = valid_rows[matched].sum(axis=1)
                bad = ((sums < 95) | (sums > 105))
                n_bad = bad.sum()
                if n_bad > 0:
                    issues.append(ValidationIssue(
                        column=", ".join(matched),
                        severity=Severity.WARNING,
                        message=(
                            f"Granulometric fractions do not sum to ~100% in {n_bad} rows. "
                            f"Range: {sums.min():.1f}% - {sums.max():.1f}%"
                        ),
                        affected_rows=n_bad,
                        details={"min_sum": sums.min(), "max_sum": sums.max(), "mean_sum": sums.mean()},
                    ))
            break

    return issues


def _check_outliers_iqr(df: pd.DataFrame, measurement_cols: list[str], factor: float = 3.0) -> list[ValidationIssue]:
    """Detect extreme outliers using IQR × factor."""
    issues = []
    for col in measurement_cols:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if len(s) < 10:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - factor * iqr, q3 + factor * iqr
        n_out = ((s < lower) | (s > upper)).sum()
        if n_out > 0:
            issues.append(ValidationIssue(
                column=col,
                severity=Severity.INFO,
                message=f"{n_out} extreme outliers (IQR×{factor})",
                affected_rows=n_out,
                details={"lower_bound": lower, "upper_bound": upper},
            ))
    return issues


def _check_constant_columns(df: pd.DataFrame) -> list[ValidationIssue]:
    """Detect columns with zero or near-zero variance."""
    issues = []
    for col in df.select_dtypes(include="number").columns:
        nunique = df[col].nunique()
        if nunique <= 1:
            issues.append(ValidationIssue(
                column=col,
                severity=Severity.WARNING,
                message="Constant column (variance = 0), will be excluded from analysis",
                affected_rows=len(df),
            ))
        elif nunique <= 3:
            issues.append(ValidationIssue(
                column=col,
                severity=Severity.INFO,
                message=f"Column with very low variability ({nunique} unique values)",
                affected_rows=len(df),
            ))
    return issues


def validate(
    df: pd.DataFrame,
    measurement_cols: Optional[list[str]] = None,
) -> ValidationReport:
    """
    Run all validation checks on an environmental data DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    measurement_cols : list[str], optional
        Numeric measurement columns. If None, all numeric columns are used.

    Returns
    -------
    ValidationReport
        Report containing all detected issues.
    """
    if measurement_cols is None:
        measurement_cols = df.select_dtypes(include="number").columns.tolist()

    report = ValidationReport(
        n_rows=len(df),
        n_cols=len(df.columns),
        completeness_pct=(1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
    )

    report.issues.extend(_check_missing_pattern(df))
    report.issues.extend(_check_negative_concentrations(df, measurement_cols))
    report.issues.extend(_check_composition_closure(df))
    report.issues.extend(_check_outliers_iqr(df, measurement_cols))
    report.issues.extend(_check_constant_columns(df))

    return report
