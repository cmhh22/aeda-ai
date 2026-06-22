"""
Export helpers: collect the result tables and serialize them to Excel/CSV.

The engine stores results as several DataFrames spread across sub-objects
(PCA loadings, correlation matrices, EF values, TEL/PEL classification, etc.).
``collect_tables`` gathers every table that is actually present into a single
ordered mapping of ``{display_name: DataFrame}``. The display names go through
``t()`` so they (and the Excel sheet names) follow the active UI language.

Only tables that exist are included — a run without interpretation simply omits
the EF/TEL-PEL sheets. Nothing here touches the analysis itself; it is a pure
read-and-serialize layer.
"""
from __future__ import annotations

import io
import re

import pandas as pd

from app.i18n import t


def _per_sample_summary(results) -> pd.DataFrame | None:
    """Per-sample table: cluster label + anomaly flag/score, aligned to the
    analyzed samples. Each column is added only when its length matches."""
    idx = None
    if results.processed_data is not None:
        idx = results.processed_data.index
    elif results.raw_data is not None:
        idx = results.raw_data.index
    if idx is None:
        return None

    n = len(idx)
    df = pd.DataFrame(index=idx)

    cl = results.clustering
    if cl is not None and getattr(cl, "labels", None) is not None and len(cl.labels) == n:
        df[t("Cluster")] = cl.labels

    an = results.anomalies
    if an is not None and getattr(an, "is_anomaly", None) is not None and len(an.is_anomaly) == n:
        df[t("Anomaly")] = an.is_anomaly
        if getattr(an, "scores", None) is not None and len(an.scores) == n:
            df[t("Anomaly score")] = an.scores

    return df if not df.empty else None


def collect_tables(results) -> dict[str, pd.DataFrame]:
    """Return an ordered {display_name: DataFrame} of every available table."""
    tables: dict[str, pd.DataFrame] = {}

    if results.raw_data is not None:
        tables[t("Raw data")] = results.raw_data
    if results.processed_data is not None:
        tables[t("Processed data")] = results.processed_data

    summary = _per_sample_summary(results)
    if summary is not None:
        tables[t("Sample classification")] = summary

    dr = results.dim_reduction
    if dr is not None:
        if getattr(dr, "loadings", None) is not None:
            tables[t("PCA loadings")] = dr.loadings
        if getattr(dr, "components", None) is not None:
            tables[t("PCA coordinates")] = dr.components
        ev = getattr(dr, "explained_variance", None)
        if ev is not None:
            tables[t("PCA explained variance")] = pd.DataFrame({
                t("Component"): [f"PC{i + 1}" for i in range(len(ev))],
                t("Explained variance"): list(ev),
            })

    corr = results.correlations
    if isinstance(corr, dict):
        for method, cr in corr.items():
            matrix = getattr(cr, "matrix", None)
            if matrix is not None:
                tables[t("Correlation ({m})").format(m=method)] = matrix
    elif corr is not None and getattr(corr, "matrix", None) is not None:
        tables[t("Correlation matrix")] = corr.matrix

    fi = results.feature_importance
    if fi is not None and getattr(fi, "importances", None) is not None:
        tables[t("Feature importance")] = fi.importances.to_frame(name=t("Importance"))

    interp = results.interpretation
    if interp is not None:
        if getattr(interp, "tel_pel_classifications", None) is not None:
            tables[t("TEL/PEL classification")] = interp.tel_pel_classifications
        if getattr(interp, "ef_classifications", None) is not None:
            tables[t("EF classification")] = interp.ef_classifications
        ef = getattr(interp, "ef_result", None)
        if ef is not None and getattr(ef, "ef_values", None) is not None:
            tables[t("Enrichment factors (EF)")] = ef.ef_values

    sa = results.surface_analysis
    if sa is not None:
        if getattr(sa, "site_means", None) is not None:
            tables[t("Surface site means")] = sa.site_means
        if getattr(sa, "site_coordinates", None) is not None:
            tables[t("Site coordinates")] = sa.site_coordinates

    return tables


def _safe_sheet_name(name: str, used: set) -> str:
    """Excel sheet names: <=31 chars, none of : \\ / ? * [ ], and unique."""
    s = re.sub(r"[:\\/?*\[\]]", "-", str(name))[:31] or "Sheet"
    base, i = s, 1
    while s in used:
        suffix = f"_{i}"
        s = base[: 31 - len(suffix)] + suffix
        i += 1
    used.add(s)
    return s


def to_excel_bytes(tables: dict[str, pd.DataFrame]) -> bytes:
    """Serialize all tables into one .xlsx workbook (one sheet per table)."""
    buf = io.BytesIO()
    used: set = set()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in tables.items():
            df.to_excel(writer, sheet_name=_safe_sheet_name(name, used), index=True)
    return buf.getvalue()


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    # utf-8-sig so Excel opens accented characters correctly.
    return df.to_csv(index=True).encode("utf-8-sig")


def safe_filename(name: str, ext: str) -> str:
    slug = re.sub(r"[^\w\-]+", "_", str(name)).strip("_").lower() or "tabla"
    return f"{slug}.{ext}"
