"""
PDF report (Phase 1: text + tables).

Builds a self-contained analysis report that consolidates what the on-screen
tables and figures do NOT capture: the expert-system **decisions** (which method
was chosen and why), the data validation, a text summary of the key results, the
environmental interpretation summary, and the methodology/parameters used.

Figures are added in Phase 2. Everything goes through ``t()`` so the report
follows the active language; engine-generated text (reasons/evidence) is already
in the run-time language.
"""
from __future__ import annotations

import io
from datetime import datetime

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    Paragraph, Spacer, Table, TableStyle, SimpleDocTemplate, KeepTogether,
)

from app.i18n import t

# Friendly category names (same mapping the Audit page uses). Keys are logic IDs.
_CATEGORY_LABELS = {
    "preprocessing": "Data preparation",
    "dimensionality": "Variable summarization",
    "clustering": "Sample grouping",
    "spatial": "Spatial analysis",
    "anomaly": "Anomaly detection",
    "correlation": "Variable relationships",
    "feature_analysis": "Most informative variables",
}
_CATEGORY_ORDER = list(_CATEGORY_LABELS.keys())

# Toxicological / enrichment class labels -> short, language-neutral display.
_TELPEL_SHORT = {"below_TEL": "< TEL", "TEL_to_PEL": "TEL–PEL", "above_PEL": "> PEL"}

_BRAND = colors.HexColor("#1f6f6b")
_GREY = colors.HexColor("#555555")
_LIGHT = colors.HexColor("#eef4f3")


def _styles():
    ss = getSampleStyleSheet()
    s = {
        "title": ParagraphStyle("t_title", parent=ss["Title"], textColor=_BRAND, fontSize=20, spaceAfter=4),
        "subtitle": ParagraphStyle("t_sub", parent=ss["Normal"], textColor=_GREY, fontSize=10, spaceAfter=14),
        "h1": ParagraphStyle("t_h1", parent=ss["Heading1"], textColor=_BRAND, fontSize=13, spaceBefore=14, spaceAfter=6),
        "h2": ParagraphStyle("t_h2", parent=ss["Heading2"], fontSize=11, spaceBefore=8, spaceAfter=2),
        "body": ParagraphStyle("t_body", parent=ss["Normal"], fontSize=9.5, leading=13, alignment=TA_LEFT),
        "small": ParagraphStyle("t_small", parent=ss["Normal"], fontSize=8.5, leading=11, textColor=_GREY),
        "bullet": ParagraphStyle("t_bul", parent=ss["Normal"], fontSize=9, leading=12, leftIndent=10, bulletIndent=2),
    }
    return s


def _kv_table(rows, st):
    """Two-column label/value table."""
    data = [[Paragraph(f"<b>{k}</b>", st["body"]), Paragraph(str(v), st["body"])] for k, v in rows]
    tbl = Table(data, colWidths=[5.5 * cm, 10.5 * cm], hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), _LIGHT),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.white),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return tbl


def _grid_table(header, rows, st, col_widths=None):
    data = [[Paragraph(f"<b>{h}</b>", st["small"]) for h in header]]
    for r in rows:
        data.append([Paragraph(str(c), st["small"]) for c in r])
    tbl = Table(data, colWidths=col_widths, hAlign="LEFT", repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), _BRAND),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, _LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    return tbl


def _section(flow, st, title):
    flow.append(Spacer(1, 6))
    flow.append(Paragraph(title, st["h1"]))


def _decisions(results, st, flow):
    plan = results.plan
    if plan is None:
        return
    _section(flow, st, t("Analysis decisions"))
    flow.append(Paragraph(
        t("For each step, the method the system chose and the reasoning behind it."),
        st["small"]))
    flow.append(Spacer(1, 4))

    by_cat: dict[str, list] = {}
    for r in plan.recommendations:
        by_cat.setdefault(r.category, []).append(r)

    for cat in _CATEGORY_ORDER:
        recs = by_cat.get(cat)
        if not recs:
            continue
        chosen = sorted(recs, key=lambda r: getattr(r, "priority", 1))[0]
        block = [Paragraph(t(_CATEGORY_LABELS[cat]), st["h2"]),
                 Paragraph(f"<b>{chosen.method}</b>", st["body"])]
        if chosen.reason:
            block.append(Paragraph(chosen.reason, st["body"]))
        for ev in (chosen.evidence or [])[:4]:
            block.append(Paragraph(f"• {ev}", st["bullet"]))
        block.append(Spacer(1, 6))
        flow.append(KeepTogether(block))

    if plan.warnings:
        flow.append(Paragraph(t("Warnings"), st["h2"]))
        for w in plan.warnings:
            flow.append(Paragraph(f"• {w}", st["bullet"]))
        flow.append(Spacer(1, 4))


def _validation(results, st, flow):
    v = results.validation
    if v is None:
        return
    _section(flow, st, t("Data validation"))
    flow.append(_kv_table([
        (t("Completeness"), f"{v.completeness_pct:.1f}%"),
        (t("Issues found"), len(v.issues)),
    ], st))
    if v.issues:
        flow.append(Spacer(1, 4))
        rows = [[i.severity.value.upper(), i.column, i.message] for i in v.issues]
        flow.append(_grid_table([t("Severity"), t("Column"), t("Detail")], rows, st,
                                col_widths=[2.2 * cm, 4 * cm, 9.8 * cm]))


def _key_results(results, st, flow):
    _section(flow, st, t("Key results"))
    rows = []
    dr = results.dim_reduction
    if dr is not None:
        cum = None
        if getattr(dr, "explained_variance", None) is not None and len(dr.explained_variance):
            cum = f"{sum(dr.explained_variance):.1%}"
        rows.append((t("PCA components"), f"{dr.n_components_selected}"
                     + (f" ({t('cumulative variance')} {cum})" if cum else "")))
    cl = results.clustering
    if cl is not None:
        metrics = getattr(cl, "metrics", {}) or {}
        sil = metrics.get("silhouette")
        rows.append((t("Clusters"), f"{cl.n_clusters}"
                     + (f" ({t('Silhouette')} {sil:.3f})" if isinstance(sil, (int, float)) else "")))
    an = results.anomalies
    if an is not None:
        rows.append((t("Anomalies"), f"{an.n_anomalies} ({an.method})"))
    corr = results.correlations
    cr = None
    if isinstance(corr, dict):
        cr = corr.get("pearson") or next(iter(corr.values()), None)
    elif corr is not None:
        cr = corr
    if cr is not None and hasattr(cr, "n_strong"):
        rows.append((t("Significant correlations"),
                     t("{s} strong, {m} moderate").format(s=cr.n_strong, m=cr.n_moderate)))
    if rows:
        flow.append(_kv_table(rows, st))


def _interpretation(results, st, flow):
    interp = results.interpretation
    if interp is None:
        return
    _section(flow, st, t("Environmental interpretation"))
    settings = results.effective_settings or {}
    flow.append(_kv_table([
        (t("Reference element"), settings.get("reference_element", "Al")),
        (t("Baseline strategy"), settings.get("baseline_strategy", "deepest")),
        (t("Metals analyzed"), ", ".join(interp.metals_analyzed)),
    ], st))
    flow.append(Spacer(1, 6))

    # TEL/PEL distribution per metal
    tp = getattr(interp, "tel_pel_classifications", None)
    if tp is not None:
        flow.append(Paragraph(t("Toxicological classification (NOAA TEL/PEL)"), st["h2"]))
        header = [t("Metal"), "< TEL", "TEL–PEL", "> PEL"]
        rows = []
        for metal in tp.columns:
            vc = tp[metal].value_counts()
            rows.append([metal,
                         int(vc.get("below_TEL", 0)),
                         int(vc.get("TEL_to_PEL", 0)),
                         int(vc.get("above_PEL", 0))])
        flow.append(_grid_table(header, rows, st,
                                col_widths=[3 * cm, 3 * cm, 3 * cm, 3 * cm]))
        flow.append(Spacer(1, 6))

    # EF means per metal
    ef = getattr(interp, "ef_result", None)
    if ef is not None and getattr(ef, "ef_values", None) is not None:
        flow.append(Paragraph(t("Enrichment factor (EF) — mean per metal"), st["h2"]))
        means = ef.ef_values.mean(numeric_only=True)
        header = [t("Metal"), t("Mean EF")]
        rows = [[m, f"{means[m]:.2f}"] for m in means.index]
        flow.append(_grid_table(header, rows, st, col_widths=[4 * cm, 4 * cm]))


def _methodology(results, st, flow):
    settings = results.effective_settings or {}
    if not settings:
        return
    _section(flow, st, t("Methodology and parameters"))
    flow.append(Paragraph(
        t("Effective settings used in this run (for reproducibility)."), st["small"]))
    flow.append(Spacer(1, 4))
    # Show scalar settings only (skip nested kwargs dicts).
    keys = ["scale_method", "impute_strategy", "dim_method", "clustering_method",
            "anomaly_method", "correlation_method", "apply_clr", "contamination",
            "reference_element", "baseline_strategy"]
    rows = [(k, settings[k]) for k in keys if k in settings and not isinstance(settings[k], (dict, list))]
    for k, val in settings.items():
        if k not in keys and not isinstance(val, (dict, list)) and val is not None:
            rows.append((k, val))
    if rows:
        flow.append(_kv_table(rows, st))


def build_report_pdf(results, filename: str | None = None) -> bytes:
    """Build the Phase-1 (text + tables) PDF report and return its bytes."""
    st = _styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
        title=t("AEDA-AI analysis report"),
    )
    flow = []

    # --- Cover / metadata ---
    flow.append(Paragraph(t("AEDA-AI analysis report"), st["title"]))
    flow.append(Paragraph(
        t("Automated exploratory data analysis for environmental data"), st["subtitle"]))

    raw = results.raw_data
    proc = results.processed_data
    info = getattr(results.plan, "profile", None) if results.plan else None
    meta = [
        (t("Dataset"), filename or "—"),
        (t("Generated"), datetime.now().strftime("%Y-%m-%d %H:%M")),
        (t("Samples"), raw.shape[0] if raw is not None else "—"),
        (t("Measurement variables"), proc.shape[1] if proc is not None else "—"),
    ]
    if info is not None:
        if getattr(info, "n_sites", 0):
            meta.append((t("Sites"), info.n_sites))
        if getattr(info, "has_depth", False):
            meta.append((t("Depth column"), t("yes")))
    flow.append(_kv_table(meta, st))

    _decisions(results, st, flow)
    _validation(results, st, flow)
    _key_results(results, st, flow)
    _interpretation(results, st, flow)
    _methodology(results, st, flow)

    flow.append(Spacer(1, 16))
    flow.append(Paragraph(
        t("Report generated by AEDA-AI. Figures are available in the application."),
        st["small"]))

    doc.build(flow)
    return buf.getvalue()
