"""
Report figures (Phase 2).

Regenerates the key figures as static PNGs with matplotlib (Agg backend) so they
can be embedded in the PDF report. We use matplotlib rather than exporting the
app's Plotly charts because it needs no extra/heavy dependency (no kaleido/
chromium) and is reliable on Streamlit Cloud. The figures are clean, print-
oriented versions of what the app shows interactively.

Each function returns PNG bytes, or ``None`` when the required data is absent.
"""
from __future__ import annotations

import io

import matplotlib
matplotlib.use("Agg")  # headless, server-safe
import matplotlib.pyplot as plt
import numpy as np

from app.i18n import t

_CMAP_CLUSTER = "tab10"


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def _pc_label(results, i: int) -> str:
    ev = getattr(results.dim_reduction, "explained_variance", None)
    if ev is not None and len(ev) > i:
        return f"PC{i + 1} ({ev[i]:.0%})"
    return f"PC{i + 1}"


def pca_biplot_png(results) -> bytes | None:
    dr = results.dim_reduction
    if dr is None or getattr(dr, "components", None) is None:
        return None
    comp = dr.components
    if comp.shape[1] < 2:
        return None
    x, y = comp.iloc[:, 0].values, comp.iloc[:, 1].values

    fig, ax = plt.subplots(figsize=(7, 5))
    labels = getattr(results.clustering, "labels", None)
    if labels is not None and len(labels) == len(x):
        sc = ax.scatter(x, y, c=labels, cmap=_CMAP_CLUSTER, s=18, alpha=0.8,
                        edgecolors="white", linewidths=0.3)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(t("Cluster"))
    else:
        ax.scatter(x, y, s=18, alpha=0.8, color="#1f6f6b")

    # Top loading arrows
    load = getattr(dr, "loadings", None)
    if load is not None and load.shape[1] >= 2:
        mag = np.sqrt(load.iloc[:, 0] ** 2 + load.iloc[:, 1] ** 2)
        top = mag.sort_values(ascending=False).head(8).index
        scale = 0.9 * max(np.abs(x).max(), np.abs(y).max()) / float(mag.max() or 1)
        for var in top:
            dx, dy = load.loc[var].iloc[0] * scale, load.loc[var].iloc[1] * scale
            ax.arrow(0, 0, dx, dy, color="#c0392b", alpha=0.7,
                     head_width=0.05 * scale, length_includes_head=True, linewidth=0.8)
            ax.text(dx * 1.08, dy * 1.08, str(var), color="#922b21", fontsize=7,
                    ha="center", va="center")

    ax.axhline(0, color="#cccccc", lw=0.6)
    ax.axvline(0, color="#cccccc", lw=0.6)
    ax.set_xlabel(_pc_label(results, 0))
    ax.set_ylabel(_pc_label(results, 1))
    ax.set_title(t("PCA biplot (PC1 vs PC2)"))
    fig.tight_layout()
    return _png(fig)


def correlation_heatmap_png(results) -> bytes | None:
    corr = results.correlations
    cr = None
    if isinstance(corr, dict):
        cr = corr.get("pearson") or next(
            (v for k, v in corr.items() if hasattr(v, "matrix")), None)
    elif corr is not None:
        cr = corr
    if cr is None or getattr(cr, "matrix", None) is None:
        return None
    m = cr.matrix
    n = m.shape[0]

    fig, ax = plt.subplots(figsize=(min(0.28 * n + 1.5, 8), min(0.28 * n + 1.2, 7.5)))
    im = ax.imshow(m.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fs = 6 if n > 25 else 8
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(m.columns, rotation=90, fontsize=fs)
    ax.set_yticklabels(m.index, fontsize=fs)
    ax.set_title(t("Correlation matrix (Pearson)"))
    fig.tight_layout()
    return _png(fig)


def anomalies_png(results) -> bytes | None:
    dr = results.dim_reduction
    an = results.anomalies
    if dr is None or getattr(dr, "components", None) is None or an is None:
        return None
    comp = dr.components
    if comp.shape[1] < 2 or getattr(an, "is_anomaly", None) is None:
        return None
    flags = np.asarray(an.is_anomaly)
    if len(flags) != comp.shape[0]:
        return None
    x, y = comp.iloc[:, 0].values, comp.iloc[:, 1].values

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x[~flags], y[~flags], s=16, alpha=0.6, color="#7f8c8d", label=t("Normal"))
    ax.scatter(x[flags], y[flags], s=42, marker="X", color="#c0392b",
               edgecolors="black", linewidths=0.4, label=t("Anomaly"))
    ax.axhline(0, color="#cccccc", lw=0.6)
    ax.axvline(0, color="#cccccc", lw=0.6)
    ax.set_xlabel(_pc_label(results, 0))
    ax.set_ylabel(_pc_label(results, 1))
    ax.set_title(t("Anomalies in PCA space"))
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return _png(fig)


def ef_bar_png(results) -> bytes | None:
    interp = results.interpretation
    if interp is None or getattr(interp, "ef_result", None) is None:
        return None
    ef = interp.ef_result
    if getattr(ef, "ef_values", None) is None:
        return None
    means = ef.ef_values.mean(numeric_only=True)
    if means.empty:
        return None

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#c0392b" if v >= 2 else ("#e67e22" if v >= 1.5 else "#1f6f6b") for v in means.values]
    ax.bar([str(m) for m in means.index], means.values, color=colors)
    ax.axhline(1.0, color="#555555", lw=0.8, linestyle="--")
    ax.text(len(means) - 0.5, 1.02, "EF = 1", fontsize=7, color="#555555", ha="right")
    ax.set_ylabel(t("Mean EF"))
    ax.set_title(t("Enrichment factor (EF) per metal"))
    fig.tight_layout()
    return _png(fig)


def all_figures(results) -> list[tuple[str, bytes]]:
    """Return [(caption, png_bytes)] for every figure that could be built."""
    out = []
    for caption_key, fn in [
        ("PCA biplot (PC1 vs PC2)", pca_biplot_png),
        ("Correlation matrix (Pearson)", correlation_heatmap_png),
        ("Anomalies in PCA space", anomalies_png),
        ("Enrichment factor (EF) per metal", ef_bar_png),
    ]:
        try:
            png = fn(results)
        except Exception:
            png = None
        if png:
            out.append((t(caption_key), png))
    return out
