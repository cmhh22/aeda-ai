"""
Theme module — central source of truth for the AEDA-AI visual identity.

The Streamlit-level theme (primary color, background, font) is configured in
.streamlit/config.toml. This module adds:

1. Named palette constants importable from anywhere (Plotly figures, custom
   widgets, etc.) so colors stay consistent across the app.
2. A small CSS layer applied via `apply_theme()` that refines typography,
   spacing, sidebar styling, metric cards and tabs without fighting the
   Streamlit base theme.

The palette is built around an environmental-sediment metaphor:
- Olive green (vegetation, sediment surface) as primary brand color.
- Ocean blue (aquatic environment) as secondary / informational color.
- Terracotta (warning, anomaly) as accent / alert color.
- Warm off-white background that reads as paper, not as a software UI.
"""

from __future__ import annotations

import streamlit as st


# =============================================================================
# Palette — earth-science scheme
# =============================================================================

# Primary — olive green family (sediment, vegetation)
PRIMARY = "#5C7548"
PRIMARY_DARK = "#3D4F2F"
PRIMARY_LIGHT = "#8BA177"

# Secondary — ocean blue family (aquatic environment, information)
OCEAN = "#2E5266"
OCEAN_DARK = "#1F3845"
OCEAN_LIGHT = "#5B7E94"

# Accent — terracotta family (warning, anomaly, attention)
TERRACOTTA = "#A0522D"
TERRACOTTA_LIGHT = "#C97B2C"

# Neutral grays — warm tone, not pure gray
BACKGROUND = "#FAF8F4"
SURFACE = "#FFFFFF"
SIDEBAR_BG = "#F0EDE5"
BORDER = "#E5E2DA"
TEXT_PRIMARY = "#2A2A2A"
TEXT_SECONDARY = "#5C5C5C"
TEXT_MUTED = "#8A8580"

# Semantic
SUCCESS = PRIMARY
WARNING = TERRACOTTA_LIGHT
ERROR = TERRACOTTA
INFO = OCEAN

# Categorical palette for plots (cluster colors, group comparisons, etc.)
# Designed to be distinguishable for color-blind users and to stay visually
# coherent with the earth-science scheme.
CATEGORICAL = [
    "#5C7548",   # olive
    "#2E5266",   # ocean
    "#A0522D",   # terracotta
    "#8BA177",   # light olive
    "#5B7E94",   # light ocean
    "#C97B2C",   # light terracotta
    "#3D4F2F",   # deep olive
    "#1F3845",   # deep ocean
]


# =============================================================================
# CSS layer
# =============================================================================

_CSS = f"""
<style>
/* ---------- Layout ---------- */
.block-container {{
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}}

/* ---------- Typography ---------- */
h1, h2, h3, h4 {{
    font-weight: 600;
    color: {TEXT_PRIMARY};
    letter-spacing: -0.01em;
}}
h1 {{ font-size: 1.875rem; margin-bottom: 0.25rem; }}
h2 {{ font-size: 1.375rem; margin-top: 1.75rem; margin-bottom: 0.5rem; }}
h3 {{ font-size: 1.125rem; margin-top: 1.25rem; color: {TEXT_SECONDARY}; }}

/* Captions read as supportive subtitles, not afterthoughts */
[data-testid="stCaptionContainer"] {{
    color: {TEXT_SECONDARY};
    font-size: 0.875rem;
    line-height: 1.45;
}}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {{
    background-color: {SIDEBAR_BG};
    border-right: 1px solid {BORDER};
}}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {{
    color: {TEXT_PRIMARY};
}}

/* Sidebar branding row: 🔬 + AEDA-AI */
.sidebar-brand {{
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.25rem 0 0.5rem 0;
}}
.sidebar-brand .brand-icon {{
    font-size: 1.5rem;
    line-height: 1;
}}
.sidebar-brand .brand-name {{
    font-size: 1.375rem;
    font-weight: 700;
    color: {PRIMARY_DARK};
    letter-spacing: -0.02em;
}}

/* Sidebar status block */
.status-empty {{
    color: {TEXT_MUTED};
    font-size: 0.875rem;
    font-style: italic;
    padding: 0.5rem 0;
}}

/* ---------- Metric cards ---------- */
[data-testid="stMetric"] {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    padding: 0.75rem 1rem;
    border-radius: 6px;
}}
[data-testid="stMetricLabel"] {{
    color: {TEXT_SECONDARY};
    font-size: 0.8125rem;
    font-weight: 500;
}}
[data-testid="stMetricValue"] {{
    font-weight: 600;
    color: {TEXT_PRIMARY};
}}

/* ---------- Buttons ---------- */
.stButton > button {{
    border-radius: 6px;
    font-weight: 500;
    border: 1px solid {BORDER};
    transition: all 0.15s ease;
}}
.stButton > button[kind="primary"] {{
    background-color: {PRIMARY};
    border-color: {PRIMARY};
}}
.stButton > button[kind="primary"]:hover {{
    background-color: {PRIMARY_DARK};
    border-color: {PRIMARY_DARK};
}}

/* ---------- Expanders ---------- */
[data-testid="stExpander"] {{
    border: 1px solid {BORDER};
    border-radius: 6px;
    background-color: {SURFACE};
}}
[data-testid="stExpander"] summary {{
    font-weight: 500;
}}

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] {{
    gap: 1rem;
}}
.stTabs [data-baseweb="tab"] {{
    font-weight: 500;
}}
.stTabs [data-baseweb="tab"][aria-selected="true"] {{
    color: {PRIMARY};
}}
.stTabs [data-baseweb="tab-highlight"] {{
    background-color: {PRIMARY};
}}

/* ---------- Dividers (less aggressive) ---------- */
hr {{
    border-color: {BORDER};
    margin: 1.5rem 0;
}}

/* ---------- Page header component (used by app/components/page_header.py) ---------- */
.page-header-row {{
    display: flex;
    align-items: center;
    gap: 0.625rem;
    margin-bottom: 0.25rem;
}}
.page-header-row .page-header-icon {{
    font-size: 1.625rem;
    line-height: 1;
}}
.page-header-row h1 {{
    margin: 0;
    font-size: 1.875rem;
}}

/* ---------- Reduce visual noise from default progress bar ---------- */
[data-testid="stProgress"] > div > div {{
    background-color: {PRIMARY};
}}
</style>
"""


def apply_theme() -> None:
    """Inject the CSS layer. Call once near the top of main.py, after
    `st.set_page_config(...)` but before any page rendering.
    """
    st.markdown(_CSS, unsafe_allow_html=True)
