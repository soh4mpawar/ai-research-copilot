"""
Dynamic Theme Engine (Light & Scientific Dark Mode).
Provides color tokens and CSS injection based on session state.
"""

import streamlit as st


def is_dark_mode() -> bool:
    """Return True if Dark Mode is active in session state."""
    return bool(st.session_state.get("dark_mode", False))


def get_theme_colors():
    """Return color dictionary matching current theme state."""
    dark = is_dark_mode()
    if dark:
        return {
            "bg": "#16181C",
            "card_bg": "#1E2126",
            "border": "#2E3238",
            "text_primary": "#F3F4F6",
            "text_secondary": "#9CA3AF",
            "accent": "#5B7FB5",
            "accent_border": "#3B6199",
            "grid": "#2E3238",
            "badge_outline_bg": "#1E2126",
            "badge_outline_text": "#E2E8F0",
            "badge_outline_border": "#4B5563",
            "badge_strong_bg": "#143521",
            "badge_strong_text": "#4ADE80",
            "badge_strong_border": "#1E5631",
            "badge_moderate_bg": "#3A230B",
            "badge_moderate_text": "#FBBF24",
            "badge_moderate_border": "#854D0E",
            "citation_chip_bg": "#262A30",
            "citation_chip_text": "#E2E8F0",
            "citation_chip_border": "#4B5563",
            "progress_bar_fill": "#1E5631",
            "progress_bar_track": "#2E3238",
            "plotly_template": "plotly_dark",
            "pyvis_bg": "#16181C",
            "pyvis_font": "#E8E8E6",
            "pyvis_edge": "#3E444E",
            "pyvis_node_primary": "#5B7FB5",
            "pyvis_node_secondary": "#94A3B8",
        }
    else:
        return {
            "bg": "#FAFAF9",
            "card_bg": "#FFFFFF",
            "border": "#E5E5E3",
            "text_primary": "#111827",
            "text_secondary": "#525252",
            "accent": "#2B4C7E",
            "accent_border": "#24416C",
            "grid": "#E5E7EB",
            "badge_outline_bg": "#F3F4F6",
            "badge_outline_text": "#1F2937",
            "badge_outline_border": "#9CA3AF",
            "badge_strong_bg": "#EBF5EE",
            "badge_strong_text": "#1E5631",
            "badge_strong_border": "#C4E3CB",
            "badge_moderate_bg": "#FEF3C7",
            "badge_moderate_text": "#854D0E",
            "badge_moderate_border": "#FDE68A",
            "citation_chip_bg": "#F3F4F6",
            "citation_chip_text": "#1F2937",
            "citation_chip_border": "#D1D5DB",
            "progress_bar_fill": "#1E5631",
            "progress_bar_track": "#E5E7EB",
            "plotly_template": "plotly_white",
            "pyvis_bg": "#FAFAF9",
            "pyvis_font": "#111827",
            "pyvis_edge": "#CBD5E1",
            "pyvis_node_primary": "#2B4C7E",
            "pyvis_node_secondary": "#64748B",
        }


def inject_theme():
    """Inject active theme CSS into Streamlit page."""
    if is_dark_mode():
        st.markdown(
            """
            <style>
            /* Base Canvas & Streamlit Header */
            .stApp {
                background-color: #16181C !important;
                color: #E8E8E6 !important;
            }
            header[data-testid="stHeader"],
            [data-testid="stHeader"] {
                background: #16181C !important;
                background-color: #16181C !important;
            }
            [data-testid="stToolbar"] *,
            [data-testid="stHeader"] * {
                color: #E8E8E6 !important;
            }

            /* Sidebar All Elements */
            [data-testid="stSidebar"] {
                background-color: #1A1D21 !important;
                border-right: 1px solid #2E3238 !important;
            }
            [data-testid="stSidebar"] div,
            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] span,
            [data-testid="stSidebar"] p,
            [data-testid="stSidebar"] [data-testid="stRadio"] label,
            [data-testid="stSidebar"] [data-testid="stRadio"] label p,
            [data-testid="stSidebar"] [data-testid="stRadio"] label span,
            [data-testid="stSidebar"] [data-testid="stRadio"] label div,
            [data-testid="stSidebar"] .stMarkdown p,
            [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
            [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
            [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
            [data-testid="stSidebar"] [data-testid="stWidgetLabel"] span {
                color: #E8E8E6 !important;
            }
            [data-testid="stSidebar"] code {
                background-color: #262A30 !important;
                color: #E2E8F0 !important;
                border: 1px solid #3E444E !important;
            }

            /* Main Canvas Markdown & Text */
            [data-testid="stMarkdownContainer"] p,
            [data-testid="stMarkdownContainer"] span,
            [data-testid="stMarkdownContainer"] li {
                color: #E8E8E6 !important;
            }

            /* Cards & Panels */
            .glass-card, .source-card, .metric-box {
                background: #1E2126 !important;
                border: 1px solid #2E3238 !important;
                color: #E8E8E6 !important;
            }
            .hero-title, .academic-title, .source-title, h1, h2, h3, h4, h5, h6,
            [data-testid="stMarkdownContainer"] h1,
            [data-testid="stMarkdownContainer"] h2,
            [data-testid="stMarkdownContainer"] h3,
            [data-testid="stMarkdownContainer"] h4 {
                color: #F3F4F6 !important;
            }
            .hero-subtitle, .source-meta, .metric-lbl {
                color: #9CA3AF !important;
            }
            .metric-val {
                color: #F3F4F6 !important;
            }

            /* Labels & Form Widgets */
            [data-testid="stWidgetLabel"] p,
            [data-testid="stWidgetLabel"] span,
            [data-testid="stWidgetLabel"] div,
            label[data-testid="stWidgetLabel"] p,
            label[data-testid="stWidgetLabel"] span,
            label[data-baseweb="checkbox"] p,
            label[data-baseweb="checkbox"] span,
            label[data-baseweb="checkbox"] div {
                color: #E8E8E6 !important;
            }
            input[type="text"], textarea, [data-baseweb="input"] input {
                background-color: #1E2126 !important;
                color: #F3F4F6 !important;
                border: 1px solid #2E3238 !important;
            }
            [data-baseweb="select"] div,
            [data-baseweb="select"] span {
                background-color: #1E2126 !important;
                color: #F3F4F6 !important;
                border-color: #2E3238 !important;
            }

            /* Badges */
            .badge-outline {
                background: #1E2126 !important;
                color: #E2E8F0 !important;
                border: 1px solid #4B5563 !important;
            }
            .badge-slate {
                background: #262A30 !important;
                color: #CBD5E1 !important;
                border: 1px solid #3E444E !important;
            }
            .badge-strong {
                background: #143521 !important;
                color: #4ADE80 !important;
                border: 1px solid #1E5631 !important;
            }
            .badge-moderate {
                background: #3A230B !important;
                color: #FBBF24 !important;
                border: 1px solid #854D0E !important;
            }

            /* Citation Chips */
            [data-testid="stMarkdownContainer"] a.citation-chip, a.citation-chip {
                background: #262A30 !important;
                color: #E2E8F0 !important;
                border: 1px solid #4B5563 !important;
            }
            [data-testid="stMarkdownContainer"] a.citation-chip:hover, a.citation-chip:hover {
                background: #323842 !important;
                color: #FFFFFF !important;
                border-color: #6B7280 !important;
            }

            /* Evidence Cards */
            .evidence-card {
                background: #1E2126 !important;
                border: 1px solid #2E3238 !important;
                border-left: 3px solid #5B7FB5 !important;
            }
            .evidence-text {
                background: #16181C !important;
                color: #E2E8F0 !important;
                border: 1px solid #2E3238 !important;
            }
            .evidence-header {
                color: #9CA3AF !important;
            }

            /* Expanders & Tabs */
            [data-testid="stExpander"] {
                background: #1E2126 !important;
                border: 1px solid #2E3238 !important;
                color: #E8E8E6 !important;
            }
            [data-testid="stTabs"] [data-baseweb="tab-list"] {
                border-bottom: 1px solid #2E3238 !important;
            }
            [data-testid="stTabs"] [data-baseweb="tab"] p,
            [data-testid="stTabs"] [data-baseweb="tab"] span,
            [data-testid="stTabs"] [data-baseweb="tab"] {
                color: #9CA3AF !important;
            }
            [data-testid="stTabs"] [aria-selected="true"] p,
            [data-testid="stTabs"] [aria-selected="true"] span,
            [data-testid="stTabs"] [aria-selected="true"] {
                color: #7AA2DC !important;
                border-bottom: 2px solid #7AA2DC !important;
            }

            /* Buttons */
            button[kind="primary"] {
                background-color: #3B6199 !important;
                color: #FFFFFF !important;
                border: 1px solid #4A73B0 !important;
            }
            button[kind="primary"] p,
            button[kind="primary"] span,
            button[kind="primary"] div {
                color: #FFFFFF !important;
            }
            button[kind="primary"]:hover {
                background-color: #4A73B0 !important;
            }
            button[kind="secondary"] {
                background-color: #1E2126 !important;
                border: 1px solid #3E444E !important;
            }
            button[kind="secondary"] p,
            button[kind="secondary"] span,
            button[kind="secondary"] div {
                color: #E8E8E6 !important;
            }
            button[kind="secondary"]:hover {
                background-color: #262A30 !important;
                border-color: #4B5563 !important;
            }

            /* Dividers */
            hr {
                border-color: #2E3238 !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
