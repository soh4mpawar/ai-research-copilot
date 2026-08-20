"""
Dynamic Theme Engine (Light & Scientific Dark Mode).
Provides color tokens and comprehensive CSS injection targeting live DOM structures.
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
            /* ====================================================================
               1. FULL-VIEWPORT DARK ROOT & CHROME OVERRIDES
               ==================================================================== */
            :root,
            html,
            body,
            .stApp,
            [data-testid="stAppViewContainer"],
            section.main {
                background-color: #16181C !important;
                background: #16181C !important;
                color: #E8E8E6 !important;
                --background-color: #16181C !important;
                --secondary-background-color: #1E2126 !important;
                --text-color: #F3F4F6 !important;
                --primary-color: #5B7FB5 !important;
                --border-color: #2E3238 !important;
            }

            /* Top Streamlit Header / AppToolbar Strip */
            header,
            .stAppHeader,
            [data-testid="stHeader"],
            header.stAppHeader,
            header[data-testid="stHeader"],
            [data-testid="stAppHeader"],
            [data-testid="stToolbar"] {
                background-color: #16181C !important;
                background: #16181C !important;
                color: #E8E8E6 !important;
            }

            header *, [data-testid="stHeader"] * {
                color: #E8E8E6 !important;
            }

            /* ====================================================================
               2. SIDEBAR FULL DARK STYLING (Modern Nav Rows & Icon Integration)
               ==================================================================== */
            [data-testid="stSidebar"],
            [data-testid="stSidebarContent"],
            [data-testid="stSidebarUserContent"],
            section[data-testid="stSidebar"] {
                background-color: #1A1D21 !important;
                background: #1A1D21 !important;
                border-right: 1px solid #2E3238 !important;
            }

            /* Section Headers */
            .sidebar-section-header {
                color: #9CA3AF !important;
            }

            /* Info Card */
            .sidebar-info-card {
                background: #1E2126 !important;
                border: 1px solid #2E3238 !important;
                color: #E8E8E6 !important;
            }

            /* Footer */
            .sidebar-footer {
                border-top: 1px solid #2E3238 !important;
            }

            .sidebar-repo-link {
                color: #7AA2DC !important;
            }

            /* Universal Sidebar Text & Radio Label Colors */
            [data-testid="stSidebar"] p,
            [data-testid="stSidebar"] span,
            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] div,
            [data-testid="stSidebar"] [role="radiogroup"] *,
            [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
            [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
            [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
            [data-testid="stSidebar"] [data-testid="stWidgetLabel"] span {
                color: #E8E8E6 !important;
            }

            /* Custom Nav-List Style in Dark Mode */
            [data-testid="stSidebar"] [role="radiogroup"] label[data-testid="stRadioOption"] {
                background: transparent !important;
                border-left: 3px solid transparent !important;
            }

            [data-testid="stSidebar"] [role="radiogroup"] label[data-testid="stRadioOption"] > div,
            [data-testid="stSidebar"] [role="radiogroup"] label[data-testid="stRadioOption"] > div > div,
            [data-testid="stSidebar"] [role="radiogroup"] label[data-testid="stRadioOption"] [data-testid="stMarkdownContainer"] {
                display: flex !important;
                align-items: center !important;
                width: 100% !important;
            }

            [data-testid="stSidebar"] [role="radiogroup"] label[data-testid="stRadioOption"][data-selected="true"] {
                background-color: #262A30 !important;
                border-left: 3px solid #5B7FB5 !important;
            }

            [data-testid="stSidebar"] [role="radiogroup"] label[data-testid="stRadioOption"][data-selected="true"] p {
                color: #FFFFFF !important;
                font-weight: 600 !important;
            }

            [data-testid="stSidebar"] [role="radiogroup"] label[data-testid="stRadioOption"]:not([data-selected="true"]):hover {
                background-color: #1E2126 !important;
            }

            [data-testid="stSidebar"] [role="radiogroup"] label[data-testid="stRadioOption"] p {
                color: #D1D5DB !important;
                display: inline-block !important;
                margin: 0 !important;
                padding: 0 !important;
                line-height: 1 !important;
            }

            /* SVG Navigation Icons in Dark Mode (%237AA2DC stroke) */
            [data-testid="stSidebar"] [role="radiogroup"] label[data-testid="stRadioOption"] [data-testid="stMarkdownContainer"]::before {
                content: "";
                display: inline-block !important;
                width: 16px !important;
                height: 16px !important;
                min-width: 16px !important;
                margin-right: 9px !important;
                flex-shrink: 0 !important;
                background-repeat: no-repeat !important;
                background-size: contain !important;
                background-position: center !important;
            }

            [data-testid="stSidebar"] [role="radiogroup"] label[data-testid="stRadioOption"]:nth-of-type(1) [data-testid="stMarkdownContainer"]::before {
                background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%237AA2DC' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='11' cy='11' r='8'%3E%3C/circle%3E%3Cline x1='21' y1='21' x2='16.65' y2='16.65'%3E%3C/line%3E%3C/svg%3E") !important;
            }

            [data-testid="stSidebar"] [role="radiogroup"] label[data-testid="stRadioOption"]:nth-of-type(2) [data-testid="stMarkdownContainer"]::before {
                background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%237AA2DC' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z'%3E%3C/path%3E%3Cpath d='M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z'%3E%3C/path%3E%3C/svg%3E") !important;
            }

            [data-testid="stSidebar"] [role="radiogroup"] label[data-testid="stRadioOption"]:nth-of-type(3) [data-testid="stMarkdownContainer"]::before {
                background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%237AA2DC' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z'%3E%3C/path%3E%3Cpolyline points='14 2 14 8 20 8'%3E%3C/polyline%3E%3Cline x1='16' y1='13' x2='8' y2='13'%3E%3C/line%3E%3Cline x1='16' y1='17' x2='8' y2='17'%3E%3C/line%3E%3Cpolyline points='10 9 9 9 8 9'%3E%3C/polyline%3E%3C/svg%3E") !important;
            }

            [data-testid="stSidebar"] [role="radiogroup"] label[data-testid="stRadioOption"]:nth-of-type(4) [data-testid="stMarkdownContainer"]::before {
                background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%237AA2DC' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='16' y='16' width='6' height='6' rx='1'%3E%3C/rect%3E%3Crect x='2' y='16' width='6' height='6' rx='1'%3E%3C/rect%3E%3Crect x='9' y='2' width='6' height='6' rx='1'%3E%3C/rect%3E%3Cpath d='M5 16v-3a1 1 0 0 1 1-1h12a1 1 0 0 1 1 1v3'%3E%3C/path%3E%3Cline x1='12' y1='12' x2='12' y2='8'%3E%3C/line%3E%3C/svg%3E") !important;
            }

            [data-testid="stSidebar"] [role="radiogroup"] label[data-testid="stRadioOption"]:nth-of-type(5) [data-testid="stMarkdownContainer"]::before {
                background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%237AA2DC' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cline x1='18' y1='20' x2='18' y2='10'%3E%3C/line%3E%3Cline x1='12' y1='20' x2='12' y2='4'%3E%3C/line%3E%3Cline x1='6' y1='20' x2='6' y2='14'%3E%3C/line%3E%3C/svg%3E") !important;
            }

            [data-testid="stSidebar"] hr {
                border-color: #2E3238 !important;
            }

            /* ====================================================================
               3. MAIN CANVAS CONTENT, TYPOGRAPHY & UNIVERSAL INLINE CODE
               ==================================================================== */
            [data-testid="stMarkdownContainer"] p,
            [data-testid="stMarkdownContainer"] span,
            [data-testid="stMarkdownContainer"] li,
            [data-testid="stMarkdownContainer"] div {
                color: #E8E8E6 !important;
            }

            .hero-title, .academic-title, .source-title,
            h1, h2, h3, h4, h5, h6,
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

            /* Universal Inline Code, Badges, and Tag Chips in Dark Mode */
            code,
            kbd,
            [data-testid="stMarkdownContainer"] code,
            .stMarkdown code,
            p code,
            span code,
            div code,
            li code,
            td code,
            th code,
            .source-meta code {
                background-color: #1E2126 !important;
                background: #1E2126 !important;
                color: #E8E8E6 !important;
                border: 1px solid #2E3238 !important;
                border-radius: 4px !important;
                padding: 2px 6px !important;
                font-size: 0.85em !important;
                font-family: 'JetBrains Mono', monospace !important;
            }

            /* Category, Venue & Domain Badges */
            .category-chip,
            .domain-tag {
                background-color: #1E2126 !important;
                color: #E8E8E6 !important;
                border: 1px solid #2E3238 !important;
            }

            /* ====================================================================
               4. CARDS & ELEVATED PANELS
               ==================================================================== */
            .glass-card, .source-card, .metric-box, .answer-box {
                background: #1E2126 !important;
                border: 1px solid #2E3238 !important;
                color: #E8E8E6 !important;
            }
            .source-card {
                scroll-margin-top: 80px;
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

            /* Expanders */
            [data-testid="stExpander"],
            details[data-testid="stExpander"],
            [data-testid="stExpander"] summary,
            details[data-testid="stExpander"] summary,
            [data-testid="stExpander"] > details > summary {
                background-color: #1E2126 !important;
                background: #1E2126 !important;
                color: #F3F4F6 !important;
                border-color: #2E3238 !important;
            }
            [data-testid="stExpander"] summary:hover {
                background-color: #262A30 !important;
            }
            [data-testid="stExpander"] summary p,
            [data-testid="stExpander"] summary span,
            [data-testid="stExpander"] summary svg {
                color: #F3F4F6 !important;
                fill: #F3F4F6 !important;
            }

            /* ====================================================================
               5. INPUT CONTROLS, DROPDOWNS & BUTTONS
               ==================================================================== */
            [data-testid="stWidgetLabel"] p,
            [data-testid="stWidgetLabel"] span,
            [data-testid="stWidgetLabel"] label,
            label[data-testid="stWidgetLabel"] p,
            label[data-baseweb="checkbox"] p,
            label[data-baseweb="checkbox"] span {
                color: #E8E8E6 !important;
            }

            [data-testid="stTextInputRootElement"],
            [data-testid="stTextInputRootElement"] input,
            [data-baseweb="input"],
            [data-baseweb="input"] input,
            input[type="text"], textarea {
                background-color: #1E2126 !important;
                background: #1E2126 !important;
                color: #F3F4F6 !important;
                border: 1px solid #2E3238 !important;
            }

            [data-baseweb="select"],
            [data-baseweb="select"] > div,
            [data-baseweb="select"] div,
            [data-baseweb="select"] span {
                background-color: #1E2126 !important;
                background: #1E2126 !important;
                color: #F3F4F6 !important;
                border-color: #2E3238 !important;
            }

            /* Primary and Secondary Buttons */
            button[kind="primary"] {
                background-color: #3B6199 !important;
                color: #FFFFFF !important;
                border: 1px solid #4A73B0 !important;
            }
            button[kind="primary"] * {
                color: #FFFFFF !important;
            }
            button[kind="primary"]:hover {
                background-color: #4A73B0 !important;
            }

            button[kind="secondary"] {
                background-color: #1E2126 !important;
                background: #1E2126 !important;
                border: 1px solid #3E444E !important;
            }
            button[kind="secondary"] * {
                color: #E8E8E6 !important;
            }
            button[kind="secondary"]:hover {
                background-color: #262A30 !important;
                border-color: #4B5563 !important;
            }

            /* ====================================================================
               6. BADGES, STATUS PILLS & CITATION CHIPS
               ==================================================================== */
            .badge-pill {
                font-family: 'Inter', sans-serif !important;
            }

            .badge-outline, a.badge-outline, span.badge-outline {
                background: #1E2126 !important;
                color: #E2E8F0 !important;
                border: 1px solid #4B5563 !important;
                font-weight: 550 !important;
            }
            .badge-outline *, a.badge-outline *, span.badge-outline * {
                color: #E2E8F0 !important;
            }

            .badge-slate, a.badge-slate, span.badge-slate {
                background: #262A30 !important;
                color: #CBD5E1 !important;
                border: 1px solid #3E444E !important;
            }
            .badge-slate *, a.badge-slate *, span.badge-slate * {
                color: #CBD5E1 !important;
            }

            .badge-strong, a.badge-strong, span.badge-strong {
                background: #143521 !important;
                color: #4ADE80 !important;
                border: 1px solid #1E5631 !important;
            }
            .badge-strong *, a.badge-strong *, span.badge-strong * {
                color: #4ADE80 !important;
            }

            .badge-moderate, a.badge-moderate, span.badge-moderate {
                background: #3A230B !important;
                color: #FBBF24 !important;
                border: 1px solid #854D0E !important;
            }
            .badge-moderate *, a.badge-moderate *, span.badge-moderate * {
                color: #FBBF24 !important;
            }

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

            /* Tabs */
            [data-testid="stTabs"] [data-baseweb="tab-list"] {
                border-bottom: 1px solid #2E3238 !important;
            }
            [data-testid="stTabs"] [data-baseweb="tab"] p,
            [data-testid="stTabs"] [data-baseweb="tab"] span {
                color: #9CA3AF !important;
            }
            [data-testid="stTabs"] [aria-selected="true"] p,
            [data-testid="stTabs"] [aria-selected="true"] span {
                color: #7AA2DC !important;
                border-bottom: 2px solid #7AA2DC !important;
            }

            /* Dividers */
            hr {
                border-color: #2E3238 !important;
            }

            /* ====================================================================
               7. UNIVERSAL DATAFRAME, DATA EDITOR & TABLE DARK THEME
               ==================================================================== */
            [data-testid="stDataFrame"],
            [data-testid="stDataFrameResizable"],
            [data-testid="stDataFrameContainer"],
            [data-testid="stDataFrame"] > div,
            [data-testid="stDataFrameResizable"] > div,
            .dvn-scroller,
            .dvn-underlay,
            .dvn-overlay,
            .glideDataGrid {
                background-color: #1E2126 !important;
                background: #1E2126 !important;
                color: #F3F4F6 !important;
                border-color: #2E3238 !important;
            }

            /* Custom Dataframe wrapper border & rounded corner styling */
            [data-testid="stDataFrame"] {
                border: 1px solid #2E3238 !important;
                border-radius: 6px !important;
                overflow: hidden !important;
            }

            /* Native Streamlit Table & pandas HTML dataframe rendering */
            [data-testid="stTable"],
            [data-testid="stTable"] table,
            table.dataframe,
            .stTable table {
                background-color: #1E2126 !important;
                color: #E8E8E6 !important;
                border: 1px solid #2E3238 !important;
                border-radius: 6px !important;
                border-collapse: collapse !important;
                width: 100% !important;
            }

            [data-testid="stTable"] th,
            table.dataframe th,
            .stTable th {
                background-color: #262A30 !important;
                color: #F3F4F6 !important;
                border: 1px solid #2E3238 !important;
                font-weight: 600 !important;
                padding: 8px 12px !important;
                font-size: 0.82rem !important;
            }

            [data-testid="stTable"] td,
            table.dataframe td,
            .stTable td {
                background-color: #1E2126 !important;
                color: #E8E8E6 !important;
                border: 1px solid #2E3238 !important;
                padding: 8px 12px !important;
                font-size: 0.82rem !important;
            }

            [data-testid="stTable"] tr:hover,
            table.dataframe tr:hover,
            .stTable tr:hover {
                background-color: #262A30 !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
