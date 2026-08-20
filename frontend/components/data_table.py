"""
Academic Data Table Component.
Renders responsive, beautifully styled tabular data with dual-theme support.
Eliminates Streamlit Glide Data Grid canvas tab-switching sizing bugs and theming gaps.
"""

import pandas as pd
import streamlit as st
from frontend.styles.theme import get_theme_colors, is_dark_mode


def render_academic_table(
    df: pd.DataFrame,
    max_height_px: int = 360,
    wrap_cells: bool = False
):
    """
    Render a responsive academic table matching the scientific instrument design system.
    Supports both light and dark mode automatically.
    """
    colors = get_theme_colors()
    dark = is_dark_mode()

    th_bg = "#262A30" if dark else "#F3F4F6"
    th_color = "#F3F4F6" if dark else "#111827"
    tr_even = "#1E2126" if dark else "#FFFFFF"
    tr_odd = "#1A1D21" if dark else "#FAFAF9"
    tr_hover = "#2A2E36" if dark else "#F1F5F9"
    border_color = "#2E3238" if dark else "#E5E5E3"
    text_color = "#E8E8E6" if dark else "#1F2937"
    text_muted = "#9CA3AF" if dark else "#6B7280"

    if df.empty:
        st.markdown(
            f'<div style="border: 1px solid {border_color}; border-radius: 6px; padding: 18px; text-align: center; color: {text_muted}; background: {tr_even}; font-size: 0.84rem; margin: 8px 0 16px 0;">No active records to display.</div>',
            unsafe_allow_html=True
        )
        return

    headers_html = "".join(
        f"<th style='padding: 10px 14px; text-align: left; font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; color: {th_color}; background-color: {th_bg}; border-bottom: 2px solid {border_color}; position: sticky; top: 0; z-index: 1;'>{col}</th>"
        for col in df.columns
    )

    rows_html = []
    for row_idx, row in df.iterrows():
        bg = tr_even if row_idx % 2 == 0 else tr_odd
        cells_html = []
        for col in df.columns:
            val = str(row[col])
            
            # Format specific scientific column patterns
            if "Rank" in col or "Score" in col or "ID" in col or "Context #" in col or "#" in val or col == "id":
                cell_content = f"<span style='font-family: \"JetBrains Mono\", monospace; font-size: 0.82rem; background: {th_bg}; padding: 2px 6px; border-radius: 4px; border: 1px solid {border_color};'>{val}</span>"
            elif val == "Injected in Context":
                cell_content = f"<span style='font-size: 0.72rem; background: {'#143521' if dark else '#EBF5EE'}; color: {'#4ADE80' if dark else '#1E5631'}; border: 1px solid {'#1E5631' if dark else '#C4E3CB'}; padding: 2px 8px; border-radius: 9999px; font-weight: 600;'>{val}</span>"
            elif col == "Status" or col == "status":
                cell_content = f"<span style='font-size: 0.72rem; background: {th_bg}; color: {text_color}; border: 1px solid {border_color}; padding: 2px 8px; border-radius: 9999px;'>{val}</span>"
            else:
                cell_content = f"<span style='font-size: 0.84rem;'>{val}</span>"

            white_space = "normal" if wrap_cells else "nowrap"
            cells_html.append(
                f"<td style='padding: 9px 14px; border-bottom: 1px solid {border_color}; color: {text_color}; white-space: {white_space}; vertical-align: middle;'>{cell_content}</td>"
            )
        
        row_str = "".join(cells_html)
        rows_html.append(
            f"<tr style='background-color: {bg}; transition: background-color 0.15s ease;' onmouseover=\"this.style.backgroundColor='{tr_hover}'\" onmouseout=\"this.style.backgroundColor='{bg}'\">{row_str}</tr>"
        )

    all_rows = "".join(rows_html)
    scroll_style = f"max-height: {max_height_px}px; overflow-y: auto;" if max_height_px else ""

    # Generate flat HTML without any leading indentation so markdown won't interpret as code block
    table_markup = (
        f'<div style="border: 1px solid {border_color}; border-radius: 6px; overflow-x: auto; {scroll_style} margin: 8px 0 16px 0; background: {tr_even};">'
        f'<table style="width: 100%; border-collapse: collapse; font-family: \'Inter\', -apple-system, sans-serif; line-height: 1.4;">'
        f'<thead><tr>{headers_html}</tr></thead>'
        f'<tbody>{all_rows}</tbody>'
        f'</table>'
        f'</div>'
    )

    st.markdown(table_markup, unsafe_allow_html=True)
