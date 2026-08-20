"""
Interactive One-Click Copy-to-Clipboard Component.
Renders a theme-aware icon button with animated checkmark confirmation.
Dual-Theme (Light & Dark Mode) aware, Zero Emojis.
"""

import json
import streamlit as st
import streamlit.components.v1 as components
from frontend.components.icons import svg_icon
from frontend.styles.theme import is_dark_mode, get_theme_colors


def render_copy_button(
    text_to_copy: str,
    label: str = "Copy Answer",
    tooltip: str = "Copy text to clipboard",
    height: int = 34
):
    """
    Render a compact, theme-styled copy button that writes to system clipboard
    and switches to a checkmark confirmation for 1.5 seconds on click.
    """
    colors = get_theme_colors()
    dark = is_dark_mode()

    # SVG markup for default copy and check confirmation
    copy_icon_color = "#9CA3AF" if dark else "#6B7280"
    check_icon_color = "#4ADE80" if dark else "#16A34A"
    
    copy_svg = svg_icon("copy", size=14, color=copy_icon_color)
    check_svg = svg_icon("check", size=14, color=check_icon_color)

    # Theme CSS properties
    btn_bg = "#1E2126" if dark else "#FFFFFF"
    btn_hover_bg = "#262A30" if dark else "#F3F4F6"
    btn_border = "#3E444E" if dark else "#D1D5DB"
    btn_text = "#E8E8E6" if dark else "#374151"
    copied_bg = "#143521" if dark else "#EBF5EE"
    copied_border = "#1E5631" if dark else "#C4E3CB"
    copied_text = "#4ADE80" if dark else "#1E5631"

    escaped_text = json.dumps(text_to_copy)
    escaped_label = json.dumps(label)

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{
                margin: 0;
                padding: 0;
                background: transparent;
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                overflow: hidden;
            }}
            .copy-btn {{
                display: inline-flex;
                align-items: center;
                gap: 6px;
                background-color: {btn_bg};
                color: {btn_text};
                border: 1px solid {btn_border};
                border-radius: 5px;
                padding: 5px 10px;
                font-size: 0.78rem;
                font-weight: 500;
                letter-spacing: 0.01em;
                cursor: pointer;
                transition: all 0.15s ease;
                outline: none;
                user-select: none;
                line-height: 1;
                height: 28px;
                box-sizing: border-box;
            }}
            .copy-btn:hover {{
                background-color: {btn_hover_bg};
                border-color: #5B7FB5;
            }}
            .copy-btn.copied {{
                background-color: {copied_bg} !important;
                border-color: {copied_border} !important;
                color: {copied_text} !important;
            }}
            .icon-wrap {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
            }}
        </style>
    </head>
    <body>
        <button class="copy-btn" id="copyBtn" onclick="executeCopy()" title="{tooltip}">
            <span class="icon-wrap" id="copyIcon">{copy_svg}</span>
            <span id="copyLabel">{label}</span>
        </button>

        <script>
            function executeCopy() {{
                const text = {escaped_text};
                const originalLabel = {escaped_label};
                const btn = document.getElementById("copyBtn");
                const icon = document.getElementById("copyIcon");
                const lbl = document.getElementById("copyLabel");

                navigator.clipboard.writeText(text).then(() => {{
                    btn.classList.add("copied");
                    icon.innerHTML = `{check_svg}`;
                    lbl.innerText = "Copied!";

                    setTimeout(() => {{
                        btn.classList.remove("copied");
                        icon.innerHTML = `{copy_svg}`;
                        lbl.innerText = originalLabel;
                    }}, 1500);
                }}).catch(err => {{
                    console.error("Failed to copy text:", err);
                }});
            }}
        </script>
    </body>
    </html>
    """

    components.html(html_code, height=height, scrolling=False)
