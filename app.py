"""
Interface Streamlit do TráfegoAI - Agente RAG de Tráfego Pago.

Uso:
    streamlit run app.py
"""

import os
import sys
import base64
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

# ---------------------------------------------------------------------------
# Ícone SVG como data URI (evita sanitização do Streamlit)
# ---------------------------------------------------------------------------
_SVG_ICON = """<svg width="48" height="48" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="tai-bg" x1="0" y1="48" x2="48" y2="0" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="#1e3a8a" stop-opacity="0.55"/>
      <stop offset="100%" stop-color="#1d4ed8" stop-opacity="0.22"/>
    </linearGradient>
    <linearGradient id="tai-bd" x1="0" y1="48" x2="48" y2="0" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="#3b82f6"/>
      <stop offset="100%" stop-color="#93c5fd"/>
    </linearGradient>
    <linearGradient id="tai-ar" x1="10" y1="36" x2="38" y2="12" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="#3b82f6"/>
      <stop offset="100%" stop-color="#bae6fd"/>
    </linearGradient>
  </defs>
  <rect x="2" y="2" width="44" height="44" rx="13" fill="url(#tai-bg)" stroke="url(#tai-bd)" stroke-width="1.5"/>
  <polyline points="10,36 19,26 27,30 38,13" stroke="url(#tai-ar)" stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
  <circle cx="38" cy="13" r="2.5" fill="#bae6fd"/>
</svg>"""
ICON_URI = "data:image/svg+xml;base64," + base64.b64encode(_SVG_ICON.encode()).decode()

sys.path.insert(0, str(Path(__file__).parent))

load_dotenv()

from src.rag.indexer import load_index
from src.agent.graph import build_agent_graph


def _secret(key: str, default: str | None = None) -> str | None:
    """Lê de st.secrets (Streamlit Cloud) com fallback para os.getenv (local)."""
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return os.getenv(key, default)


# ---------------------------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TráfegoAI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Estado do tema
# ---------------------------------------------------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

_dark = st.session_state.theme == "dark"

if _dark:
    C = dict(
        app_bg       = "#060b18",
        sidebar_bg   = "#08101e",
        card_bg      = "#0d1527",
        border       = "rgba(59,130,246,0.15)",
        text         = "#e2e8f0",
        text_muted   = "#8ba4c0",
        title_grad   = "linear-gradient(90deg,#60a5fa 0%,#bae6fd 100%)",
        accent       = "#3b82f6",
        accent_rgb   = "59,130,246",
        tag_text     = "#93c5fd",
        divider      = "rgba(59,130,246,0.08)",
        input_bg     = "#0d1527",
        status_bg    = "rgba(34,197,94,0.08)",
        status_brd   = "rgba(34,197,94,0.2)",
        status_txt   = "#4ade80",
        status_dot   = "#22c55e",
        btn_bg       = "rgba(59,130,246,0.1)",
        btn_brd      = "rgba(59,130,246,0.3)",
        btn_txt      = "#60a5fa",
        shadow       = "0 4px 24px rgba(0,0,0,0.5)",
        theme_icon   = "\u2600\ufe0f",
        theme_label  = "Modo Claro",
        next_theme   = "light",
    )
else:
    C = dict(
        app_bg       = "#f0f4f8",
        sidebar_bg   = "#e2e8f0",
        card_bg      = "#ffffff",
        border       = "rgba(30,64,175,0.12)",
        text         = "#0f172a",
        text_muted   = "#475569",
        title_grad   = "linear-gradient(90deg,#1e40af 0%,#2563eb 100%)",
        accent       = "#2563eb",
        accent_rgb   = "37,99,235",
        tag_text     = "#1e40af",
        divider      = "rgba(30,64,175,0.08)",
        input_bg     = "#ffffff",
        status_bg    = "rgba(22,163,74,0.08)",
        status_brd   = "rgba(22,163,74,0.25)",
        status_txt   = "#166534",
        status_dot   = "#16a34a",
        btn_bg       = "rgba(37,99,235,0.06)",
        btn_brd      = "rgba(37,99,235,0.2)",
        btn_txt      = "#1d4ed8",
        shadow       = "0 2px 12px rgba(0,0,0,0.07)",
        theme_icon   = "\U0001f319",
        theme_label  = "Modo Escuro",
        next_theme   = "dark",
    )

# CSS extra condicional: dark mode precisa sobrescrever toolbar do Streamlit
# que renderiza com base="light" (icons/texto dark por padrão)
_extra_css = f"""
    /* -- Hide toolbar menu button -- */
    #MainMenu,
    [data-testid="stToolbar"] [data-testid="stToolbarActions"],
    [data-testid="stDecoration"] {{
        display: none !important;
        visibility: hidden !important;
    }}
    /* -- Hide Streamlit footer branding -- */
    footer, [data-testid="stStatusWidget"],
    .viewerBadge_container__r5tak,
    .stApp > footer,
    [data-testid="manage-app-button"] {{
        display: none !important;
        visibility: hidden !important;
    }}
    /* -- Dark: toolbar icons -- */
    [data-testid="stHeader"] button,
    [data-testid="stHeader"] a {{
        filter: brightness(0) invert(0.7) !important;
    }}
    [data-testid="stHeader"] button:hover {{
        filter: brightness(0) invert(1) !important;
        background: rgba(255,255,255,0.06) !important;
    }}
    /* -- Dark: all text -- */
    p, li, label, .stMarkdown, h1, h2, h3, h4, span {{
        color: {C['text']} !important;
    }}
    /* -- Dark: stBottom — wipe wrappers, then re-style with higher specificity -- */
    [data-testid="stBottom"] {{
        background-color: {C['app_bg']} !important;
        border-top: 1px solid rgba({C['accent_rgb']},0.12) !important;
    }}
    /* Only paint direct wrappers of stBottom, not every nested div */
    [data-testid="stBottom"] > div {{
        background-color: {C['app_bg']} !important;
        box-shadow: none !important;
    }}
    /* Inner divs of stChatInput: transparent (prevents color patches) */
    [data-testid="stBottom"] [data-testid="stChatInput"] div {{
        background: transparent !important;
        box-shadow: none !important;
    }}
    /* Chat input visual box: only the outer container gets the background */
    [data-testid="stBottom"] [data-testid="stChatInput"] > div {{
        background: {C['input_bg']} !important;
        border: 1.5px solid rgba({C['accent_rgb']},0.55) !important;
        border-radius: 16px !important;
        box-shadow: none !important;
        overflow: hidden !important;
        background-clip: padding-box !important;
    }}
    [data-testid="stBottom"] [data-testid="stChatInput"] > div:focus-within {{
        border-color: {C['accent']} !important;
        box-shadow: 0 0 0 3px rgba({C['accent_rgb']},0.18),
                    0 0 24px rgba({C['accent_rgb']},0.22) !important;
    }}
    /* textarea padding */
    [data-testid="stChatInputTextArea"] {{
        padding: 0.65rem 1rem !important;
    }}
    /* specificity (0,1,1) for textarea inside bottom */
    [data-testid="stBottom"] textarea {{
        background-color: transparent !important;
        color: {C['text']} !important;
        caret-color: {C['accent']} !important;
    }}
    /* submit button: transparent, arrow only */
    [data-testid="stBottom"] [data-testid="stChatInputSubmitButton"] button {{
        background: transparent !important;
        border: none !important;
        border-radius: 0 !important;
        box-shadow: none !important;
    }}
    [data-testid="stBottom"] [data-testid="stChatInputSubmitButton"] button * {{
        color: {C['accent']} !important;
        fill: {C['accent']} !important;
        stroke: {C['accent']} !important;
    }}
    /* -- Dark: all buttons (sidebar + main) -- */
    [data-testid="stSidebar"] {{
        background-color: {C['sidebar_bg']} !important;
    }}
    [data-testid="stSidebar"] button,
    .stButton button {{
        background-color: {C['btn_bg']} !important;
        border: 1px solid {C['btn_brd']} !important;
        color: {C['btn_txt']} !important;
    }}
    [data-testid="stSidebar"] button:hover,
    .stButton button:hover {{
        background-color: rgba({C['accent_rgb']},0.18) !important;
        border-color: {C['accent']} !important;
        box-shadow: 0 0 10px rgba({C['accent_rgb']},0.2) !important;
    }}
""" if _dark else """"""

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif !important; }}
    .stApp, [data-testid="stAppViewContainer"],
    [data-testid="stMain"], [data-testid="stMainBlockContainer"] {{
        background: {C['app_bg']} !important;
    }}
    .block-container {{ padding: 1.5rem 2rem 1rem 2rem !important; max-width: 100% !important; }}
    /* Quando sidebar colapsada, main ocupa toda a largura */
    [data-testid="stSidebar"][aria-expanded="false"] ~ [data-testid="stMain"],
    [data-testid="stSidebar"][aria-expanded="false"] ~ [data-testid="stAppViewContainer"] {{
        margin-left: 0 !important;
    }}
    [data-testid="stMain"] {{
        transition: margin-left 0.3s ease !important;
    }}
    [data-testid="stHeader"] {{
        background: {C['app_bg']} !important;
        border-bottom: 1px solid {C['border']} !important;
    }}
    [data-testid="stBottom"], [data-testid="stBottom"] > div {{
        background: {C['app_bg']} !important;
        border-top: 1px solid {C['border']} !important;
    }}
    [data-testid="stSidebar"], [data-testid="stSidebar"] > div {{
        background: {C['sidebar_bg']} !important;
        border-right: 1px solid {C['border']} !important;
    }}
    .hero {{
        display: flex; align-items: center; justify-content: space-between;
        padding: 0.8rem 0 0.9rem 0;
        border-bottom: 1px solid {C['border']};
        margin-bottom: 1.6rem;
    }}
    .hero-left {{ display: flex; flex-direction: row; align-items: center; gap: 14px; }}
    .hero-text-col {{ display: flex; flex-direction: column; gap: 2px; }}
    .hero-icon-wrap {{
        flex-shrink: 0; width: 44px; height: 44px; line-height: 0; overflow: visible !important;
        background: #0f172a; border-radius: 12px;
    }}
    .hero-icon-wrap img {{
        width: 44px !important; height: 44px !important;
        min-width: 44px !important; min-height: 44px !important;
        display: block !important; border-radius: 10px !important;
        object-fit: contain !important; overflow: visible !important;
    }}
    .hero-title {{
        font-size: 1.6rem; font-weight: 800; letter-spacing: -0.6px; line-height: 1.1;
        background: {C['title_grad']};
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    }}
    .hero-subtitle {{ font-size: 0.78rem; color: {C['text_muted']}; }}
    .status-badge {{
        display: inline-flex; align-items: center; gap: 7px;
        background: {C['status_bg']}; border: 1px solid {C['status_brd']};
        border-radius: 20px; padding: 5px 14px;
        font-size: 0.72rem; color: {C['status_txt']}; font-weight: 600;
    }}
    .status-dot {{
        width: 7px; height: 7px; background: {C['status_dot']};
        border-radius: 50%; box-shadow: 0 0 8px {C['status_dot']}, 0 0 16px {C['status_dot']};
        animation: pulse 2s infinite;
    }}
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.4; }}
    }}
    [data-testid="stChatMessage"] {{
        background: {C['card_bg']} !important;
        border: 1px solid {C['border']} !important;
        border-radius: 12px !important;
        box-shadow: {C['shadow']} !important;
    }}
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] span {{
        color: {C['text']} !important;
    }}
    [data-testid="stChatInput"], textarea,
    [data-testid="stChatInputTextArea"], .stChatInput textarea {{
        background: {C['input_bg']} !important;
        color: {C['text']} !important;
        caret-color: {C['accent']} !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }}
    textarea::placeholder {{ color: {C['text_muted']} !important; }}
    /* textarea: transparent — container is the visual field */
    [data-testid="stChatInputTextArea"] {{
        padding: 0.65rem 3.5rem 0.65rem 1rem !important;
        background: transparent !important;
        border: none !important;
        border-radius: 0 !important;
        box-sizing: border-box !important;
        width: 100% !important;
        color: {C['text']} !important;
        caret-color: {C['accent']} !important;
        outline: none !important;
        box-shadow: none !important;
    }}
    /* Chat input outer container: visual field with border */
    [data-testid="stChatInput"] > div {{
        background: {C['input_bg']} !important;
        border: 1.5px solid rgba({C['accent_rgb']},0.5) !important;
        border-radius: 16px !important;
        box-shadow: none !important;
    }}
    /* button wrapper e container filhos: transparentes */
    [data-testid="stChatInput"] > div > div:has(button) {{
        background: transparent !important;
        border-radius: 0 !important;
    }}
    button[data-testid="stChatInputSubmitButton"] {{
        background: transparent !important;
        border: none !important;
        border-radius: 0 !important;
        box-shadow: none !important;
        color: {C['accent']} !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: color 0.15s ease, filter 0.15s ease !important;
    }}
    button[data-testid="stChatInputSubmitButton"]:hover {{
        color: #93c5fd !important;
        filter: drop-shadow(0 0 6px rgba({C['accent_rgb']},0.7)) !important;
    }}
    button[data-testid="stChatInputSubmitButton"] svg {{
        width: 28px !important;
        height: 28px !important;
    }}
    .sidebar-bottom-info {{
        position: fixed !important;
        left: 0 !important;
        bottom: 1.5rem !important;
        width: 220px !important;
        padding: 0 1rem !important;
        box-sizing: border-box !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }}
    .sidebar-logo-row {{ display: flex; align-items: center; gap: 9px; margin-bottom: 0.65rem; }}
    .sidebar-logo-text {{
        font-size: 1rem; font-weight: 700;
        background: {C['title_grad']};
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    }}
    .sidebar-desc {{ font-size: 0.76rem; color: {C['text_muted']}; line-height: 1.55; margin-bottom: 0.75rem; overflow-wrap: normal !important; word-break: normal !important; }}
    .sidebar-tags {{ display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 0.75rem; }}
    .sidebar-tag {{
        background: rgba({C['accent_rgb']},0.08);
        border: 1px solid rgba({C['accent_rgb']},0.24);
        border-radius: 7px; padding: 3px 8px;
        font-size: 0.66rem; color: {C['tag_text']}; font-weight: 600;
    }}
    .sidebar-dev {{
        font-size: 0.66rem; color: {C['text_muted']}; opacity: {'0.55' if _dark else '0.85'};
        padding-top: 0.65rem; border-top: 1px solid {C['divider']};
        overflow-wrap: normal !important; word-break: normal !important;
        text-align: center !important;
    }}


    [data-testid="stSidebar"][aria-expanded="true"] {{
        min-width: 220px !important;
        max-width: 220px !important;
    }}
    [data-testid="stSidebar"] > div:first-child {{
        resize: none !important;
    }}
    [data-testid="stSidebarUserContent"] {{
        resize: none !important;
    }}
    [data-testid="stSidebar"] > div {{
        padding-left: 0.8rem !important;
        padding-right: 0.8rem !important;
        padding-top: 0.5rem !important;
    }}
    [data-testid="stSidebarContent"] {{
        padding-left: 0.8rem !important;
        padding-right: 0.8rem !important;
    }}
    .sidebar-actions-title {{
        color: {C['text_muted']} !important; font-size: 0.76rem !important;
        font-weight: 800 !important; text-transform: uppercase !important;
        letter-spacing: 0.09em !important;
        margin: 0 0 0.4rem 0.1rem !important;
        padding-left: 0 !important;
    }}
    .sidebar-actions-title-spaced {{
        margin-top: 0.7rem !important;
    }}
    /* Toggle styling */
    [data-testid="stSidebar"] .stToggle {{
        margin-bottom: 0.5rem !important;
    }}
    [data-testid="stSidebar"] .stToggle label span {{
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }}
    /* Clear chat button: small and subtle */
    [data-testid="stSidebar"] .stButton {{
        margin-bottom: 0 !important;
        margin-top: 0 !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
    }}
    [data-testid="stSidebar"] .stButton button {{
        height: 32px !important;
        width: 100% !important;
        background: transparent !important;
        border: none !important;
        color: {C['text_muted']} !important;
        border-radius: 10px !important; font-weight: 600 !important;
        font-size: 0.85rem !important; justify-content: flex-start !important;
        padding: 0 0.5rem !important; box-shadow: none !important;
        transition: all 0.18s ease !important; white-space: nowrap !important;
    }}
    [data-testid="stSidebar"] .stButton button:hover {{
        background: rgba({C['accent_rgb']},0.08) !important;
        color: {C['text']} !important;
    }}
    [data-testid="stSidebar"] .stButton button:active {{
        box-shadow: none !important;
    }}
    [data-testid="stMain"] .stButton button {{
        background: {C['btn_bg']} !important;
        border: 1px solid {C['btn_brd']} !important;
        color: {C['btn_txt']} !important;
        border-radius: 8px !important; font-weight: 500 !important;
        white-space: nowrap !important;
        transition: all 0.15s ease !important;
    }}
    [data-testid="stMain"] .stButton button:hover {{
        background: rgba({C['accent_rgb']},0.15) !important;
        border-color: {C['accent']} !important;
    }}
    hr {{ border-color: {C['divider']} !important; }}
    ::-webkit-scrollbar {{ width: 5px; }}
    ::-webkit-scrollbar-thumb {{ background: rgba({C['accent_rgb']},0.25); border-radius: 3px; }}
    /* Expander (referências) */
    [data-testid="stExpander"] {{
        background: rgba({C['accent_rgb']},0.04) !important;
        border: 1px solid rgba({C['accent_rgb']},0.15) !important;
        border-radius: 10px !important;
        box-shadow: none !important;
    }}
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary span,
    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] p {{
        color: {C['text_muted']} !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        background: transparent !important;
    }}
    [data-testid="stExpander"] summary:hover,
    [data-testid="stExpander"] summary:hover span {{
        color: {C['text']} !important;
    }}
    [data-testid="stExpander"] summary svg {{
        color: {C['text_muted']} !important;
        fill: {C['text_muted']} !important;
    }}
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] {{
        background: transparent !important;
    }}
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] p,
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] li {{
        color: {C['text_muted']} !important;
        font-size: 0.8rem !important;
    }}
    .custom-loading {{
        display: inline-flex;
        align-items: center;
        gap: 0.6rem;
        background: rgba({C['accent_rgb']}, 0.07);
        border: 1px solid rgba({C['accent_rgb']}, 0.22);
        color: {C['text_muted']};
        border-radius: 10px;
        padding: 0.55rem 0.85rem;
        font-size: 0.86rem;
        font-weight: 500;
    }}
    .custom-loading-dot {{
        width: 9px; height: 9px; border-radius: 50%;
        background: {C['accent']};
        box-shadow: 0 0 10px rgba({C['accent_rgb']}, 0.8);
        animation: pulse 1.2s infinite ease-in-out;
    }}
{_extra_css}
    /* Fix definitivo: cantos arredondados uniformes no chat input */
    [data-testid="stBottom"] [data-testid="stChatInput"] {{
        border-radius: 16px !important;
        overflow: hidden !important;
        background: transparent !important;
    }}
    [data-testid="stBottom"] [data-testid="stChatInput"] > div {{
        background: {C['input_bg']} !important;
        border: 1.5px solid rgba({C['accent_rgb']},0.55) !important;
        border-radius: 16px !important;
        overflow: hidden !important;
        box-shadow: none !important;
        background-clip: padding-box !important;
    }}
    [data-testid="stBottom"] [data-testid="stChatInput"] > div div {{
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }}
    [data-testid="stBottom"] [data-testid="stChatInputTextArea"],
    [data-testid="stBottom"] textarea {{
        background: transparent !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }}
    [data-testid="stBottom"] button[data-testid="stChatInputSubmitButton"],
    [data-testid="stBottom"] button[data-testid="stChatInputSubmitButton"]:hover,
    [data-testid="stBottom"] button[data-testid="stChatInputSubmitButton"]:focus,
    [data-testid="stBottom"] button[data-testid="stChatInputSubmitButton"]:active,
    [data-testid="stBottom"] button[data-testid="stChatInputSubmitButton"]:focus-visible {{
        background: transparent !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# JS injector: appends <style> to parent document.head AFTER React renders
# This is the only reliable way to override Emotion/Streamlit generated CSS
# ---------------------------------------------------------------------------
_inject_css = f"""\
[data-testid="stBottom"] {{
  background-color: {C['app_bg']} !important;
}}
/* textarea: transparent — visual field is the container */
[data-testid="stBottom"] [data-testid="stChatInputTextArea"] {{
  background: transparent !important;
  border: none !important;
  border-radius: 0 !important;
  padding: 0.65rem 3.5rem 0.65rem 1rem !important;
  box-sizing: border-box !important;
  width: 100% !important;
  color: {C['text']} !important;
  caret-color: {C['accent']} !important;
  outline: none !important;
  box-shadow: none !important;
}}
[data-testid="stBottom"] textarea {{
  color: {C['text']} !important;
  caret-color: {C['accent']} !important;
}}
/* submit button: transparent bg, blue arrow only — all states */
button[data-testid="stChatInputSubmitButton"],
button[data-testid="stChatInputSubmitButton"]:hover,
button[data-testid="stChatInputSubmitButton"]:focus,
button[data-testid="stChatInputSubmitButton"]:active,
button[data-testid="stChatInputSubmitButton"]:focus-visible {{
  background: transparent !important;
  border: 0 !important;
  outline: 0 !important;
  box-shadow: none !important;
  border-radius: 0 !important;
  color: {C['accent']} !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
}}
button[data-testid="stChatInputSubmitButton"]:hover,
button[data-testid="stChatInputSubmitButton"]:focus-visible {{
  color: #93c5fd !important;
  filter: drop-shadow(0 0 6px rgba({C['accent_rgb']},0.7)) !important;
}}
button[data-testid="stChatInputSubmitButton"] svg {{
  width: 28px !important;
  height: 28px !important;
}}
/* Main menu popover — cover every layer including the inner white div */
[data-testid="stMainMenuPopover"],
[data-testid="stMainMenuPopover"] > div,
[data-testid="stMainMenuPopover"] > div > div {{
  background-color: {C['card_bg']} !important;
  border-radius: 12px !important;
  box-shadow: 0 8px 40px rgba(0,0,0,0.55) !important;
}}
[data-testid="stMainMenuPopover"] {{
  border: 1px solid rgba({C['accent_rgb']},0.2) !important;
  padding: 4px 0 !important;
  overflow: hidden !important;
}}
/* list background */
[data-testid="stMainMenuList"] {{
  background-color: {C['card_bg']} !important;
  border: none !important;
  box-shadow: none !important;
  padding: 4px 0 !important;
}}
/* each item row */
[data-testid="stMainMenuPopover"] [role="option"] {{
  background-color: transparent !important;
  border-radius: 6px !important;
  margin: 1px 6px !important;
}}
[data-testid="stMainMenuPopover"] [role="option"]:hover {{
  background-color: rgba({C['accent_rgb']},0.12) !important;
}}
/* text inside items */
[data-testid="stMainMenuPopover"] span,
[data-testid="stMainMenuPopover"] li {{
  color: {C['text']} !important;
  font-size: 0.85rem !important;
  background-color: transparent !important;
}}
/* text inside items — normal weight */
[data-testid="stMainMenuPopover"] [role="option"] li {{
  font-weight: 400 !important;
  font-size: 0.85rem !important;
}}
/* divider */
[data-testid="stMainMenuDivider"] {{
  border-top: 1px solid rgba({C['accent_rgb']},0.12) !important;
  margin: 4px 12px !important;
  height: 0 !important;
  background: none !important;
}}
/* keyboard shortcut label */
[data-testid="stMainMenuPopover"] span[class*="rj14pv"] {{
  color: {C['text_muted']} !important;
  opacity: 0.6 !important;
  font-size: 0.72rem !important;
}}
/* Hero popover (⋮ menu) */
[data-testid="stPopover"],
[data-testid="stPopover"] > div,
[data-testid="stPopover"] > div > div {{
  background-color: {C['card_bg']} !important;
  border-radius: 12px !important;
  box-shadow: 0 8px 40px rgba(0,0,0,0.55) !important;
}}
[data-testid="stPopover"] {{
  border: 1px solid rgba({C['accent_rgb']},0.2) !important;
  overflow: hidden !important;
}}
[data-testid="stPopover"] button {{
  background: transparent !important;
  border: 1px solid rgba({C['accent_rgb']},0.15) !important;
  color: {C['text']} !important;
  border-radius: 8px !important;
  width: 100% !important;
}}
[data-testid="stPopover"] button:hover {{
  background: rgba({C['accent_rgb']},0.12) !important;
  border-color: rgba({C['accent_rgb']},0.4) !important;
}}
[data-testid="stPopover"] button p {{
  color: {C['text']} !important;
  font-size: 0.85rem !important;
}}
/* Modals / Dialogs (Settings, Clear Caches) */
[role="dialog"],
[role="dialog"] > div {{
  background-color: {C['card_bg']} !important;
  border: 1px solid rgba({C['accent_rgb']},0.2) !important;
  border-radius: 14px !important;
  box-shadow: 0 12px 48px rgba(0,0,0,0.6) !important;
  color: {C['text']} !important;
}}
[role="dialog"] h1,
[role="dialog"] h2,
[role="dialog"] h3,
[role="dialog"] p,
[role="dialog"] span,
[role="dialog"] label,
[role="dialog"] li {{
  color: {C['text']} !important;
}}
[role="dialog"] [data-baseweb="modal-header"],
[role="dialog"] [data-baseweb="modal-body"],
[role="dialog"] [data-baseweb="modal-footer"] {{
  background-color: {C['card_bg']} !important;
}}
[role="dialog"] button {{
  background: rgba({C['accent_rgb']},0.08) !important;
  border: 1px solid rgba({C['accent_rgb']},0.25) !important;
  color: {C['text']} !important;
  border-radius: 8px !important;
}}
[role="dialog"] button:hover {{
  background: rgba({C['accent_rgb']},0.16) !important;
  border-color: rgba({C['accent_rgb']},0.5) !important;
}}
/* Modal close button (X) */
[role="dialog"] button[aria-label="Close"],
[role="dialog"] [data-testid="stModalCloseButton"] {{
  background: transparent !important;
  border: none !important;
  color: {C['text_muted']} !important;
}}
[role="dialog"] button[aria-label="Close"]:hover {{
  color: {C['text']} !important;
  background: rgba({C['accent_rgb']},0.1) !important;
}}
/* Modal backdrop */
[data-baseweb="modal-backdrop"],
div[class*="backdrop"] {{
  background-color: rgba(0,0,0,0.6) !important;
}}
/* Select/dropdown inside modals */
[role="dialog"] [data-baseweb="select"],
[role="dialog"] [data-baseweb="select"] > div {{
  background-color: {C['app_bg']} !important;
  border: 1px solid rgba({C['accent_rgb']},0.25) !important;
  color: {C['text']} !important;
  border-radius: 8px !important;
}}
/* Checkbox in modals */
[role="dialog"] [data-baseweb="checkbox"] span {{
  background-color: transparent !important;
  border-color: rgba({C['accent_rgb']},0.4) !important;
}}
/* Section headers in Settings */
[role="dialog"] h2 {{
  color: {C['accent']} !important;
  font-size: 0.9rem !important;
}}
""" if _dark else ""

components.html(
    f"""<script>
(function() {{
  var css = `{_inject_css}`;
  function inject() {{
    var p = window.parent;
    if (!p || !p.document) return;
    if (!p.document.querySelector('[data-testid="stBottom"]')) {{
      setTimeout(inject, 300);
      return;
    }}
    var el = p.document.getElementById('tai-fix');
    if (!el) {{
      el = p.document.createElement('style');
      el.id = 'tai-fix';
      p.document.head.appendChild(el);
    }}
    el.textContent = css;
    var cont = p.document.querySelector('[data-testid="stChatInput"] > div');
    var btn = p.document.querySelector('button[data-testid="stChatInputSubmitButton"]');
    var bw = btn ? btn.parentElement : null;
    var ta = p.document.querySelector('[data-testid="stChatInputTextArea"]');

    var BLUE = 'rgba({C['accent_rgb']},0.55)';
    var INPUT_BG = '{C['input_bg']}';
    var ACCENT = '{C['accent']}';

    // All inner divs of stChatInput: transparent
    var chatInputEl = p.document.querySelector('[data-testid="stChatInput"]');
    if (chatInputEl) {{
      chatInputEl.querySelectorAll('div').forEach(function(d) {{
        d.style.setProperty('background', 'transparent', 'important');
        d.style.setProperty('box-shadow', 'none', 'important');
      }});
    }}
    // Container: only the outer box gets the background
    if (cont) {{
      cont.style.setProperty('background', INPUT_BG, 'important');
      cont.style.setProperty('border', '1.5px solid ' + BLUE, 'important');
      cont.style.setProperty('border-radius', '16px', 'important');
      cont.style.setProperty('box-shadow', 'none', 'important');
      cont.style.setProperty('overflow', 'hidden', 'important');
      cont.style.setProperty('background-clip', 'padding-box', 'important');
    }}
    // Textarea: transparent
    if (ta) {{
      ta.style.setProperty('background', 'transparent', 'important');
      ta.style.setProperty('border', 'none', 'important');
      ta.style.setProperty('outline', 'none', 'important');
      ta.style.setProperty('box-shadow', 'none', 'important');
      ta.style.setProperty('color', '{C['text']}', 'important');
      ta.style.setProperty('caret-color', ACCENT, 'important');
    }}
    // Button wrapper: transparent — no border, no background
    if (bw) {{
      bw.style.setProperty('background', 'transparent', 'important');
      bw.style.setProperty('border', 'none', 'important');
      bw.style.setProperty('box-shadow', 'none', 'important');
    }}
    // Button: transparent, blue arrow
    if (btn) {{
      btn.style.setProperty('background', 'transparent', 'important');
      btn.style.setProperty('border', 'none', 'important');
      btn.style.setProperty('outline', 'none', 'important');
      btn.style.setProperty('box-shadow', 'none', 'important');
      btn.style.setProperty('color', ACCENT, 'important');
      btn.style.setProperty('display', 'flex', 'important');
      btn.style.setProperty('align-items', 'center', 'important');
      btn.style.setProperty('justify-content', 'center', 'important');
    }}


  }}
  inject();
  setTimeout(inject, 600);
  setTimeout(inject, 1800);
  setTimeout(inject, 4000);
}})();
</script>""",
    height=0,
)

# ---------------------------------------------------------------------------
# Header / Hero
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="hero-row" style="position:relative;display:flex;align-items:center;justify-content:flex-start;padding:2.0rem 0 1.2rem 0;min-height:60px;">
    <div class="hero-left">
        <div class="hero-icon-wrap">
            <img src="{ICON_URI}" width="44" height="44" alt="TráfegoAI icon"/>
        </div>
        <div class="hero-text-col">
            <div class="hero-title">TráfegoAI</div>
            <div class="hero-subtitle">Seu mentor de tr&aacute;fego pago</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

_status_placeholder = st.empty()

st.markdown(f'<div style="border-bottom:1px solid {C["border"]};margin-bottom:1.6rem;"></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Carregamento do índice e agente (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_agent():
    provider = _secret("AI_PROVIDER", "openai").lower()

    if provider == "gemini":
        api_key = _secret("GEMINI_API_KEY")
        if not api_key:
            return None, "GEMINI_API_KEY não configurada.\n\nObtenha gratuitamente em: https://aistudio.google.com/apikey"
        agent_model = _secret("AGENT_MODEL", "gemini-2.0-flash")
    else:
        api_key = _secret("OPENAI_API_KEY")
        if not api_key:
            return None, "OPENAI_API_KEY não configurada."
        agent_model = _secret("AGENT_MODEL", "gpt-4.1-mini")

    vectorstore_path = _secret("VECTORSTORE_PATH", "vectorstore")
    embedding_model = _secret("EMBEDDING_MODEL", "text-embedding-3-small")

    index_path = Path(vectorstore_path)
    if not index_path.exists() or not any(index_path.iterdir()):
        return None, (
            f"Índice FAISS não encontrado em '{index_path.resolve()}'.\n\n"
            "Execute os scripts de setup:\n"
            "1. `python scripts/01_generate_docs.py`\n"
            "2. `python scripts/02_build_index.py`"
        )

    try:
        vectorstore = load_index(
            vectorstore_path=vectorstore_path,
            embedding_model=embedding_model,
            api_key=api_key,
            provider=provider,
        )
        agent = build_agent_graph(
            vectorstore=vectorstore,
            agent_model=agent_model,
            api_key=api_key,
            provider=provider,
        )
        return agent, None
    except Exception as e:
        return None, f"Erro ao carregar o agente: {e}"


_loading = st.empty()
_loading.markdown('<div class="custom-loading"><div class="custom-loading-dot"></div>Carregando base de conhecimento...</div>', unsafe_allow_html=True)
agent, error_msg = load_agent()
_loading.empty()

# Atualiza badge de status no header
if error_msg:
    _status_placeholder.markdown(f"""
    <div style="display:flex;justify-content:flex-end;margin-top:-4.0rem;margin-bottom:0.5rem;">
        <div class="status-badge" style="background:rgba(239,68,68,0.08);border-color:rgba(239,68,68,0.2);">
            <div class="status-dot" style="background:#ef4444;box-shadow:0 0 8px #ef4444,0 0 16px #ef4444;animation:none;"></div>
            <span style="color:#f87171;">Offline</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    _status_placeholder.markdown(f"""
    <div style="display:flex;justify-content:flex-end;margin-top:-4.0rem;margin-bottom:0.5rem;">
        <div class="status-badge">
            <div class="status-dot"></div>
            Online
        </div>
    </div>
    """, unsafe_allow_html=True)

if error_msg:
    st.error(error_msg)
    st.stop()


# ---------------------------------------------------------------------------
# Histórico de mensagens
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibe histórico
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)


# ---------------------------------------------------------------------------
# Input do usuário
# ---------------------------------------------------------------------------
if prompt := st.chat_input("Faça sua pergunta aqui..."):
    # Exibe mensagem do usuário
    with st.chat_message("user"):
        st.markdown(prompt)

    # Adiciona ao histórico
    st.session_state.messages.append(HumanMessage(content=prompt))

    # Invoca o agente
    with st.chat_message("assistant"):
        with st.spinner("Consultando..."):
            try:
                result = agent.invoke(
                    {
                        "messages": st.session_state.messages,
                        "context": [],
                    }
                )
                response = result["messages"][-1].content

            except Exception as e:
                response = f"Erro ao processar a pergunta: {e}"
                result = {}

        st.markdown(response)

        # Referências abaixo da resposta
        if result.get("context"):
            seen = set()
            refs = []
            for doc in result["context"]:
                module = doc.metadata.get("module", "")
                lesson = doc.metadata.get("lesson", "")
                ref = f"{module} / {lesson}" if module and lesson else (module or lesson)
                if ref and ref not in seen:
                    refs.append(ref)
                    seen.add(ref)
            if refs:
                with st.expander("📚 Referências", expanded=False):
                    for ref in refs:
                        st.markdown(f"- {ref}")

    # Salva resposta no histórico
    st.session_state.messages.append(AIMessage(content=response))


# ---------------------------------------------------------------------------
# Sidebar - Ações e informações
# ---------------------------------------------------------------------------
with st.sidebar:
    # Toggle de tema
    st.markdown('<p class="sidebar-actions-title">Aparência</p>', unsafe_allow_html=True)
    dark_mode = st.toggle("Modo Escuro", value=_dark, key="theme_toggle")
    if dark_mode != _dark:
        st.session_state.theme = "dark" if dark_mode else "light"
        st.rerun()

    # Informações fixas no rodapé
    st.markdown(f"""
    <div class="sidebar-bottom-info">
        <div class="sidebar-logo-row">
            <img src="{ICON_URI}" width="24" height="24"
                 style="display:block; flex-shrink:0; min-width:24px; min-height:24px; background:#0f172a; border-radius:6px;"/>
            <span class="sidebar-logo-text">Tr&aacute;fegoAI</span>
        </div>
        <div class="sidebar-desc">
            Agente de IA especializado em tr&aacute;fego pago e estrat&eacute;gias de performance.
        </div>
        <div class="sidebar-tags">
            <span class="sidebar-tag">Facebook Ads</span>
            <span class="sidebar-tag">Google Ads</span>
            <span class="sidebar-tag">Performance</span>
        </div>
        <div class="sidebar-dev">Desenvolvido por Major &middot; AI Developer</div>
    </div>
    """, unsafe_allow_html=True)

    # Título Ações
    st.markdown('<p class="sidebar-actions-title sidebar-actions-title-spaced">Ações</p>', unsafe_allow_html=True)
    if st.button("Iniciar novo chat", key="clear_chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
