"""
Interface Streamlit do TrafegoAI - Agente RAG de Tráfego Pago.

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
      <stop offset="0%" stop-color="#1e3a8a" stop-opacity="0.5"/>
      <stop offset="100%" stop-color="#1d4ed8" stop-opacity="0.2"/>
    </linearGradient>
    <linearGradient id="tai-bd" x1="0" y1="48" x2="48" y2="0" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="#3b82f6"/>
      <stop offset="100%" stop-color="#93c5fd"/>
    </linearGradient>
    <linearGradient id="tai-ar" x1="8" y1="36" x2="40" y2="12" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="#3b82f6"/>
      <stop offset="100%" stop-color="#bae6fd"/>
    </linearGradient>
  </defs>
  <rect x="2" y="2" width="44" height="44" rx="12" fill="url(#tai-bg)" stroke="url(#tai-bd)" stroke-width="1.5"/>
  <polyline points="9,35 18,24 25,29 36,16" stroke="url(#tai-ar)" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
  <polyline points="30,13 37,16 34,23" stroke="url(#tai-ar)" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
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
    page_title="TrafegoAI",
    page_icon="📈",
    layout="wide",
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
    /* descendant * keeps app_bg on ALL wrappers */
    [data-testid="stBottom"] div {{
        background-color: {C['app_bg']} !important;
        box-shadow: none !important;
    }}
    /* chat input visual box */
    [data-testid="stBottom"] [data-testid="stChatInput"] > div {{
        background-color: {C['input_bg']} !important;
        border: 1.5px solid rgba({C['accent_rgb']},0.5) !important;
        border-radius: 14px !important;
        box-shadow: none !important;
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
        background-color: {C['input_bg']} !important;
        color: {C['text']} !important;
        caret-color: {C['accent']} !important;
    }}
    /* specificity (0,2,1) for submit button */
    [data-testid="stBottom"] [data-testid="stChatInputSubmitButton"] button {{
        background-color: {C['accent']} !important;
        border: none !important;
        border-radius: 8px !important;
        box-shadow: 0 0 10px rgba({C['accent_rgb']},0.55),
                    0 0 20px rgba({C['accent_rgb']},0.28) !important;
    }}
    [data-testid="stBottom"] [data-testid="stChatInputSubmitButton"] button * {{
        color: #fff !important;
        fill: #fff !important;
        stroke: #fff !important;
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
    .hero-icon-wrap {{ flex-shrink: 0; width: 44px; height: 44px; line-height: 0; overflow: visible !important; }}
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
    /* textarea padding */
    [data-testid="stChatInputTextArea"] {{
        padding: 0.65rem 1rem !important;
    }}
    /* Chat input visual box */
    [data-testid="stChatInput"] > div {{
        background: {C['input_bg']} !important;
        border: 1.5px solid rgba({C['accent_rgb']},0.5) !important;
        border-radius: 14px !important;
        box-shadow: none !important;
    }}
    [data-testid="stChatInput"] > div:focus-within {{
        border-color: {C['accent']} !important;
        box-shadow: 0 0 0 3px rgba({C['accent_rgb']},0.15),
                    0 0 22px rgba({C['accent_rgb']},0.2) !important;
    }}
    button[data-testid="stChatInputSubmitButton"] {{
        background: transparent !important;
        border: none !important;
        border-radius: 8px !important;
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
        fill: currentColor !important;
        stroke: currentColor !important;
        width: 20px !important;
        height: 20px !important;
    }}
    .sidebar-card {{
        background: {C['card_bg']};
        border: 1px solid {C['border']};
        border-radius: 12px; padding: 1rem;
        box-shadow: {C['shadow']};
    }}
    .sidebar-logo-row {{ display: flex; align-items: center; gap: 9px; margin-bottom: 0.5rem; }}
    .sidebar-logo-text {{
        font-size: 1rem; font-weight: 700;
        background: {C['title_grad']};
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    }}
    .sidebar-desc {{ font-size: 0.77rem; color: {C['text_muted']}; line-height: 1.6; margin-bottom: 0.5rem; }}
    .sidebar-tags {{ display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 0.6rem; }}
    .sidebar-tag {{
        background: rgba({C['accent_rgb']},0.08);
        border: 1px solid rgba({C['accent_rgb']},0.2);
        border-radius: 6px; padding: 2px 8px;
        font-size: 0.67rem; color: {C['tag_text']}; font-weight: 500;
    }}
    .sidebar-dev {{
        font-size: 0.68rem; color: {C['text_muted']}; opacity: 0.6;
        padding-top: 0.5rem; border-top: 1px solid {C['divider']};
    }}
    .stButton button {{
        background: {C['btn_bg']} !important;
        border: 1px solid {C['btn_brd']} !important;
        color: {C['btn_txt']} !important;
        border-radius: 8px !important; font-weight: 500 !important;
        white-space: nowrap !important;
        transition: all 0.15s ease !important;
    }}
    .stButton button:hover {{
        background: rgba({C['accent_rgb']},0.15) !important;
        border-color: {C['accent']} !important;
    }}
    hr {{ border-color: {C['divider']} !important; }}
    ::-webkit-scrollbar {{ width: 5px; }}
    ::-webkit-scrollbar-thumb {{ background: rgba({C['accent_rgb']},0.25); border-radius: 3px; }}
{_extra_css}
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
/* reset all divs inside stBottom, but NOT buttons/svg/img */
[data-testid="stBottom"] div {{
  background-color: {C['app_bg']} !important;
  box-shadow: none !important;
}}
[data-testid="stBottom"] [data-testid="stChatInput"] > div {{
  background-color: {C['input_bg']} !important;
  border: 1.5px solid rgba({C['accent_rgb']},0.55) !important;
  border-radius: 14px !important;
  box-shadow: none !important;
}}
[data-testid="stBottom"] [data-testid="stChatInput"] > div:focus-within {{
  border-color: {C['accent']} !important;
  box-shadow: 0 0 0 3px rgba({C['accent_rgb']},0.18),
              0 0 24px rgba({C['accent_rgb']},0.22) !important;
}}
[data-testid="stBottom"] [data-testid="stChatInput"] > div > div {{
  background-color: {C['input_bg']} !important;
}}
/* wrapper que contém o botão deve ser transparente */
[data-testid="stBottom"] [data-testid="stChatInput"] > div > div:has(button) {{
  background-color: transparent !important;
}}
[data-testid="stChatInputTextArea"] {{
  padding: 0.65rem 1rem !important;
}}
[data-testid="stBottom"] textarea {{
  background-color: {C['input_bg']} !important;
  color: {C['text']} !important;
  caret-color: {C['accent']} !important;
}}
/* submit button: transparent bg, blue arrow only */
button[data-testid="stChatInputSubmitButton"] {{
  background: transparent !important;
  border: none !important;
  border-radius: 8px !important;
  box-shadow: none !important;
  color: {C['accent']} !important;
  transition: color 0.15s ease, filter 0.15s ease !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
}}
button[data-testid="stChatInputSubmitButton"]:hover {{
  color: #93c5fd !important;
  filter: drop-shadow(0 0 6px rgba({C['accent_rgb']},0.7)) !important;
}}
button[data-testid="stChatInputSubmitButton"] svg {{
  fill: currentColor !important;
  stroke: currentColor !important;
  width: 20px !important;
  height: 20px !important;
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
  }}
  inject();
  setTimeout(inject, 600);
  setTimeout(inject, 1800);
  setTimeout(inject, 4000);
}})();
</script>""",
    height=0,
)

st.markdown(f"""
<div class="hero">
    <div class="hero-left">
        <div class="hero-icon-wrap">
            <img src="{ICON_URI}" width="44" height="44" alt="TrafegoAI icon"/>
        </div>
        <div class="hero-text-col">
            <div class="hero-title">TrafegoAI</div>
            <div class="hero-subtitle">Seu mentor de tr&aacute;fego pago</div>
        </div>
    </div>
    <div class="status-badge">
        <div class="status-dot"></div>
        Online
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Carregamento do índice e agente (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Carregando base de conhecimento...")
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


agent, error_msg = load_agent()

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

                # Exibe fontes consultadas (opcional, na sidebar)
                if result.get("context"):
                    with st.sidebar:
                        st.subheader("📚 Fontes consultadas")
                        seen = set()
                        for doc in result["context"]:
                            ref = f"**{doc.metadata.get('module', '')}** / {doc.metadata.get('lesson', '')}"
                            if ref not in seen:
                                st.markdown(f"- {ref}")
                                seen.add(ref)

            except Exception as e:
                response = f"Erro ao processar a pergunta: {e}"

        st.markdown(response)

    # Salva resposta no histórico
    st.session_state.messages.append(AIMessage(content=response))


# ---------------------------------------------------------------------------
# Sidebar - Informações
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"""
    <div style="padding: 0.5rem 0 0.2rem 0;">
        <div class="sidebar-card">
            <div class="sidebar-logo-row">
                <img src="{ICON_URI}" width="24" height="24"
                     style="display:block; flex-shrink:0; min-width:24px; min-height:24px;"/>
                <span class="sidebar-logo-text">TrafegoAI</span>
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
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button(f"{C['theme_icon']} {C['theme_label']}", use_container_width=True):
            st.session_state.theme = C['next_theme']
            st.rerun()
    with col2:
        if st.button("🗑️ Limpar", use_container_width=True, help="Apagar todo o histórico da conversa"):
            st.session_state.messages = []
            st.rerun()
