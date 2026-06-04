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
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

# ---------------------------------------------------------------------------
# Ícone SVG como data URI (evita sanitização do Streamlit)
# ---------------------------------------------------------------------------
_SVG_ICON = """<svg width="48" height="48" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="bg" x1="0" y1="48" x2="48" y2="0" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="#1e3a8a" stop-opacity="0.5"/>
      <stop offset="100%" stop-color="#1d4ed8" stop-opacity="0.2"/>
    </linearGradient>
    <linearGradient id="bd" x1="0" y1="48" x2="48" y2="0" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="#3b82f6"/>
      <stop offset="100%" stop-color="#93c5fd"/>
    </linearGradient>
    <linearGradient id="ar" x1="8" y1="36" x2="40" y2="12" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="#3b82f6"/>
      <stop offset="100%" stop-color="#bae6fd"/>
    </linearGradient>
  </defs>
  <rect x="2" y="2" width="44" height="44" rx="12" fill="url(#bg)" stroke="url(#bd)" stroke-width="1.5"/>
  <polyline points="9,35 18,24 25,29 36,16" stroke="url(#ar)" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
  <polyline points="30,13 37,16 34,23" stroke="url(#ar)" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
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

st.markdown(
    f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
        }}

        .block-container {{
            padding-top: 2rem !important;
        }}

        /* Cabeçalho */
        .hero {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 1.4rem 0 1.2rem 0;
            border-bottom: 1px solid rgba(128,128,128,0.15);
            margin-bottom: 2rem;
        }}
        .hero-left {{
            display: flex;
            flex-direction: column;
            gap: 3px;
        }}
        .hero-title-row {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .hero-icon-wrap {{
            flex-shrink: 0;
        }}
        .hero-title {{
            font-size: 1.9rem;
            font-weight: 800;
            letter-spacing: -0.8px;
            color: var(--primary-color);
            line-height: 1.1;
        }}
        .hero-subtitle {{
            font-size: 0.85rem;
            color: var(--text-color);
            opacity: 0.5;
            letter-spacing: 0.3px;
        }}
        .status-badge {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(34,197,94,0.1);
            border: 1px solid rgba(34,197,94,0.3);
            border-radius: 20px;
            padding: 4px 12px;
            font-size: 0.72rem;
            color: #15803d;
            font-weight: 500;
        }}
        .status-dot {{
            width: 6px;
            height: 6px;
            background: #16a34a;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }}
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.4; }}
        }}

        /* Chat */
        .stChatInputContainer {{
            border: 1px solid rgba(128,128,128,0.2) !important;
            border-radius: 12px !important;
        }}
        .stChatInputContainer:focus-within {{
            border-color: var(--primary-color) !important;
            box-shadow: 0 0 0 3px rgba(59,130,246,0.1) !important;
        }}

        /* Sidebar */
        .sidebar-card {{
            background: var(--secondary-background-color);
            border: 1px solid rgba(128,128,128,0.12);
            border-radius: 10px;
            padding: 0.9rem;
        }}
        .sidebar-logo {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 1rem;
            font-weight: 700;
            color: var(--primary-color);
            margin-bottom: 0.4rem;
        }}
        .sidebar-desc {{
            font-size: 0.78rem;
            color: var(--text-color);
            opacity: 0.5;
            line-height: 1.6;
        }}
        .sidebar-tags {{
            margin-top: 0.6rem;
            display: flex;
            flex-wrap: wrap;
            gap: 4px;
        }}
        .sidebar-tag {{
            background: rgba(128,128,128,0.08);
            border: 1px solid rgba(128,128,128,0.18);
            border-radius: 5px;
            padding: 2px 7px;
            font-size: 0.68rem;
            color: var(--primary-color);
        }}
        .sidebar-dev {{
            font-size: 0.7rem;
            color: var(--text-color);
            opacity: 0.3;
            margin-top: 0.7rem;
            padding-top: 0.5rem;
            border-top: 1px solid rgba(128,128,128,0.1);
        }}
        .stButton > button {{
            border-radius: 8px !important;
        }}
    </style>

    <div class="hero">
        <div class="hero-left">
            <div class="hero-title-row">
                <div class="hero-icon-wrap">
                    <img src="{ICON_URI}" width="48" height="48" style="display:block; border-radius:12px;"/>
                </div>
                <div class="hero-title">TrafegoAI</div>
            </div>
            <div class="hero-subtitle">Seu mentor de tráfego pago</div>
        </div>
        <div class="status-badge">
            <div class="status-dot"></div>
            Online
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


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
    st.markdown(
        f"""
        <div style="padding: 0.8rem 0 0.5rem 0;">
            <div class="sidebar-card">
                <div class="sidebar-logo">
                    <img src="{ICON_URI}" width="20" height="20" style="display:inline-block; vertical-align:middle; border-radius:5px;"/>TrafegoAI
                </div>
                <div class="sidebar-desc">
                    Agente de IA especializado em tráfego pago e estratégias de performance.
                </div>
                <div class="sidebar-tags">
                    <span class="sidebar-tag">Facebook Ads</span>
                    <span class="sidebar-tag">Google Ads</span>
                    <span class="sidebar-tag">Performance</span>
                </div>
                <div class="sidebar-dev">Desenvolvido por Major · AI Developer</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    if st.button("Limpar conversa"):
        st.session_state.messages = []
        st.rerun()
