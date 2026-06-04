"""
Grafo LangGraph do TPMD Agent.

Fluxo:
  START -> retrieve -> generate -> END
"""

from langchain_community.vectorstores import FAISS
from langgraph.graph import END, START, StateGraph

from src.agent.state import AgentState
from src.agent.nodes import make_retrieve_node, make_generate_node


def build_agent_graph(
    vectorstore: FAISS,
    agent_model: str = "gpt-4.1-mini",
    api_key: str | None = None,
    provider: str = "openai",
    k: int = 6,
):
    """
    Constrói e compila o grafo do agente RAG.

    Args:
        vectorstore: Índice FAISS carregado
        agent_model: Modelo para geração de respostas
        api_key: Chave da API
        provider: "openai" ou "gemini"
        k: Número de documentos a recuperar por consulta

    Returns:
        Grafo compilado (CompiledGraph).
    """
    retrieve = make_retrieve_node(vectorstore, k=k)
    generate = make_generate_node(model=agent_model, api_key=api_key, provider=provider)

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()
