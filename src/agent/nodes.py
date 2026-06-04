"""
Nodes do grafo LangGraph para o TPMD Agent.

Nodes:
- retrieve: Busca os documentos mais relevantes no FAISS
- generate: Gera a resposta usando os documentos recuperados
"""

from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from src.agent.state import AgentState


SYSTEM_PROMPT = """Você é um especialista em tráfego pago, Facebook Ads, Google Ads e marketing digital de alta performance. Você tem anos de experiência prática e domina profundamente esse universo.

Você está numa conversa direta com alguém que quer aprender e crescer. Seu papel é orientar, ensinar e provocar reflexões — como um mentor que já trilhou esse caminho e quer ver a pessoa evoluir de verdade.

Como se comportar:
- Fale como alguém que simplesmente sabe o que está dizendo, de forma natural e direta
- NUNCA mencione "o curso", "módulo", "aula", "material" ou qualquer referência a uma fonte de conhecimento. Você só sabe — ponto
- Evite listas e formatação excessiva. Prefira parágrafos fluidos, como numa conversa real
- Use analogias e exemplos práticos para tornar os conceitos concretos
- Seja direto, sem enrolação, mas com calor humano
- Se não souber responder algo, diga com naturalidade, sem citar falta de material
- Responda sempre em português do Brasil
- Nunca pareça um chatbot, um documento ou um assistente virtual. Pareça uma pessoa real que domina o assunto

Conhecimento disponível para embasar suas respostas:
{context}"""


def make_retrieve_node(vectorstore: FAISS, k: int = 6):
    """
    Cria o node de retrieval com o vectorstore configurado.

    Args:
        vectorstore: Instância FAISS carregada
        k: Número de documentos a recuperar

    Returns:
        Função node para o grafo LangGraph.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    def retrieve(state: AgentState) -> AgentState:
        """Recupera documentos relevantes para a última mensagem do usuário."""
        last_human = next(
            (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            None,
        )
        if last_human is None:
            return {**state, "context": []}

        query = last_human.content
        docs = retriever.invoke(query)

        return {**state, "context": docs}

    return retrieve


def make_generate_node(
    model: str = "gpt-4.1-mini",
    api_key: str | None = None,
    provider: str = "openai",
):
    """
    Cria o node de geração de resposta.

    Args:
        model: Modelo a usar
        api_key: Chave da API
        provider: "openai" ou "gemini"

    Returns:
        Função node para o grafo LangGraph.
    """
    if provider == "gemini":
        gemini_model = model if model != "gpt-4.1-mini" else "gemini-2.0-flash"
        llm = ChatGoogleGenerativeAI(model=gemini_model, temperature=0.7, google_api_key=api_key)
    else:
        llm = ChatOpenAI(model=model, temperature=0.7, api_key=api_key)

    def generate(state: AgentState) -> AgentState:
        """Gera resposta usando os documentos recuperados."""
        docs = state.get("context", [])

        context_text = "\n\n---\n\n".join(
            f"[Módulo: {doc.metadata.get('module', 'N/A')} | Aula: {doc.metadata.get('lesson', 'N/A')}]\n{doc.page_content}"
            for doc in docs
        )

        system_msg = {"role": "system", "content": SYSTEM_PROMPT.format(context=context_text)}

        # Constrói histórico de mensagens
        history = [
            {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
            for m in state["messages"]
        ]

        response = llm.invoke([system_msg, *history])

        new_messages = list(state["messages"]) + [AIMessage(content=response.content)]
        return {**state, "messages": new_messages}

    return generate
