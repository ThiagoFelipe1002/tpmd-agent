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

# Palavras-chave que indicam perguntas dentro do escopo
_TOPIC_KEYWORDS = [
    "tráfego", "trafego", "ads", "anúncio", "anuncio", "campanha", "pixel",
    "facebook", "meta", "google", "instagram", "conversão", "conversao",
    "cpc", "cpm", "ctr", "roas", "roi", "funil", "lead", "público", "publico",
    "remarketing", "retargeting", "criativo", "copy", "landing page", "lp",
    "orçamento", "orcamento", "escala", "otimização", "otimizacao", "bid",
    "lance", "segmentação", "segmentacao", "lookalike", "custom audience",
    "público semelhante", "evento", "api de conversão", "capi",
    "gerenciador", "business manager", "bm", "conta de anúncio",
    "performance", "marketing digital", "gestão de tráfego", "gestor de tráfego",
    "mídia paga", "midia paga", "teste a/b", "crm", "utm", "atribuição",
    "métricas", "metricas", "dashboard", "relatório", "relatorio",
    "verba", "investimento", "cliente", "agência", "agencia",
    "proposta", "precificação", "precificacao", "nicho",
    "youtube ads", "tiktok ads", "linkedin ads", "pinterest ads",
    "rede de display", "rede de pesquisa", "search", "display",
    "shopping", "pmax", "performance max", "broad", "frase", "exata",
    "palavra-chave", "palavra chave", "negativa", "extensão",
    "qualidade", "relevância", "relevancia", "cpa", "cpv", "cpl",
    "estrutura", "conjunto de anúncio", "adset", "ad set",
    "objetivo", "alcance", "engajamento", "mensagem", "catálogo",
    "pixel do facebook", "tag do google", "gtm", "google tag manager",
    "super ads", "superads",
]

_OFF_TOPIC_RESPONSE = (
    "Esse assunto está fora do meu escopo. Posso ajudar com questões "
    "relacionadas a tráfego pago, Facebook Ads, Google Ads e estratégias "
    "de performance em marketing digital."
)

_CONFIRM_WORDS = {"sim", "isso", "exato", "isso mesmo", "s", "yes", "correto", "isso aí", "pode ser", "é isso"}


def _is_on_topic(text: str) -> bool:
    """Verifica se a pergunta é sobre tráfego pago / marketing digital."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in _TOPIC_KEYWORDS)


def _find_typo_suggestion(text: str) -> str | None:
    """Se a pergunta parece off-topic, verifica se alguma palavra é um possível erro de digitação."""
    import difflib
    words = text.lower().split()
    for word in words:
        if len(word) < 3:
            continue
        matches = difflib.get_close_matches(word, _TOPIC_KEYWORDS, n=1, cutoff=0.7)
        if matches:
            return matches[0]
    return None


SYSTEM_PROMPT = """Você é um especialista em tráfego pago, Facebook Ads, Google Ads e marketing digital de alta performance, com vasta experiência prática nessa área.

Seu papel é orientar e ensinar com profundidade, baseando suas respostas exclusivamente no conhecimento disponível abaixo. Seja preciso, claro e aprofundado, sem ser prolixo.

Diretrizes de comportamento:
- Use linguagem formal e profissional. Evite gírias, expressões informais ou coloquialismos como "sacou?", "show", "cara", "top" e similares
- Nunca mencione "o curso", "módulo", "aula", "material" ou qualquer referência a uma fonte de conhecimento. Apresente o conhecimento como seu
- Prefira parágrafos coesos a listas extensas. Use listas apenas quando a estrutura realmente ajudar a clareza
- Use exemplos práticos e objetivos para tornar os conceitos concretos
- Seja direto e aprofundado, sem enrolação e sem respostas superficiais
- Responda sempre em português do Brasil
- Se a pergunta não tiver relação com tráfego pago, marketing digital ou gestão de campanhas, responda educadamente que o assunto está fora do seu escopo e informe que pode ajudar com questões relacionadas a tráfego pago, Facebook Ads, Google Ads e estratégias de performance
- Se o conhecimento disponível não for suficiente para responder com precisão, diga isso de forma direta, sem inventar informações

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

    def _extract_suggestion_from_last_ai(messages) -> str | None:
        """Se a última mensagem do assistente foi uma sugestão de typo, extrai o termo sugerido."""
        for m in reversed(messages):
            if isinstance(m, AIMessage):
                if 'Você quis dizer "' in m.content:
                    start = m.content.index('Você quis dizer "') + len('Você quis dizer "')
                    end = m.content.index('"', start)
                    return m.content[start:end]
                break
        return None

    def retrieve(state: AgentState) -> AgentState:
        """Recupera documentos relevantes para a última mensagem do usuário."""
        last_human = next(
            (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            None,
        )
        if last_human is None:
            return {**state, "context": []}

        query = last_human.content

        # Se o usuário confirmou uma sugestão anterior, usa o termo sugerido
        if query.strip().lower() in _CONFIRM_WORDS:
            suggested = _extract_suggestion_from_last_ai(state["messages"])
            if suggested:
                query = suggested

        # Se a pergunta não é sobre o tema, não consulta o índice
        if not _is_on_topic(query):
            return {**state, "context": []}

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

        # Se não há documentos e a pergunta é off-topic, retorna resposta padrão
        last_human = next(
            (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            None,
        )
        if not docs and last_human and not _is_on_topic(last_human.content):
            # Não tratar como off-topic se o usuário estava confirmando uma sugestão
            is_confirmation = last_human.content.strip().lower() in _CONFIRM_WORDS
            if not is_confirmation:
                suggestion = _find_typo_suggestion(last_human.content)
                if suggestion:
                    reply = f"Não encontrei exatamente o que você digitou. Você quis dizer \"{suggestion}\"?"
                else:
                    reply = _OFF_TOPIC_RESPONSE
                new_messages = list(state["messages"]) + [AIMessage(content=reply)]
                return {**state, "messages": new_messages}

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
