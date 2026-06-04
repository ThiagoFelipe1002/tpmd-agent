"""
Estado do agente LangGraph.
"""

from typing import TypedDict
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document


class AgentState(TypedDict):
    messages: list[BaseMessage]
    context: list[Document]
