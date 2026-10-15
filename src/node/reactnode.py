"""LangGraph nodes for RAG workflow + ReAct Agent inside generate_content"""

import os
from typing import List, Optional

import wikipedia
from src.state.rag_state import RAGState
from src.node.sources import SourceRegistry

from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

# Wikipedia tool
from langchain_community.utilities import WikipediaAPIWrapper

SYSTEM_PROMPT = (
    "You are a helpful RAG agent. The question comes with numbered passages already "
    "retrieved from the user's documents. If they answer it, respond directly. "
    "Otherwise call 'retriever' to search the documents with a different query, or "
    "'wikipedia' for general knowledge the documents don't cover. "
    "Cite every fact with the number of the passage it came from, like [1] or [2][3], "
    "and only use numbers you were given. If no source contains the answer, say so. "
    "Return only the final useful answer."
)


class RAGNodes:
    """Contains node functions for RAG workflow"""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.wikipedia = WikipediaAPIWrapper(
            top_k_results=2, doc_content_chars_max=2000, lang="en"
        )
        # Wikipedia rate-limits the library's shared default User-Agent (HTTP 429)
        wikipedia.set_user_agent(os.environ.get("USER_AGENT", "rag1-agentic-rag/0.1"))

    def retrieve_docs(self, state: RAGState) -> RAGState:
        """Classic retriever node"""
        docs = self.retriever.invoke(state.question)
        return RAGState(
            question=state.question,
            retrieved_docs=docs
        )

    def _build_tools(self, registry: SourceRegistry) -> List[Tool]:
        """Build retriever + wikipedia tools that number their results in the registry"""

        def retriever_tool_fn(query: str) -> str:
            docs: List[Document] = self.retriever.invoke(query)
            if not docs:
                return "No documents found."
            return registry.format(docs, "document")

        def wikipedia_tool_fn(query: str) -> str:
            try:
                pages = self.wikipedia.load(query)
            except Exception as e:
                return f"Wikipedia search failed: {e}"
            if not pages:
                return "No Wikipedia results found."
            return registry.format(pages, "wikipedia")

        retriever_tool = Tool(
            name="retriever",
            description="Search the user's indexed documents. Returns numbered passages.",
            func=retriever_tool_fn,
        )
        wikipedia_tool = Tool(
            name="wikipedia",
            description="Search Wikipedia for general knowledge. Returns numbered passages.",
            func=wikipedia_tool_fn,
        )

        return [retriever_tool, wikipedia_tool]

    def generate_answer(self, state: RAGState) -> RAGState:
        """
        Generate answer using ReAct agent with retriever + wikipedia.
        The agent is built per question so each answer has its own source numbering.
        """
        registry = SourceRegistry()
        context = registry.format(state.retrieved_docs, "document")
        agent = create_react_agent(
            self.llm, tools=self._build_tools(registry), prompt=SYSTEM_PROMPT
        )

        content = f"Question: {state.question}\n\n"
        if context:
            content += f"Passages retrieved from the documents:\n\n{context}"
        else:
            content += "No passages were retrieved from the documents."

        result = agent.invoke({"messages": [HumanMessage(content=content)]})

        messages = result.get("messages", [])
        answer: Optional[str] = None
        if messages:
            answer_msg = messages[-1]
            answer = getattr(answer_msg, "content", None)
        answer = answer or "Could not generate answer."

        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=answer,
            sources=registry.finalize(answer),
        )
