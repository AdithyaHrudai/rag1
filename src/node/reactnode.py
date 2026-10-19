"""LangGraph nodes for RAG workflow + ReAct Agent inside generate_content"""

import os
from typing import List, Optional

import wikipedia
from src.state.rag_state import RAGState
from src.node.sources import SourceRegistry, message_text

from langchain_core.documents import Document
from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

# Wikipedia tool
from langchain_community.utilities import WikipediaAPIWrapper

MAX_AGENT_STEPS = 9

SYSTEM_PROMPT = (
    "You are a helpful RAG agent. The question comes with numbered passages already "
    "retrieved from the user's documents. If they fully answer it, respond directly. "
    "If they do not, you MUST use your tools before answering: call 'retriever' to search "
    "the documents again with a different query, and call 'wikipedia' for general knowledge "
    "the documents don't cover. "
    "Cite every fact with the number of the passage it came from, like [1] or [2][3], "
    "and only use numbers you were given. Only say that you could not find the answer "
    "after you have tried the tools and they returned nothing relevant. "
    "Use at most two tool calls in total, then answer with the best evidence you have. "
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

    def _build_tools(self, registry: SourceRegistry) -> List[StructuredTool]:
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

        # Typed single-argument schemas: Gemini's function calling is unreliable with the
        # untyped schema the legacy Tool wrapper produces (MALFORMED_FUNCTION_CALL)
        retriever_tool = StructuredTool.from_function(
            func=retriever_tool_fn,
            name="retriever",
            description="Search the user's indexed documents. Returns numbered passages.",
        )
        wikipedia_tool = StructuredTool.from_function(
            func=wikipedia_tool_fn,
            name="wikipedia",
            description="Search Wikipedia for general knowledge. Returns numbered passages.",
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

        # Each tool round costs two graph steps; the limit keeps a free-tier API call budget in check
        result = agent.invoke(
            {"messages": [HumanMessage(content=content)]},
            {"recursion_limit": MAX_AGENT_STEPS},
        )

        messages = result.get("messages", [])
        answer = ""
        if messages:
            answer = message_text(getattr(messages[-1], "content", None))
        answer = answer or "Could not generate answer."

        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=answer,
            sources=registry.finalize(answer),
        )
