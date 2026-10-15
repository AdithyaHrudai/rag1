"""Numbered source tracking so answers can cite [1], [2], ... consistently"""

import re
from pathlib import Path
from typing import List

from langchain_core.documents import Document

_CITATION_RE = re.compile(r"\[(\d+)\]")


def describe_document(doc: Document) -> tuple:
    """Return a human-readable (title, location) pair for an indexed chunk"""
    meta = doc.metadata or {}
    source = str(meta.get("source", ""))
    if source.lower().endswith(".pdf"):
        name = Path(source).name
        page = meta.get("page")
        location = f"page {int(page) + 1}" if page is not None else source
        return name, location
    title = (meta.get("title") or source or "Document").strip()
    return title, source


class SourceRegistry:
    """Collects the passages used for one answer and gives each a stable number"""

    def __init__(self):
        self._sources: List[dict] = []
        self._ids = {}

    def add(self, doc: Document, source_type: str = "document") -> int:
        """
        Register a passage and return its citation number (duplicates reuse their number)

        Args:
            doc: Retrieved chunk or Wikipedia page
            source_type: "document" or "wikipedia"
        """
        if source_type == "wikipedia":
            meta = doc.metadata or {}
            title = meta.get("title", "Wikipedia")
            location = meta.get("source", "")
        else:
            title, location = describe_document(doc)

        key = (location, doc.page_content)
        if key not in self._ids:
            self._ids[key] = len(self._sources) + 1
            self._sources.append({
                "id": self._ids[key],
                "type": source_type,
                "title": title,
                "location": location,
                "content": doc.page_content,
                "cited": False,
            })
        return self._ids[key]

    def format(self, docs: List[Document], source_type: str = "document") -> str:
        """Register docs and render them as numbered passages for the LLM"""
        blocks = []
        for doc in docs:
            sid = self.add(doc, source_type)
            source = self._sources[sid - 1]
            blocks.append(f"[{sid}] {source['title']} ({source['location']})\n{doc.page_content}")
        return "\n\n".join(blocks)

    def finalize(self, answer: str) -> List[dict]:
        """Mark which sources the answer actually cites and return all of them"""
        cited = {int(n) for n in _CITATION_RE.findall(answer or "")}
        for source in self._sources:
            source["cited"] = source["id"] in cited
        return self._sources
