"""Unit tests for citation handling (no API key or network needed)"""

from langchain_core.documents import Document

from src.node.sources import SourceRegistry, describe_document, message_text


def test_message_text_passes_strings_through():
    assert message_text("plain answer [1]") == "plain answer [1]"


def test_message_text_flattens_content_blocks():
    blocks = [{"type": "text", "text": "Attention [1]"}, {"type": "text", "text": " works."}]
    assert message_text(blocks) == "Attention [1] works."


def test_message_text_ignores_non_text_blocks_and_none():
    assert message_text([{"type": "image", "url": "x"}]) == ""
    assert message_text(None) == ""


def test_describe_document_pdf_uses_file_name_and_one_based_page():
    doc = Document(page_content="x", metadata={"source": "data/attention.pdf", "page": 2})
    assert describe_document(doc) == ("attention.pdf", "page 3")


def test_describe_document_web_page_uses_title_and_url():
    doc = Document(page_content="x", metadata={"source": "https://example.com", "title": "Example"})
    assert describe_document(doc) == ("Example", "https://example.com")


def test_registry_numbers_passages_and_reuses_duplicates():
    registry = SourceRegistry()
    a = Document(page_content="alpha", metadata={"source": "a.pdf", "page": 0})
    b = Document(page_content="beta", metadata={"source": "b.pdf", "page": 0})
    assert registry.add(a) == 1
    assert registry.add(b) == 2
    assert registry.add(a) == 1


def test_finalize_marks_only_cited_sources():
    registry = SourceRegistry()
    registry.add(Document(page_content="alpha", metadata={"source": "a.pdf", "page": 0}))
    registry.add(Document(page_content="beta", metadata={"source": "b.pdf", "page": 0}))
    sources = registry.finalize("Alpha is true [1].")
    assert [s["cited"] for s in sources] == [True, False]


def test_finalize_handles_empty_answer():
    registry = SourceRegistry()
    registry.add(Document(page_content="alpha", metadata={"source": "a.pdf", "page": 0}))
    assert registry.finalize("")[0]["cited"] is False


def test_finalize_recognises_grouped_citations():
    registry = SourceRegistry()
    for i in range(1, 5):
        registry.add(Document(page_content=f"passage {i}", metadata={"source": f"{i}.pdf", "page": 0}))
    sources = registry.finalize("First [3]. Then both [2, 4] and also [1][3].")
    assert [s["cited"] for s in sources] == [True, True, True, True]


def test_finalize_ignores_non_citation_brackets():
    registry = SourceRegistry()
    registry.add(Document(page_content="alpha", metadata={"source": "a.pdf", "page": 0}))
    assert registry.finalize("Lists look like [a] or [1 2] here.")[0]["cited"] is False
