"""Unit tests for configuration (no API key or network needed)"""

import pytest

from src.config.config import Config


def test_get_llm_requires_api_key(monkeypatch):
    monkeypatch.setattr(Config, "GOOGLE_API_KEY", None)
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        Config.get_llm()


def test_get_sources_lists_urls_and_pdfs(tmp_path, monkeypatch):
    (tmp_path / "doc.pdf").write_bytes(b"%PDF-1.4")
    urls = tmp_path / "url.txt"
    urls.write_text("https://example.com/a\n\nhttps://example.com/b\n", encoding="utf-8")
    monkeypatch.setattr(Config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(Config, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(Config, "URLS_FILE", urls)

    sources = Config.get_sources()

    assert sources[:2] == ["https://example.com/a", "https://example.com/b"]
    assert str(tmp_path / "doc.pdf") in sources


def test_get_sources_falls_back_to_default_urls(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(Config, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(Config, "URLS_FILE", tmp_path / "missing.txt")
    assert Config.get_sources() == Config.DEFAULT_URLS
