"""Configuration module for Agentic RAG system"""

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# Load environment variables
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

class Config:
    """Configuration class for RAG system"""

    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Model Configuration
    LLM_MODEL = os.getenv("LLM_MODEL", "openai:gpt-4o")

    # Document Processing
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    # Local data: every PDF in DATA_DIR is indexed, URLS_FILE lists web pages (one per line)
    DATA_DIR = PROJECT_ROOT / "data"
    URLS_FILE = DATA_DIR / "url.txt"

    # Default URLs (used when URLS_FILE is missing or empty)
    DEFAULT_URLS = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/"
    ]

    @classmethod
    def get_sources(cls):
        """Return the list of sources to index: URLs + the local PDF directory"""
        urls = []
        if cls.URLS_FILE.exists():
            with open(cls.URLS_FILE, "r", encoding="utf-8") as f:
                urls = [line.strip() for line in f if line.strip()]
        sources = urls or list(cls.DEFAULT_URLS)
        if cls.DATA_DIR.is_dir() and any(cls.DATA_DIR.glob("*.pdf")):
            sources.append(str(cls.DATA_DIR))
        return sources

    @classmethod
    def get_llm(cls):
        """Initialize and return the LLM model"""
        if not cls.OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        os.environ["OPENAI_API_KEY"] = cls.OPENAI_API_KEY
        return init_chat_model(cls.LLM_MODEL)
