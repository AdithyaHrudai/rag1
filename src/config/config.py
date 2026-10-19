"""Configuration module for Agentic RAG system"""

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Identify the app to websites and Wikipedia (WebBaseLoader warns when unset; Wikipedia throttles anonymous clients)
os.environ.setdefault("USER_AGENT", "rag1-agentic-rag/0.1 (https://github.com/AdithyaHrudai/rag1)")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

class Config:
    """Configuration class for RAG system"""

    # API Keys (Gemini has a free tier: https://aistudio.google.com/apikey)
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    # Model Configuration (chat runs on Gemini, embeddings run locally and need no key)
    LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

    # Document Processing
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    # Local data: every PDF in DATA_DIR and UPLOAD_DIR is indexed, URLS_FILE lists web pages (one per line)
    DATA_DIR = PROJECT_ROOT / "data"
    URLS_FILE = DATA_DIR / "url.txt"
    UPLOAD_DIR = DATA_DIR / "uploads"
    
    # Saved FAISS index (rebuilt automatically when the sources change)
    INDEX_DIR = PROJECT_ROOT / "faiss_index"

    # Default URLs (used when URLS_FILE is missing or empty)
    DEFAULT_URLS = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/"
    ]

    @classmethod
    def get_sources(cls):
        """Return the list of sources to index: URLs + every PDF in data/ and data/uploads/"""
        urls = []
        if cls.URLS_FILE.exists():
            with open(cls.URLS_FILE, "r", encoding="utf-8") as f:
                urls = [line.strip() for line in f if line.strip()]
        sources = urls or list(cls.DEFAULT_URLS)
        for folder in (cls.DATA_DIR, cls.UPLOAD_DIR):
            if folder.is_dir():
                sources.extend(str(pdf) for pdf in sorted(folder.glob("*.pdf")))
        return sources

    @classmethod
    def get_llm(cls):
        """Initialize and return the LLM model"""
        if not cls.GOOGLE_API_KEY:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        # Few retries: a quota error (429) won't clear by retrying, and the UI should fail fast
        return ChatGoogleGenerativeAI(
            model=cls.LLM_MODEL,
            google_api_key=cls.GOOGLE_API_KEY,
            temperature=0,
            max_retries=2,
        )
