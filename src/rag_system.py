"""AgenticRAG: wires config, ingestion, the vector store and the LangGraph workflow together"""

import json
from pathlib import Path
from typing import List, Optional

from src.config.config import Config, PROJECT_ROOT
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder

MANIFEST_NAME = "manifest.json"


class AgenticRAG:
    """Main Agentic RAG application"""

    def __init__(self, urls: Optional[List[str]] = None):
        """
        Initialize Agentic RAG system

        Args:
            urls: List of sources (URLs / local paths) to process (uses Config.get_sources() if None)
        """
        print("🚀 Initializing Agentic RAG System...")

        # Use configured sources (data/url.txt + PDFs in data/ and data/uploads/) if none provided
        self.urls = list(urls or Config.get_sources())
        self.index_dir = Path(Config.INDEX_DIR)

        # Initialize components
        self.llm = Config.get_llm()
        self.doc_processor = DocumentProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        self.vector_store = VectorStore(embedding_model=Config.EMBEDDING_MODEL)

        # Load the saved index, or process documents and build a new one
        self._setup_vectorstore()

        # Build graph
        self.graph_builder = GraphBuilder(
            retriever=self.vector_store.get_retriever(),
            llm=self.llm
        )
        self.graph_builder.build()

        print("✅ System initialized successfully!\n")

    def _signature(self) -> dict:
        """
        Describe everything the index depends on. If any of it changes
        (sources, file sizes/dates, chunking, embedding model) the index is rebuilt.
        """
        entries = []
        for src in self.urls:
            path = Path(src)
            if path.is_file():
                stat = path.stat()
                try:
                    name = path.resolve().relative_to(PROJECT_ROOT).as_posix()
                except ValueError:
                    name = path.resolve().as_posix()
                entries.append(f"{name}|{stat.st_size}|{int(stat.st_mtime)}")
            else:
                entries.append(src)
        return {
            "sources": sorted(entries),
            "chunk_size": Config.CHUNK_SIZE,
            "chunk_overlap": Config.CHUNK_OVERLAP,
            "embedding_model": Config.EMBEDDING_MODEL,
        }

    def _manifest_matches(self) -> bool:
        """Check whether the saved index was built from the current sources"""
        manifest = self.index_dir / MANIFEST_NAME
        if not manifest.exists():
            return False
        try:
            saved = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return saved == self._signature()

    def _save_index(self):
        """Write the FAISS index and the manifest describing its sources"""
        self.vector_store.save(self.index_dir)
        (self.index_dir / MANIFEST_NAME).write_text(
            json.dumps(self._signature(), indent=2), encoding="utf-8"
        )

    def _setup_vectorstore(self):
        """Load the saved index if it is up to date, otherwise build and save it"""
        if self._manifest_matches():
            try:
                self.vector_store.load(self.index_dir)
                print(f"💾 Loaded saved index ({self.vector_store.size()} chunks) from {self.index_dir.name}/")
                return
            except Exception as e:
                print(f"⚠️ Could not load saved index ({e}); rebuilding...")

        print(f"📄 Processing {len(self.urls)} sources...")
        documents = self.doc_processor.process_urls(self.urls)
        print(f"📊 Created {len(documents)} document chunks")

        print("🔍 Creating vector store...")
        self.vector_store.create_vectorstore(documents)
        self._save_index()
        print(f"💾 Saved index to {self.index_dir.name}/")

    def add_pdf(self, file_path) -> int:
        """
        Add a PDF to the knowledge base without restarting

        Args:
            file_path: Path to the PDF file

        Returns:
            Number of chunks added
        """
        file_path = Path(file_path)
        chunks = self.doc_processor.split_documents(
            self.doc_processor.load_from_pdf(file_path)
        )
        if not chunks:
            raise ValueError(f"No text found in {file_path.name} (is it a scanned PDF?)")

        self.vector_store.add_documents(chunks)
        if str(file_path) not in self.urls:
            self.urls.append(str(file_path))
        self._save_index()
        print(f"➕ Added {file_path.name}: {len(chunks)} chunks")
        return len(chunks)

    def chunk_count(self) -> int:
        """Number of chunks currently indexed"""
        return self.vector_store.size()

    def query(self, question: str) -> dict:
        """
        Run the RAG workflow and return the full result

        Args:
            question: User question

        Returns:
            Final state: question, answer, retrieved_docs, sources
        """
        return self.graph_builder.run(question)

    def ask(self, question: str) -> str:
        """
        Ask a question to the RAG system and print the answer with its sources

        Args:
            question: User question

        Returns:
            Generated answer
        """
        print(f"❓ Question: {question}\n")
        print("🤔 Processing...")

        result = self.query(question)
        answer = result['answer']

        print(f"✅ Answer: {answer}\n")
        cited = [s for s in result.get('sources', []) if s['cited']]
        if cited:
            print("📚 Sources:")
            for s in cited:
                print(f"  [{s['id']}] {s['title']} — {s['location']}")
            print()
        return answer

    def interactive_mode(self):
        """Run in interactive mode"""
        print("💬 Interactive Mode - Type 'quit' to exit\n")

        while True:
            question = input("Enter your question: ").strip()

            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break

            if question:
                self.ask(question)
                print("-" * 80 + "\n")
