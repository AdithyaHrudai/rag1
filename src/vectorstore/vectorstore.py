"""Vector store module for document embedding and retrieval"""

from pathlib import Path
from typing import List, Union
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.documents import Document

class VectorStore:
    """Manages vector store operations"""

    def __init__(self, embedding_model: str = "BAAI/bge-small-en-v1.5"):
        """
        Initialize vector store with local (FastEmbed) embeddings

        Args:
            embedding_model: FastEmbed model name (downloaded once, runs locally)
        """
        self.embedding = FastEmbedEmbeddings(model_name=embedding_model)
        self.vectorstore = None
        self.retriever = None

    def create_vectorstore(self, documents: List[Document]):
        """
        Create vector store from documents

        Args:
            documents: List of documents to embed
        """
        self.vectorstore = FAISS.from_documents(documents, self.embedding)
        self.retriever = self.vectorstore.as_retriever()

    def add_documents(self, documents: List[Document]):
        """
        Embed and add documents to the existing vector store.
        The retriever sees them immediately because it wraps the same index.

        Args:
            documents: List of documents to embed
        """
        if self.vectorstore is None:
            self.create_vectorstore(documents)
        else:
            self.vectorstore.add_documents(documents)

    def save(self, directory: Union[str, Path]):
        """
        Save the FAISS index and its documents to disk

        Args:
            directory: Folder to write the index files into
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        self.vectorstore.save_local(str(directory))

    def load(self, directory: Union[str, Path]):
        """
        Load a FAISS index previously written by save()

        Args:
            directory: Folder containing the index files
        """
        # The index is a pickle written by this app, so deserializing it is safe.
        # Never point this at index files from an untrusted source.
        self.vectorstore = FAISS.load_local(
            str(directory), self.embedding, allow_dangerous_deserialization=True
        )
        self.retriever = self.vectorstore.as_retriever()

    def size(self) -> int:
        """
        Number of chunks in the index

        Returns:
            Chunk count (0 if not initialized)
        """
        return self.vectorstore.index.ntotal if self.vectorstore else 0

    def get_retriever(self):
        """
        Get the retriever instance

        Returns:
            Retriever instance
        """
        if self.retriever is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        return self.retriever

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """
        Retrieve relevant documents for a query

        Args:
            query: Search query
            k: Number of documents to retrieve

        Returns:
            List of relevant documents
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        return self.vectorstore.similarity_search(query, k=k)
