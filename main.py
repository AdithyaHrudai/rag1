"""Main application entry point for Agentic RAG system"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

# Let the emoji in the output print on Windows consoles that default to cp1252
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

from src.rag_system import AgenticRAG

def main():
    """Main function"""
    # Initialize RAG system (sources come from data/url.txt and PDFs in data/ and data/uploads/)
    rag = AgenticRAG()
    
    # Example questions
    example_questions = [
        "What is the concept of agent loop in autonomous agents?",
        "What are the key components of LLM-powered agents?",
        "Explain the concept of diffusion models for video generation."
    ]
    
    print("=" * 80)
    print("📝 Running example questions:")
    print("=" * 80 + "\n")
    
    for question in example_questions:
        rag.ask(question)
        print("=" * 80 + "\n")
    
    # Optional: Run interactive mode
    print("\n" + "=" * 80)
    user_input = input("Would you like to enter interactive mode? (y/n): ")
    if user_input.lower() == 'y':
        rag.interactive_mode()

if __name__ == "__main__":
    main()