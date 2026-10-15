"""Streamlit UI for Agentic RAG System - Simplified Version"""

import streamlit as st
from pathlib import Path
import sys
import time

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.config.config import Config
from src.rag_system import AgenticRAG

# Page configuration
st.set_page_config(
    page_title="🤖 RAG Search",
    page_icon="🔍",
    layout="centered"
)

# Simple CSS
st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables"""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'history' not in st.session_state:
        st.session_state.history = []

@st.cache_resource
def initialize_rag():
    """Initialize the RAG system (cached; failures are raised so they are not cached)"""
    # Loads the saved FAISS index, or builds it from data/url.txt + PDFs in data/ and data/uploads/
    return AgenticRAG()

def upload_sidebar(rag):
    """Sidebar: show the knowledge base and let users add their own PDFs"""
    with st.sidebar:
        st.header("📂 Knowledge base")
        st.caption(f"{rag.chunk_count()} chunks indexed")

        files = st.file_uploader(
            "Add PDF documents",
            type=["pdf"],
            accept_multiple_files=True
        )
        if st.button("➕ Add to knowledge base", disabled=not files):
            Config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            for file in files:
                dest = Config.UPLOAD_DIR / Path(file.name).name
                if dest.exists():
                    if dest.stat().st_size == file.size:
                        st.info(f"{dest.name} is already in the knowledge base")
                    else:
                        st.warning(f"A different file named {dest.name} already exists. Rename it and try again.")
                    continue

                dest.write_bytes(file.getbuffer())
                with st.spinner(f"Indexing {dest.name}..."):
                    try:
                        chunks = rag.add_pdf(dest)
                        st.success(f"Added {dest.name} ({chunks} chunks)")
                    except Exception as e:
                        dest.unlink(missing_ok=True)
                        st.error(f"Could not add {dest.name}: {e}")

        uploads = sorted(Config.UPLOAD_DIR.glob("*.pdf")) if Config.UPLOAD_DIR.is_dir() else []
        if uploads:
            st.markdown("**Uploaded documents**")
            for pdf in uploads:
                st.markdown(f"- {pdf.name}")

def show_sources(sources):
    """List the sources behind an answer: cited ones first, then the rest"""
    if not sources:
        return
    cited = [s for s in sources if s['cited']]
    other = [s for s in sources if not s['cited']]

    with st.expander(f"📚 Sources ({len(cited)} cited)"):
        for group, label in ((cited, "Cited in the answer"), (other, "Also retrieved")):
            if not group:
                continue
            st.markdown(f"**{label}**")
            for s in group:
                icon = "🌐" if s['type'] == "wikipedia" else "📄"
                location = s['location']
                if location.startswith("http"):
                    location = f"[{location}]({location})"
                st.markdown(f"**[{s['id']}]** {icon} {s['title']} — {location}")
                st.caption(s['content'][:300] + ("..." if len(s['content']) > 300 else ""))

def main():
    """Main application"""
    init_session_state()

    # Title
    st.title("🔍 RAG Document Search")
    st.markdown("Ask questions about the loaded documents")

    # Initialize system
    if not st.session_state.initialized:
        with st.spinner("Loading system..."):
            try:
                rag_system = initialize_rag()
            except Exception as e:
                st.error(f"Failed to initialize: {str(e)}")
                rag_system = None
            if rag_system:
                st.session_state.rag_system = rag_system
                st.session_state.initialized = True
                st.success(f"✅ System ready! ({rag_system.chunk_count()} document chunks loaded)")

    if st.session_state.rag_system:
        upload_sidebar(st.session_state.rag_system)

    st.markdown("---")

    # Search interface
    with st.form("search_form"):
        question = st.text_input(
            "Enter your question:",
            placeholder="What would you like to know?"
        )
        submit = st.form_submit_button("🔍 Search")

    # Process search
    if submit and question:
        if st.session_state.rag_system:
            with st.spinner("Searching..."):
                start_time = time.time()

                # Get answer
                result = st.session_state.rag_system.query(question)

                elapsed_time = time.time() - start_time

                # Add to history
                st.session_state.history.append({
                    'question': question,
                    'answer': result['answer'],
                    'time': elapsed_time
                })

                # Display answer
                st.markdown("### 💡 Answer")
                st.success(result['answer'])

                # Show the passages and Wikipedia pages the answer is based on
                show_sources(result.get('sources', []))

                st.caption(f"⏱️ Response time: {elapsed_time:.2f} seconds")

    # Show history
    if st.session_state.history:
        st.markdown("---")
        st.markdown("### 📜 Recent Searches")

        for item in reversed(st.session_state.history[-3:]):  # Show last 3
            with st.container():
                st.markdown(f"**Q:** {item['question']}")
                st.markdown(f"**A:** {item['answer'][:200]}...")
                st.caption(f"Time: {item['time']:.2f}s")
                st.markdown("")

if __name__ == "__main__":
    main()
