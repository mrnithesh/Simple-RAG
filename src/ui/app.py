import streamlit as st

from src.utils.document_utils import (
    initialize_fixed_document,
    get_fixed_vectorstore,
    clear_fixed_vectorstore,
    get_fixed_vector_count,
)
from src.models.agent import get_simple_chat_agent
from src.config.constants import FIXED_PDF_PATH

def main():
    st.set_page_config(
        page_title="PDF Chat Assistant",
        layout="wide",
        initial_sidebar_state="collapsed",
        page_icon="📄"
    )
    
    st.title("📄 PDF Chat Assistant")
    st.markdown("*Chat with your document using a local LLM powered by Ollama*")

    # Initialize the system with the hardcoded document
    if "system_initialized" not in st.session_state:
        with st.spinner("🔄 Initializing system and loading document..."):
            success, message = initialize_fixed_document()
            
            if success:
                st.success(f"✅ {message}")
                st.session_state.system_initialized = True
            else:
                st.error(f"❌ {message}")
                st.markdown("""
                ### Setup Required:
                1. Place your PDF file at: `documents/document.pdf`
                2. Make sure the file exists and is readable
                3. Refresh the page to retry
                """)
                st.stop()

    # Sidebar for model settings
    with st.sidebar:
        st.header("🤖 Model Settings")
        model_name = "gemma3:1b"
        
        st.markdown("---")
        st.markdown("### 📄 Document Info")
        vector_count = get_fixed_vector_count()
        st.info(f"**File:** `{FIXED_PDF_PATH}`\n\n**Chunks indexed:** {vector_count}")
        
        if st.button("🔄 Reload Document", key="reload_doc"):
            # Clear the vectorstore and reinitialize
            cleared, _ = clear_fixed_vectorstore()
            st.session_state.system_initialized = False
            st.rerun()
        
        if st.button("🗑️ Clear Chat History", key="clear_chat"):
            st.session_state.chat_history = []
            st.success("Chat history cleared!")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # list[dict]: {role: "user"|"assistant", content: str}

    # Lazy agent init to reduce initial load time
    if "_agent" not in st.session_state:
        st.session_state._agent = None
        st.session_state._vectorstore = None

    # Display welcome message
    if not st.session_state.chat_history:
        st.markdown("""
        ### 👋 Welcome!
        I'm ready to answer questions about your document. You can ask me:
        - **Summarize** specific sections or the entire document
        - **Find information** about particular topics
        - **Explain concepts** mentioned in the document
        - **Extract key points** or important details
        
        Just type your question below to get started!
        """)

    # Display chat history
    for message in st.session_state.chat_history:
        role = message.get("role")
        content = message.get("content", "")
        if role in ("user", "assistant"):
            with st.chat_message("user" if role == "user" else "assistant"):
                st.write(content)

    # Chat input
    if query := st.chat_input("Ask a question about the document..."):
        # Add user message to chat
        with st.chat_message("user"):
            st.write(query)

        st.session_state.chat_history.append({"role": "user", "content": query})

        # Lazy-load vectorstore and agent on first query
        if st.session_state._agent is None:
            try:
                st.session_state._vectorstore = get_fixed_vectorstore()
                st.session_state._agent = get_simple_chat_agent(
                    st.session_state._vectorstore, model_name
                )
            except Exception as e:
                st.error(f"Error loading chat system: {e}")
                st.markdown("Make sure:")
                st.markdown("- Ollama is running")
                st.markdown(f"- Model '{model_name}' is available")
                st.markdown("- Document is properly loaded")
                st.stop()

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing document and generating response..."):
                try:
                    response = st.session_state._agent(query, st.session_state.chat_history)
                    st.write(response)

                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)

                    # Show helpful error message
                    if "ollama" in str(e).lower():
                        st.markdown("**Possible solutions:**")
                        st.markdown("- Make sure Ollama is running: `ollama serve`")
                        st.markdown("- Check if model is available: `ollama list`")
                        st.markdown(f"- Pull the model if needed: `ollama pull {model_name}`")

                    st.session_state.chat_history.append({"role": "assistant", "content": f"I encountered an error: {error_msg}"})

if __name__ == "__main__":
    main() 