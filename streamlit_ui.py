import streamlit as st
from query_helper import QueryHelper
from llm_reasoner import LLMReasoner
import time

# Page config
st.set_page_config(
    page_title="MSIS Placement Assistant",
    page_icon="🎓",
    layout="wide"
)

# Initialize (cache to avoid reloading)
@st.cache_resource
def load_system():
    return QueryHelper(use_llm=True), LLMReasoner()

helper, reasoner = load_system()

# Title
st.title("🎓 MSIS Placement Assistant")
st.markdown("Ask questions about MSIS placement records (2025-2026)")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    top_k = st.slider("Number of documents to retrieve", 3, 15, 5)
    show_draft = st.checkbox("Show draft answer", value=False)
    show_context = st.checkbox("Show retrieved context", value=False)
    
    st.markdown("---")
    st.header("📊 Stats")
    if 'query_count' not in st.session_state:
        st.session_state.query_count = 0
    st.metric("Queries Asked", st.session_state.query_count)
    
    st.markdown("---")
    st.header("💡 Example Queries")
    examples = [
        "Which companies visited MSIS?",
        "What is Amazon eligibility?",
        "Companies with CTC above 15 LPA",
        "Tell me about Intel selection process"
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex}"):
            st.session_state.current_query = ex

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show metadata if available
        if "metadata" in message:
            with st.expander("📋 Details"):
                st.json(message["metadata"])

# Chat input
if prompt := st.chat_input("Ask about MSIS placements..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching placement records..."):
            # Get draft answer
            draft_result = helper.generate_answer(prompt, top_k=top_k)
            
            # Show draft if enabled
            if show_draft:
                with st.expander("📝 Draft Answer"):
                    st.markdown(draft_result['answer'])
            
            # Show context if enabled
            if show_context:
                with st.expander("📄 Retrieved Context"):
                    st.text(draft_result['context'][:1000] + "...")
        
        with st.spinner("✨ Refining answer..."):
            # Refine answer
            refined = reasoner.refine_answer(
                query=prompt,
                draft_answer=draft_result['answer'],
                context=draft_result['context'],
                sources=draft_result['sources']
            )
        
        # Display refined answer
        st.markdown(refined['refined_answer'])
        
        # Show metadata
        col1, col2, col3 = st.columns(3)
        with col1:
            st.caption(f"📊 Format: {refined['format_type']}")
        with col2:
            st.caption(f"📚 Sources: {', '.join(refined['sources'][:3])}")
        with col3:
            status = "✅" if refined['validation']['passed'] else "⚠️"
            st.caption(f"{status} Validation")
        
        # Add to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": refined['refined_answer'],
            "metadata": {
                "format": refined['format_type'],
                "sources": refined['sources'],
                "validation": refined['validation']
            }
        })
        
        st.session_state.query_count += 1

# Clear chat button
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.session_state.query_count = 0
    st.rerun()