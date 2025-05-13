import streamlit as st
from backend.utils import extract_text
from backend.vec import generate_answer

st.title("🧠 Chat with Your Document")

# Initialize session state variables
if "document_text" not in st.session_state:
    st.session_state["document_text"] = None

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []  # Store (user, ai) tuples

# File Upload Section
uploaded_file = st.file_uploader("Upload a document (PDF or TXT)", type=["pdf", "txt"])

if uploaded_file:
    st.session_state["document_text"] = extract_text(uploaded_file)
    st.success("📄 Document uploaded and processed!")

# Chat UI
st.subheader("💬 Chat")

# Display chat history
for user_msg, ai_msg in st.session_state["chat_history"]:
    with st.chat_message("user"):
        st.markdown(f"**User:** {user_msg}")

    with st.chat_message("assistant"):
        st.markdown(f"**AI:** {ai_msg}")

# User input
query = st.chat_input("Ask a question about the document...")

if query:
    with st.chat_message("user"):
        st.markdown(f"**User:** {query}")

    answer = generate_answer(query)

    with st.chat_message("assistant"):
        st.markdown(f"**AI:** {answer}")

    # Store chat history
    st.session_state["chat_history"].append((query, answer))
