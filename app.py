from groq import GroqError
import streamlit as st

from rag import load_resources, ask

# Cache the heavy stuff so it only loads once, not on every interaction
@st.cache_resource
def get_resources():
    return load_resources()

# ---- Streamlit UI ----
st.set_page_config(page_title="TWD Wiki Q&A", page_icon="🧟")
st.title("🧟 Walking Dead Wiki Q&A")
st.caption("Ask about characters from the TV show. Answers are generated from wiki content using RAG.")

try:
    model, collection, groq_client = get_resources()
except RuntimeError as e:
    st.error(str(e))
    st.stop()

# Keep chat history across reruns
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.write(f"- {s}")

# Chat input
question = st.chat_input("Ask a question about The Walking Dead...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, sources = ask(question, model, collection, groq_client)
            except GroqError as e:
                st.error(f"Could not get an answer from Groq: {e}")
                st.stop()
        st.write(answer)
        with st.expander("Sources"):
            for s in sources:
                st.write(f"- {s}")

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})