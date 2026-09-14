import os
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
import streamlit as st

load_dotenv()

# Cache the heavy stuff so it only loads once, not on every interaction
@st.cache_resource
def load_resources():
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="twd_characters_v2")
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return model, collection, groq_client

model, collection, groq_client = load_resources()

def ask(question, n_results=5):
    query_embedding = model.encode([question]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=n_results)

    context_chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    context = "\n\n".join(context_chunks)

    prompt = f"""Answer the question based only on the context below. If the context doesn't contain the answer, say so.

Context:
{context}

Question: {question}

Answer:"""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content, sorted(set(sources))

# ---- Streamlit UI ----
st.set_page_config(page_title="TWD Wiki Q&A", page_icon="🧟")
st.title("🧟 Walking Dead Wiki Q&A")
st.caption("Ask about characters from the TV show. Answers are generated from wiki content using RAG.")

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
            answer, sources = ask(question)
        st.write(answer)
        with st.expander("Sources"):
            for s in sources:
                st.write(f"- {s}")

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})