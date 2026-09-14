#old version of ask.py, updated to app.py now its irrelevant


import os
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

load_dotenv()

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="twd_characters_v2")

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

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
    
    return response.choices[0].message.content, sources

question = "Who is the governor and how did he die?"
answer, sources = ask(question)
print("Answer:", answer)
print("\nSources:", set(sources))