#old chat.py, updated to app.py now its irrelevant

import os
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

load_dotenv()

print("Loading model and database...")
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

    return response.choices[0].message.content, sorted(set(sources))

print("\nWalking Dead Wiki Q&A — type 'quit' or 'exit' to stop.\n")

while True:
    question = input("You: ").strip()
    if question.lower() in ("quit", "exit"):
        print("Bye!")
        break
    if not question:
        continue

    answer, sources = ask(question)
    print(f"\nBot: {answer}")
    print(f"Sources: {', '.join(sources)}\n")