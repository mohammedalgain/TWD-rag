import os
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

load_dotenv()

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "openai/gpt-oss-20b"
DB_PATH = "./chroma_db"
COLLECTION_NAME = "twd_characters_v2"


def load_resources():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to a .env file (see README).")

    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    if collection.count() == 0:
        raise RuntimeError("The vector database is empty. Run scrape.py, chunk.py and embed.py first.")

    model = SentenceTransformer(EMBED_MODEL)
    groq_client = Groq(api_key=api_key)
    return model, collection, groq_client


def ask(question, model, collection, groq_client, n_results=5):
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
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content, sorted(set(sources))
