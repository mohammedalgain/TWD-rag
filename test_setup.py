import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="twd_characters_v2")

query = "How did Rick Grimes lose his hand?"
query_embedding = model.encode([query]).tolist()

results = collection.query(
    query_embeddings=query_embedding,
    n_results=3
)

for i, doc in enumerate(results["documents"][0]):
    source = results["metadatas"][0][i]["source"]
    print(f"\n--- Result {i+1} (from: {source}) ---")
    print(doc[:300])