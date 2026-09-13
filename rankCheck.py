import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="twd_characters_v2")

query = "How did Rick Grimes lose his hand?"
query_embedding = model.encode([query]).tolist()

results = collection.query(
    query_embeddings=query_embedding,
    n_results=20
)

target_id = "Rick Grimes (TV Universe)_974"
ids = results["ids"][0]

if target_id in ids:
    rank = ids.index(target_id) + 1
    print(f"Found at rank {rank} out of 20")
else:
    print("Still not in top 20")