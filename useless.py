import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="twd_characters")

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
    print("Not in top 20 at all")

print("\nTop 3 results for comparison:")
for i in range(3):
    print(f"{i+1}. {results['metadatas'][0][i]['source']} (id: {results['ids'][0][i]})")