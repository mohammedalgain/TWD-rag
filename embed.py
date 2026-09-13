import json
import chromadb
from sentence_transformers import SentenceTransformer

# Load chunks
with open("twd_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} chunks")

# Load the embedding model (downloads once, then cached)
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# Set up ChromaDB (persistent, saves to disk in ./chroma_db folder)
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="twd_characters_v2")

# Process in batches so we can see progress
batch_size = 100
for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i+batch_size]
    
    texts = [c["text"] for c in batch]
    ids = [f"{c['source']}_{c['chunk_id']}" for c in batch]
    metadatas = [{"source": c["source"]} for c in batch]
    
    embeddings = model.encode(texts).tolist()
    
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )
    
    print(f"Processed {min(i+batch_size, len(chunks))}/{len(chunks)} chunks")

print("Done! Embeddings stored in ./chroma_db")