import json

# Load the scraped articles
with open("twd_articles.json", "r", encoding="utf-8") as f:
    articles = json.load(f)

def chunk_text(text, chunk_size=400, overlap=80):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

all_chunks = []

for title, text in articles.items():
    text = text.strip()
    if not text or text == "TBA":
        continue
    chunks = chunk_text(text, chunk_size=400, overlap=80)
    for i, chunk in enumerate(chunks):
        all_chunks.append({
            "source": title,
            "chunk_id": i,
            "text": chunk
        })

print(f"Created {len(all_chunks)} chunks from {len(articles)} articles")

with open("twd_chunks.json", "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=2)

print("Saved to twd_chunks.json")