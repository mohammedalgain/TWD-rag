# Walking Dead Wiki Q&A (RAG)

A Retrieval-Augmented Generation app that answers questions about The Walking Dead TV show, built from scratch: scraping the Fandom wiki, chunking and embedding the content, storing it in a vector database, and generating answers with an LLM.

## Pipeline

1. **`scrape.py`**  pulls ~900 TV Universe character pages from the Walking Dead Fandom wiki via the MediaWiki API, cleans the HTML down to paragraph text, and saves it to `twd_articles.json`.
2. **`chunk.py`**  splits articles into overlapping ~400-character chunks (`twd_chunks.json`), sized to keep each chunk focused on a single idea rather than diluting multiple events together.
3. **`embed.py`**  embeds each chunk using `BAAI/bge-small-en-v1.5` (a retrieval-tuned model) and stores the vectors in a persistent ChromaDB collection.
4. **`chat.py`**  a terminal chat loop: takes a question, retrieves the most relevant chunks by semantic similarity, and passes them to Groq's `openai/gpt-oss-20b` model to generate an answer, citing which wiki pages it drew from.
5. **`app.py`**  a Streamlit chat UI version of the same pipeline, with persistent chat history and an expandable "Sources" section under each answer.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file with:
```
GROQ_API_KEY=your_key_here
```

Run the pipeline in order (only needed once, or after changing chunking/embedding parameters):
```bash
python scrape.py
python chunk.py
python embed.py
```

Then chat:
```bash
python chat.py
```

Or launch the Streamlit UI:
```bash
streamlit run app.py
```

## Example

> **You:** who is brian heriot
>
> **Bot:** Brian Heriot is a survivor of the outbreak featured in AMC's *The Walking Dead*. He is a relatively obscure character: nothing is known about his life before the outbreak, and his whereabouts after the outbreak begin are unknown. Later in the series, the Governor adopts "Brian Heriot" as a pseudonym.
>
> *Sources: Brian Heriot (TV Series)*

## Notes / Known Limitations

- Retrieval is semantic (embedding-based), not keyword-based — it can miss very specific facts if they're phrased very differently from the question, even when the fact exists in the dataset. Tested with `BAAI/bge-small-en-v1.5` after an initial general-purpose model (`all-MiniLM-L6-v2`) underperformed on precise factual queries.
- The app is instructed to say "the context doesn't contain this" rather than guess, when retrieval doesn't surface the right chunk — a deliberate choice to avoid hallucination over completeness.
