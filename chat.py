from groq import GroqError

from rag import load_resources, ask

print("Loading model and database...")
try:
    model, collection, groq_client = load_resources()
except RuntimeError as e:
    print(f"Error: {e}")
    raise SystemExit(1)

print("\nWalking Dead Wiki Q&A — type 'quit' or 'exit' to stop.\n")

while True:
    question = input("You: ").strip()
    if question.lower() in ("quit", "exit"):
        print("Bye!")
        break
    if not question:
        continue

    try:
        answer, sources = ask(question, model, collection, groq_client)
    except GroqError as e:
        print(f"\nCould not get an answer from Groq: {e}\n")
        continue

    print(f"\nBot: {answer}")
    print(f"Sources: {', '.join(sources)}\n")
