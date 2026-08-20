import ollama


MODEL = "nomic-embed-text"


def create_embedding(text):
    response = ollama.embed(
        model=MODEL,
        input=text,
    )

    return response["embeddings"][0]
