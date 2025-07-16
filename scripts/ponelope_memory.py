import os
import uuid
import openai
from pinecone import Pinecone

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "YOUR_API_KEY_HERE")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "YOUR_ENVIRONMENT")
INDEX_NAME = os.getenv("INDEX_NAME", "ponelope-memory")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_KEY_HERE")
NAMESPACE = "ponelope"


def _openai_client():
    try:
        return openai.OpenAI(api_key=OPENAI_API_KEY)
    except AttributeError:
        openai.api_key = OPENAI_API_KEY
        return openai


pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index = pc.Index(INDEX_NAME)
client = _openai_client()


def embed(text: str) -> list:
    """Embed text using text-embedding-3-small."""
    response = client.embeddings.create(model="text-embedding-3-small", input=[text])
    return response.data[0].embedding


def add_memory(text: str, metadata: dict | None = None) -> str:
    """Store GPT output text in Pinecone."""
    vector = embed(text)
    vector_id = str(uuid.uuid4())
    index.upsert(vectors=[{"id": vector_id, "values": vector, "metadata": metadata or {}}], namespace=NAMESPACE)
    return vector_id


def query_memory(query: str, top_k: int = 3):
    """Query for the most relevant memories."""
    vector = embed(query)
    result = index.query(vector=vector, top_k=top_k, namespace=NAMESPACE, include_metadata=True)
    return result.matches


if __name__ == "__main__":
    msg = "Example GPT-generated text to remember."
    vid = add_memory(msg)
    print(f"Stored vector ID: {vid}")
    out = query_memory("Example GPT-generated query")
    for match in out:
        print(match)
