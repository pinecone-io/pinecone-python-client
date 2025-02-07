# Inference API

The Pinecone SDK now supports creating embeddings via the [Inference API](https://docs.pinecone.io/guides/inference/understanding-inference).


```python
from pinecone import Pinecone, EmbedModel

pc = Pinecone(api_key="YOUR_API_KEY")

# Embed documents
text = [
    "Turkey is a classic meat to eat at American Thanksgiving.",
    "Many people enjoy the beautiful mosques in Turkey.",
]
text_embeddings = pc.inference.embed(
    model=EmbedModel.Multilingual_E5_Large,
    inputs=text,
    parameters={
        "input_type": "passage",
        "truncate": "END"
    },
)

# Upsert documents into Pinecone index

# Embed a query
query = ["How should I prepare my turkey?"]
query_embeddings = pc.inference.embed(
    model=model,
    inputs=query,
    parameters={
        "input_type": "query",
        "truncate": "END"
    },
)

# Send query to Pinecone index to retrieve similar documents
```
