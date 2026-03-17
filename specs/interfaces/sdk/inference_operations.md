# Inference Operations

Methods for generating embeddings and reranking documents via the `Inference` class (available on `Pinecone` and `PineconeAsyncio` client instances as `.inference`).

This spec documents the `embed()` and `rerank()` methods — the two primary inference operations. Model discovery methods (`list_models()` and `get_model()`) and async variants are not covered in this spec and are reserved for future documentation.

---

## Inference.embed

Generates embeddings for text inputs using the specified embedding model.

**Source:** `pinecone/inference/inference.py:155-227`

**Added:** v1.0
**Deprecated:** No

### Signature

```python
def embed(
    self,
    model: EmbedModel | str,
    inputs: str | list[Dict] | list[str],
    parameters: dict[str, Any] | None = None,
) -> EmbeddingsList:
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|-----------|-------------|
| `model` | `EmbedModel \| str` | Yes | — | v1.0 | No | The embedding model to use. Can be a string like `"text-embedding-3-small"` or an `EmbedModel` enum value. |
| `inputs` | `str \| list[str] \| list[dict]` | Yes | — | v1.0 | No | Text inputs to embed. Can be a single string, a list of strings, or a list of dictionaries with custom fields. For dictionary inputs, the `text` field is typically used for embedding. |
| `parameters` | `dict[str, Any] \| None` | No | `None` | v1.0 | No | Model-specific parameters to control embedding behavior. Common parameters: `input_type` (e.g., "passage", "query"), `truncate` (e.g., "END" to truncate long inputs). Exact parameters depend on the model; invalid parameters are ignored. |

### Returns

**Type:** `EmbeddingsList`

A wrapper object containing:
- `data`: A list of embedding vectors, one per input. Each embedding is a dictionary with a `values` key containing the float vector.
- `model`: The name of the model used.
- `usage`: A dictionary with `total_tokens` — the total number of tokens used across all inputs.

### Raises

| Exception | Condition |
|-----------|-----------|
| `pinecone.PineconeApiException` | Invalid model name or API service error. |
| `Exception` | Inputs is an empty list or invalid format. |

### Idempotency

Non-idempotent. Repeated identical calls generate embeddings independently each time (no caching).

### Side Effects

None. Embeddings are generated and returned; no state is modified.

### Example

```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-api-key")

# Embed a single string
output = pc.inference.embed(
    model="text-embedding-3-small",
    inputs="Hello, world!"
)
print(output.data[0]['values'][:5])  # First 5 dimensions
print(output.usage['total_tokens'])  # Token usage

# Embed a list of strings
outputs = pc.inference.embed(
    model="multilingual-e5-large",
    inputs=["Document 1", "Document 2", "Document 3"],
    parameters={"input_type": "passage", "truncate": "END"}
)
print(f"Generated {len(outputs.data)} embeddings")

# Embed dictionaries with custom fields
outputs = pc.inference.embed(
    model="text-embedding-3-small",
    inputs=[
        {"text": "First document", "source": "web"},
        {"text": "Second document", "source": "pdf"}
    ]
)
```

### Notable Behavior

- **Single string input:** When `inputs` is a single string (not a list), the method returns `EmbeddingsList` with exactly one embedding in the `data` list.
- **Token counting:** The `usage` object reports total tokens across all inputs, not per-input. The token count depends on the model and input text length.
- **Batch processing:** All inputs are sent in a single request. There is no automatic batching for large input lists.
- **Model enum:** The `EmbedModel` enum provides constants for available models (e.g., `EmbedModel.Multilingual_E5_Large`). String model names are also accepted.

---

## Inference.rerank

Reranks documents by relevance to a query using the specified reranking model.

**Source:** `pinecone/inference/inference.py:229-345`

**Added:** v1.0
**Deprecated:** No

### Signature

```python
def rerank(
    self,
    model: RerankModel | str,
    query: str,
    documents: list[str] | list[dict[str, Any]],
    rank_fields: list[str] = ["text"],
    return_documents: bool = True,
    top_n: int | None = None,
    parameters: dict[str, Any] | None = None,
) -> RerankResult:
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|-----------|-------------|
| `model` | `RerankModel \| str` | Yes | — | v1.0 | No | The reranking model to use. Can be a string like `"bge-reranker-v2-m3"` or a `RerankModel` enum value. |
| `query` | `str` | Yes | — | v1.0 | No | The query string to rank documents against. |
| `documents` | `list[str] \| list[dict]` | Yes | — | v1.0 | No | Documents to rerank. Can be a list of strings or a list of dictionaries. For dictionaries, the fields specified in `rank_fields` are used for reranking. |
| `rank_fields` | `list[str]` | No | `["text"]` | v1.0 | No | The document fields to use for ranking. For string inputs, defaults to implicit field `"text"`. For dictionary inputs, specifies which fields contain the content to rank. |
| `return_documents` | `bool` | No | `True` | v1.0 | No | Whether to include the original documents in the response. When `False`, response contains only scores and indices. |
| `top_n` | `int \| None` | No | `None` | v1.0 | No | Maximum number of top-ranked documents to return. When `None`, returns all documents in ranked order. |
| `parameters` | `dict[str, Any] \| None` | No | `None` | v1.0 | No | Model-specific parameters to control reranking behavior. Common parameters: `truncate` (e.g., "END" to truncate long documents). Exact parameters depend on the model. |

### Returns

**Type:** `RerankResult`

A wrapper object containing:
- `data`: A list of ranked document objects. Each object has attributes:
  - `index`: The 0-based index of the document in the original input list.
  - `score`: A float in range [0, 1] indicating relevance to the query (higher is more relevant).
  - `document`: The original document (a string or object with a `text` attribute, omitted if `return_documents=False`).
- `model`: The name of the model used.
- `usage`: A dictionary with `rerank_units` — the number of reranking units consumed.

### Raises

| Exception | Condition |
|-----------|-----------|
| `pinecone.PineconeApiException` | Invalid model name or API service error. |
| `Exception` | `documents` is empty or invalid format. |

### Idempotency

Non-idempotent. Repeated identical calls rerank independently each time.

### Side Effects

None. Documents are reranked and scored; no state is modified.

### Example

```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-api-key")

# Rerank string documents
result = pc.inference.rerank(
    model="bge-reranker-v2-m3",
    query="Tell me about tech companies",
    documents=[
        "Apple is a popular fruit.",
        "Apple Inc. revolutionized computing.",
        "Orange is a citrus fruit.",
        "Microsoft is a software company."
    ],
    top_n=2
)

# Result contains the 2 most relevant documents
for ranked_doc in result.data:
    print(f"Index {ranked_doc.index}: score {ranked_doc.score:.3f}")
    print(f"  Text: {ranked_doc.document.text}")

# Rerank dictionaries with custom fields
result = pc.inference.rerank(
    model="pinecone-rerank-v0",
    query="What is machine learning?",
    documents=[
        {"text": "Machine learning is AI.", "category": "tech"},
        {"text": "Cooking recipes.", "category": "food"},
    ],
    rank_fields=["text"],
    top_n=1
)

print(f"Best match: {result.data[0].document.text}")
print(f"Relevance score: {result.data[0].score:.3f}")

# Get scores without returning original documents
result = pc.inference.rerank(
    model="bge-reranker-v2-m3",
    query="Your query",
    documents=["doc1", "doc2", "doc3"],
    return_documents=False
)

# result.data contains only index and score, not the document text
```

### Notable Behavior

- **Ranking order:** Results are always sorted by relevance score in descending order (highest score first).
- **Score range:** Scores are float values typically in the range [0, 1], where 1.0 represents perfect relevance and 0.0 represents no relevance. Exact scale depends on the model.
- **Index field:** The `index` attribute in each ranked result refers to the 0-based position in the original `documents` list. Use this to map results back to original documents.
- **top_n truncation:** When `top_n` is specified, only the top N documents are returned. When `None`, all input documents are returned ranked. Setting `top_n=0` is invalid and raises an error.
- **Batch processing:** All documents are sent in a single request. There is no automatic batching.
- **Model enum:** The `RerankModel` enum provides constants for available models (e.g., `RerankModel.Bge_Reranker_V2_M3`). String model names are also accepted.
- **Document objects:** The document objects in the result support both attribute access (`result.data[0].document.text`) and dictionary-style access for compatibility.

---

## Data Models

### EmbeddingsList

Response wrapper from `embed()` method.

| Field | Type | Nullable | Description |
|-------|------|----------|-------------|
| `data` | `array of object` | No | List of embeddings. Each object has a `values` key containing an array of floats (the embedding vector). |
| `model` | `string` | No | The name of the embedding model used. |
| `usage` | `object` | No | Token usage statistics. Contains `total_tokens`: the total number of tokens processed across all inputs. |

**Source:** `pinecone/inference/models.py`

### RerankResult

Response wrapper from `rerank()` method.

| Field | Type | Nullable | Description |
|-------|------|----------|-------------|
| `data` | `array of object` | No | List of ranked documents. Each object contains: `index` (integer, 0-based position in input), `score` (float, relevance score), and `document` (original document, omitted if `return_documents=False`). |
| `model` | `string` | No | The name of the reranking model used. |
| `usage` | `object` | No | Usage statistics. Contains `rerank_units`: the number of reranking units consumed. |

**Source:** `pinecone/inference/models.py`

### EmbedModel

Enumeration of available embedding models. Access as `EmbedModel.<value>` (e.g., `EmbedModel.Multilingual_E5_Large`).

| Value | String Value | Description |
|-------|--------------|-------------|
| `Multilingual_E5_Large` | `"multilingual-e5-large"` | Multilingual E5 large embedding model. |
| `Pinecone_Sparse_English_V0` | `"pinecone-sparse-english-v0"` | Pinecone's sparse English embedding model for BM25-style search. |

**Source:** `pinecone/inference/inference_request_builder.py:13-15`

### RerankModel

Enumeration of available reranking models. Access as `RerankModel.<value>` (e.g., `RerankModel.Bge_Reranker_V2_M3`).

| Value | String Value | Description |
|-------|--------------|-------------|
| `Bge_Reranker_V2_M3` | `"bge-reranker-v2-m3"` | BGE reranker v2 medium model. |
| `Cohere_Rerank_3_5` | `"cohere-rerank-3.5"` | Cohere's rerank 3.5 model. |
| `Pinecone_Rerank_V0` | `"pinecone-rerank-v0"` | Pinecone's reranking model. |

**Source:** `pinecone/inference/inference_request_builder.py:18-21`
