# Inference Operations

This module documents inference operations on the Pinecone and PineconeAsyncio clients: generating embeddings and reranking documents. The Inference API uses pre-trained models to generate vector embeddings from text and to rank documents by relevance to a query.

---

## `Inference.embed()`

Generates embeddings for the provided inputs using the specified embedding model.

**Source:** `pinecone/inference/inference.py:155-227`, `pinecone/inference/inference_asyncio.py:55-141` (async equivalent)
**Added:** v3.0.0
**Deprecated:** No
**Idempotency:** Safe to retry
**Side effects:** None

### Signature

```python
def embed(
    self,
    model: str | EmbedModel,
    inputs: str | list[str] | list[dict],
    parameters: dict[str, Any] | None = None
) -> EmbeddingsList
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `model` | `str \| EmbedModel` | Yes | — | v3.0.0 | No | The model to use for generating embeddings. Can be a string (e.g., `"text-embedding-3-small"`) or an `EmbedModel` enum value. |
| `inputs` | `str \| list[str] \| list[dict]` | Yes | — | v3.0.0 | No | The input(s) to generate embeddings for. Can be a single string, a list of strings, or a list of dictionaries with text fields. For dictionary inputs, the `text` field is typically used for embedding. |
| `parameters` | `dict[str, Any]` | No | `None` | v3.0.0 | No | Optional parameters for the embedding model. Varies by model; common parameters include `input_type` (e.g., "passage", "query") and `truncate` (e.g., "END", "NONE"). Exact parameters depend on the model; invalid parameters are ignored. |

### Returns

**Type:** `EmbeddingsList` — An iterable collection of embeddings with the following structure:
- `data` — List of embedding dictionaries, each with a `values` key containing the vector as a list of floats
- `model` — The model name used to generate the embeddings
- `usage` — Dictionary containing `total_tokens` (total tokens used)

### Raises

| Exception | Condition |
|-----------|-----------|
| `PineconeApiException` | The request failed. May occur due to invalid model name, malformed inputs, or service errors. |
| `Exception` | `inputs` is empty or an invalid type. |
| `UnauthorizedException` | The API key is invalid or missing. |

### Behavior

- **Single string input:** When `inputs` is a single string (not a list), the method returns `EmbeddingsList` with exactly one embedding in the `data` list.
- **Token counting:** The `usage` object reports total tokens across all inputs, not per-input. The token count depends on the model and input text length.
- **Batch processing:** All inputs are sent in a single request. There is no automatic batching for large input lists.
- **Model enum:** The `EmbedModel` enum provides constants for available models (e.g., `EmbedModel.Multilingual_E5_Large`). String model names are also accepted.

### Example

```python
from pinecone import Pinecone

pc = Pinecone(api_key="sk-example-key-do-not-use")

# Single string input
output = pc.inference.embed(
    model="text-embedding-3-small",
    inputs="Hello, world!"
)
print(f"Embedding: {output.data[0]['values']}")

# Multiple inputs with parameters
outputs = pc.inference.embed(
    model="multilingual-e5-large",
    inputs=["Who created the first computer?", "What is artificial intelligence?"],
    parameters={"input_type": "passage", "truncate": "END"}
)
print(f"Generated {len(outputs)} embeddings")
for i, embedding in enumerate(outputs):
    print(f"Tokens used: {outputs.usage['total_tokens']}")

# Embed dictionaries with custom fields
outputs = pc.inference.embed(
    model="text-embedding-3-small",
    inputs=[
        {"text": "First document", "source": "web"},
        {"text": "Second document", "source": "pdf"}
    ]
)
```

### Notes

- The `EmbeddingsList` object is iterable and supports indexing: `outputs[0]`, `len(outputs)`
- Inputs are processed in parallel, so order is preserved in the response
- Token usage is reported for the entire batch in the `usage` field
- Embedding precision (float16 or float32) depends on the model; float32 is typical

---

## `AsyncioInference.embed()`

Asynchronous version of `embed()`. Generates embeddings for the provided inputs.

**Source:** `pinecone/inference/inference_asyncio.py:55-141`
**Added:** v3.0.0
**Deprecated:** No
**Idempotency:** Safe to retry
**Side effects:** None

### Signature

```python
async def embed(
    self,
    model: str,
    inputs: str | list[str] | list[dict],
    parameters: dict[str, Any] | None = None
) -> EmbeddingsList
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `model` | `str` | Yes | — | v3.0.0 | No | The model to use for generating embeddings (e.g., `"text-embedding-3-small"`). |
| `inputs` | `str \| list[str] \| list[dict]` | Yes | — | v3.0.0 | No | The input(s) to generate embeddings for. |
| `parameters` | `dict[str, Any]` | No | `None` | v3.0.0 | No | Optional model parameters. |

### Returns

**Type:** `EmbeddingsList` — An `EmbeddingsList` object (when awaited).

### Raises

| Exception | Condition |
|-----------|-----------|
| `PineconeApiException` | The request failed. |
| `PineconeApiValueError` | `inputs` is empty or an invalid type. |
| `UnauthorizedException` | The API key is invalid or missing. |

### Example

```python
import asyncio
from pinecone import PineconeAsyncio

async def embed_texts():
    async with PineconeAsyncio(api_key="sk-example-key-do-not-use") as pc:
        # Generate embeddings asynchronously
        outputs = await pc.inference.embed(
            model="text-embedding-3-small",
            inputs=["Document 1", "Document 2", "Document 3"]
        )

        print(f"Generated {len(outputs)} embeddings")
        for i, embedding in enumerate(outputs):
            print(f"Embedding {i}: {len(embedding['values'])} dimensions")

asyncio.run(embed_texts())
```

---

## `Inference.rerank()`

Reranks documents by relevance to a query using the specified reranking model.

**Source:** `pinecone/inference/inference.py:229-345`, `pinecone/inference/inference_asyncio.py:182-309` (async equivalent)
**Added:** v3.0.0
**Deprecated:** No
**Idempotency:** Safe to retry
**Side effects:** None

### Signature

```python
def rerank(
    self,
    model: str | RerankModel,
    query: str,
    documents: list[str] | list[dict[str, Any]],
    rank_fields: list[str] = ["text"],
    return_documents: bool = True,
    top_n: int | None = None,
    parameters: dict[str, Any] | None = None
) -> RerankResult
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `model` | `str \| RerankModel` | Yes | — | v3.0.0 | No | The reranking model to use. Can be a string (e.g., `"bge-reranker-v2-m3"`) or a `RerankModel` enum value. |
| `query` | `str` | Yes | — | v3.0.0 | No | The query to compare with documents. |
| `documents` | `list[str] \| list[dict[str, Any]]` | Yes | — | v3.0.0 | No | A list of documents to rank. Can be a list of strings or a list of dictionaries with document text and metadata. |
| `rank_fields` | `list[str]` | No | `["text"]` | v3.0.0 | No | Which fields in document dictionaries to use for ranking. For string inputs, defaults to implicit field `"text"`. For dictionary inputs, specifies which fields contain the content to rank. |
| `return_documents` | `bool` | No | `True` | v3.0.0 | No | Whether to include the documents in the response. If False, only scores and indices are returned. |
| `top_n` | `int` | No | `None` | v3.0.0 | No | The number of top-ranked documents to return. If None, returns all documents ranked. Setting `top_n=0` is invalid and raises an error. |
| `parameters` | `dict[str, Any]` | No | `None` | v3.0.0 | No | Optional parameters for the reranking model. Varies by model; common parameters include `truncate` (e.g., "END", "NONE"). |

### Returns

**Type:** `RerankResult` — A result object with the following structure:
- `data` — List of ranked documents, sorted by relevance (highest to lowest). Each item contains:
  - `index` — The position (0-based) of the document in the original input list
  - `score` — Relevance score (typically between 0 and 1, higher = more relevant)
  - `document` — The original document (if `return_documents=True`)
- `model` — The model name used for reranking
- `usage` — Dictionary containing `rerank_units` (units consumed for reranking)

### Raises

| Exception | Condition |
|-----------|-----------|
| `PineconeApiException` | The request failed. May occur due to invalid model name, malformed documents, or service errors. |
| `Exception` | `query` is empty, `documents` is empty, or an invalid type is provided. |
| `UnauthorizedException` | The API key is invalid or missing. |

### Behavior

- **Ranking order:** Results are always sorted by relevance score in descending order (highest score first).
- **Score range:** Scores are float values typically in the range [0, 1], where 1.0 represents perfect relevance and 0.0 represents no relevance. Exact scale depends on the model.
- **Index field:** The `index` attribute in each ranked result refers to the 0-based position in the original `documents` list. Use this to map results back to original documents.
- **top_n truncation:** When `top_n` is specified, only the top N documents are returned. When `None`, all input documents are returned ranked.
- **Batch processing:** All documents are sent in a single request. There is no automatic batching.
- **Model enum:** The `RerankModel` enum provides constants for available models (e.g., `RerankModel.Bge_Reranker_V2_M3`). String model names are also accepted.
- **Document objects:** The document objects in the result support both attribute access (`result.data[0].document.text`) and dictionary-style access for compatibility.

### Example

```python
from pinecone import Pinecone

pc = Pinecone(api_key="sk-example-key-do-not-use")

# Rerank documents with string inputs
result = pc.inference.rerank(
    model="bge-reranker-v2-m3",
    query="Tell me about tech companies",
    documents=[
        "Apple is a popular fruit known for its sweetness and crisp texture.",
        "Software is still eating the world.",
        "Many people enjoy eating apples as a healthy snack.",
        "Acme Inc. has revolutionized the tech industry with its sleek designs and user-friendly interfaces.",
        "An apple a day keeps the doctor away, as the saying goes.",
    ],
    top_n=2,
    return_documents=True
)

print(f"Top 2 results for query: '{result.data[0]['document']}'")
for ranked_doc in result.data:
    print(f"Document {ranked_doc['index']}: score={ranked_doc['score']:.4f}")

# Rerank documents with dictionaries and custom fields
result = pc.inference.rerank(
    model="pinecone-rerank-v0",
    query="What is machine learning?",
    documents=[
        {"text": "Machine learning is a subset of AI.", "category": "tech"},
        {"text": "Cooking recipes for pasta.", "category": "food"},
    ],
    rank_fields=["text"],
    top_n=1
)

# Get scores without returning original documents
result = pc.inference.rerank(
    model="bge-reranker-v2-m3",
    query="Your query",
    documents=["doc1", "doc2", "doc3"],
    return_documents=False
)
# result.data contains only index and score, not the document text
```

### Notes

- Documents are ranked by relevance score in descending order (highest relevance first)
- The `index` field in results maps back to the original input position, allowing you to correlate ranked results with source documents
- Reranking is compute-intensive; batch multiple queries if possible to optimize cost
- The `top_n` parameter allows you to return only the top K results without computing scores for all documents (some models optimize for this)
- Setting `return_documents=False` reduces response payload size when you only need scores and indices

---

## `AsyncioInference.rerank()`

Asynchronous version of `rerank()`. Reranks documents by relevance to a query.

**Source:** `pinecone/inference/inference_asyncio.py:182-309`
**Added:** v3.0.0
**Deprecated:** No
**Idempotency:** Safe to retry
**Side effects:** None

### Signature

```python
async def rerank(
    self,
    model: str,
    query: str,
    documents: list[str] | list[dict[str, Any]],
    rank_fields: list[str] = ["text"],
    return_documents: bool = True,
    top_n: int | None = None,
    parameters: dict[str, Any] | None = None
) -> RerankResult
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `model` | `str` | Yes | — | v3.0.0 | No | The reranking model to use (e.g., `"bge-reranker-v2-m3"`). |
| `query` | `str` | Yes | — | v3.0.0 | No | The query to compare with documents. |
| `documents` | `list[str] \| list[dict[str, Any]]` | Yes | — | v3.0.0 | No | A list of documents to rank. |
| `rank_fields` | `list[str]` | No | `["text"]` | v3.0.0 | No | Which fields to use for ranking. |
| `return_documents` | `bool` | No | `True` | v3.0.0 | No | Whether to include the documents in the response. |
| `top_n` | `int` | No | `None` | v3.0.0 | No | The number of top-ranked documents to return. |
| `parameters` | `dict[str, Any]` | No | `None` | v3.0.0 | No | Optional model parameters. |

### Returns

**Type:** `RerankResult` — A `RerankResult` object (when awaited).

### Raises

| Exception | Condition |
|-----------|-----------|
| `PineconeApiException` | The request failed. |
| `PineconeApiValueError` | `query` is empty, `documents` is empty, or parameters are invalid. |
| `UnauthorizedException` | The API key is invalid or missing. |

### Example

```python
import asyncio
from pinecone import PineconeAsyncio

async def rerank_documents():
    async with PineconeAsyncio(api_key="sk-example-key-do-not-use") as pc:
        result = await pc.inference.rerank(
            model="bge-reranker-v2-m3",
            query="What are the best practices for machine learning?",
            documents=[
                "Machine learning requires large datasets.",
                "Practice makes perfect.",
                "Data preprocessing is crucial for model performance.",
                "The weather today is sunny.",
            ],
            top_n=2
        )

        for ranked_doc in result.data:
            print(f"Document {ranked_doc['index']}: {ranked_doc['score']:.4f}")

asyncio.run(rerank_documents())
```

---

## `Inference.list_models()`

Lists all available inference models, with optional filtering by model type and vector type.

**Source:** `pinecone/inference/inference.py:348-384`
**Added:** v3.0.0
**Deprecated:** No
**Idempotency:** Safe to retry
**Side effects:** None

### Signature

```python
def list_models(
    self,
    *,
    type: str | None = None,
    vector_type: str | None = None
) -> ModelInfoList
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `type` | `str` | No | `None` | v3.0.0 | No | Filter by model type. Accepted values: `"embed"`, `"rerank"`. When `None`, returns all model types. |
| `vector_type` | `str` | No | `None` | v3.0.0 | No | Filter by vector type. Accepted values: `"dense"`, `"sparse"`. When `None`, returns all vector types. |

### Returns

**Type:** `ModelInfoList` -- An iterable collection of `ModelInfo` objects. Supports `len()`, iteration, and integer indexing. Has a `.names()` method that returns a list of model name strings.

### Raises

| Exception | Condition |
|-----------|-----------|
| `PineconeApiException` | The request failed due to a server error. |
| `UnauthorizedException` | The API key is invalid or missing. |

### Example

```python
from pinecone import Pinecone

pc = Pinecone(api_key="sk-example-key-do-not-use")

# List all models
models = pc.inference.list_models()
print(f"Available models: {models.names()}")

# Filter by model type
embed_models = pc.inference.list_models(type="embed")
rerank_models = pc.inference.list_models(type="rerank")

# Filter by vector type
sparse_models = pc.inference.list_models(vector_type="sparse")

# Combine filters
dense_embed_models = pc.inference.list_models(type="embed", vector_type="dense")

# Iterate over models
for model in models:
    print(f"{model.model}: {model.short_description}")
```

### Notes

- All parameters are keyword-only (enforced by the `*` separator in the signature)
- Filtering is performed server-side; only matching models are returned
- The returned `ModelInfoList` supports integer indexing (`models[0]`) and iteration

---

## `Inference.get_model()`

Gets detailed information about a specific model by name.

**Source:** `pinecone/inference/inference.py:386-429`
**Added:** v3.0.0
**Deprecated:** No
**Idempotency:** Safe to retry
**Side effects:** None

### Signature

```python
def get_model(self, model_name: str) -> ModelInfo
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `model_name` | `str` | Yes | -- | v3.0.0 | No | The name of the model to retrieve (e.g., `"pinecone-rerank-v0"`, `"multilingual-e5-large"`). Must be passed as a keyword argument. |

### Returns

**Type:** `ModelInfo` -- An object containing model details including name, description, type, supported parameters, dimensions, sequence length limits, and supported metrics. See the `ModelInfo` data model below.

### Raises

| Exception | Condition |
|-----------|-----------|
| `PineconeApiException` | The request failed or the model name was not found. |
| `UnauthorizedException` | The API key is invalid or missing. |
| `TypeError` | `model_name` was passed as a positional argument (keyword-only enforcement via `@require_kwargs`). |

### Example

```python
from pinecone import Pinecone

pc = Pinecone(api_key="sk-example-key-do-not-use")

model_info = pc.inference.get_model(model_name="pinecone-rerank-v0")
print(f"Model: {model_info.model}")
print(f"Type: {model_info.type}")
print(f"Description: {model_info.short_description}")
print(f"Max batch size: {model_info.max_batch_size}")
print(f"Max sequence length: {model_info.max_sequence_length}")
print(f"Supported metrics: {model_info.supported_metrics}")
```

### Notes

- The `model_name` parameter must be passed as a keyword argument; positional arguments raise `TypeError` due to the `@require_kwargs` decorator
- The returned `ModelInfo` object supports both attribute access (`model_info.model`) and dictionary-style access (`model_info["model"]`)
- Use `model_info.to_dict()` to convert the result to a plain dictionary

---

## `AsyncioInference.list_models()`

Asynchronous version of `list_models()`. Lists all available inference models with optional filtering.

**Source:** `pinecone/inference/inference_asyncio.py:312-353`
**Added:** v3.0.0
**Deprecated:** No
**Idempotency:** Safe to retry
**Side effects:** None

### Signature

```python
async def list_models(
    self,
    *,
    type: str | None = None,
    vector_type: str | None = None
) -> ModelInfoList
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `type` | `str` | No | `None` | v3.0.0 | No | Filter by model type. Accepted values: `"embed"`, `"rerank"`. |
| `vector_type` | `str` | No | `None` | v3.0.0 | No | Filter by vector type. Accepted values: `"dense"`, `"sparse"`. |

### Returns

**Type:** `ModelInfoList` -- A `ModelInfoList` object (when awaited).

### Raises

| Exception | Condition |
|-----------|-----------|
| `PineconeApiException` | The request failed. |
| `UnauthorizedException` | The API key is invalid or missing. |

### Example

```python
import asyncio
from pinecone import PineconeAsyncio

async def list_available_models():
    async with PineconeAsyncio(api_key="sk-example-key-do-not-use") as pc:
        # List all models
        models = await pc.inference.list_models()
        print(f"Available models: {models.names()}")

        # Filter to embedding models only
        embed_models = await pc.inference.list_models(type="embed")
        for model in embed_models:
            print(f"{model.model}: {model.type}")

asyncio.run(list_available_models())
```

---

## `AsyncioInference.get_model()`

Asynchronous version of `get_model()`. Gets detailed information about a specific model.

**Source:** `pinecone/inference/inference_asyncio.py:355-385`
**Added:** v3.0.0
**Deprecated:** No
**Idempotency:** Safe to retry
**Side effects:** None

### Signature

```python
async def get_model(self, model_name: str) -> ModelInfo
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `model_name` | `str` | Yes | -- | v3.0.0 | No | The name of the model to retrieve. Must be passed as a keyword argument. |

### Returns

**Type:** `ModelInfo` -- A `ModelInfo` object (when awaited).

### Raises

| Exception | Condition |
|-----------|-----------|
| `PineconeApiException` | The request failed or the model name was not found. |
| `UnauthorizedException` | The API key is invalid or missing. |
| `TypeError` | `model_name` was passed as a positional argument. |

### Example

```python
import asyncio
from pinecone import PineconeAsyncio

async def get_model_details():
    async with PineconeAsyncio(api_key="sk-example-key-do-not-use") as pc:
        model_info = await pc.inference.get_model(model_name="multilingual-e5-large")
        print(f"Model: {model_info.model}")
        print(f"Type: {model_info.type}")
        print(f"Supported metrics: {model_info.supported_metrics}")

asyncio.run(get_model_details())
```

---

## Error Handling

All inference operations may raise the following exceptions:

| Exception | Cause | Handling |
|-----------|-------|----------|
| `PineconeApiException` | Unexpected server error | Implement retry logic with exponential backoff |
| `PineconeApiValueError` | Invalid parameter values or empty inputs | Validate inputs before retrying; check parameter format |
| `UnauthorizedException` | Invalid or missing API key | Verify `PINECONE_API_KEY` environment variable or constructor argument |
| `PineconeApiTypeError` | Type mismatch in parameters or response | Ensure correct parameter types (str vs list, etc.) |

---

## Usage Patterns

### Complete Embedding Workflow

```python
from pinecone import Pinecone

pc = Pinecone(api_key="sk-example-key-do-not-use")

# Generate embeddings for a corpus
documents = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a subset of artificial intelligence.",
    "Python is a popular programming language for data science."
]

embeddings = pc.inference.embed(
    model="text-embedding-3-small",
    inputs=documents
)

print(f"Generated {len(embeddings)} embeddings")
for i, embedding in enumerate(embeddings):
    vector = embedding['values']
    print(f"Document {i}: {len(vector)} dimensions")
```

### Complete Reranking Workflow

```python
from pinecone import Pinecone

pc = Pinecone(api_key="sk-example-key-do-not-use")

query = "What is machine learning?"
documents = [
    "Machine learning is a field of AI.",
    "Python is a programming language.",
    "Deep learning is a subset of machine learning.",
    "The weather today is sunny."
]

result = pc.inference.rerank(
    model="bge-reranker-v2-m3",
    query=query,
    documents=documents,
    top_n=2
)

print(f"Query: {query}")
print(f"Top {len(result.data)} results:")
for ranked in result.data:
    print(f"  Document {ranked['index']}: score={ranked['score']:.4f}")
```

### Error Handling

```python
from pinecone import Pinecone
from pinecone.exceptions import PineconeApiValueError, UnauthorizedException

pc = Pinecone(api_key="sk-example-key-do-not-use")

try:
    embeddings = pc.inference.embed(
        model="text-embedding-3-small",
        inputs=[]  # Empty inputs will raise an error
    )
except PineconeApiValueError as e:
    print(f"Invalid input: {e}")
except UnauthorizedException:
    print("API key is invalid or missing")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Async Workflow

```python
import asyncio
from pinecone import PineconeAsyncio

async def process_documents():
    async with PineconeAsyncio(api_key="sk-example-key-do-not-use") as pc:
        # Embed documents
        embeddings = await pc.inference.embed(
            model="text-embedding-3-small",
            inputs=["Document A", "Document B", "Document C"]
        )

        # Rerank with query
        query = "relevant documents"
        docs = [f"doc{i}" for i in range(5)]

        results = await pc.inference.rerank(
            model="bge-reranker-v2-m3",
            query=query,
            documents=docs
        )

        return embeddings, results

embeddings, results = asyncio.run(process_documents())
```

---

## Data Models

### Enumerations

#### `EmbedModel`

An enumeration of available embedding models for use with the `embed()` method.

**Source:** `pinecone/inference/inference_request_builder.py:13-15`

| Member | Value | Description |
|--------|-------|-------------|
| `Multilingual_E5_Large` | `"multilingual-e5-large"` | A multilingual embedding model that supports multiple languages |
| `Pinecone_Sparse_English_V0` | `"pinecone-sparse-english-v0"` | A sparse embedding model optimized for English text |

**Example**

```python
from pinecone import Pinecone
from pinecone import EmbedModel

pc = Pinecone(api_key="sk-example-key-do-not-use")

# Use enum for embedding with text-embedding-3-small model
embeddings = pc.inference.embed(
    model=EmbedModel.Multilingual_E5_Large,
    inputs=["Hello world", "Bonjour le monde"]
)
```

#### `RerankModel`

An enumeration of available reranking models for use with the `rerank()` method.

**Source:** `pinecone/inference/inference_request_builder.py:18-21`

| Member | Value | Description |
|--------|-------|-------------|
| `Bge_Reranker_V2_M3` | `"bge-reranker-v2-m3"` | BGE reranker v2 model optimized for large-scale ranking |
| `Cohere_Rerank_3_5` | `"cohere-rerank-3.5"` | Cohere's reranking model v3.5 |
| `Pinecone_Rerank_V0` | `"pinecone-rerank-v0"` | Pinecone's proprietary reranking model |

**Example**

```python
from pinecone import Pinecone
from pinecone import RerankModel

pc = Pinecone(api_key="sk-example-key-do-not-use")

# Use enum for reranking
result = pc.inference.rerank(
    model=RerankModel.Bge_Reranker_V2_M3,
    query="best machine learning practices",
    documents=["doc1", "doc2", "doc3"]
)
```

---

### `EmbeddingsList`

An iterable collection of embeddings returned by the `embed()` method. Wraps the API response and delegates attribute access to the underlying OpenAPI model.

**Source:** `pinecone/inference/models/embedding_list.py:4-33`

**Fields**

| Field | Type | Nullable | Since | Deprecated | Description |
|-------|------|----------|-------|------------|-------------|
| `model` | `str` | No | v3.0.0 | No | The name of the model used to generate the embeddings. |
| `vector_type` | `str` | No | v3.0.0 | No | The type of vectors generated (e.g., `"dense"` for dense embeddings, `"sparse"` for sparse). |
| `data` | `list[dict]` | No | v3.0.0 | No | List of embedding objects. Each contains a `values` key with the embedding vector as a list of floats. |
| `usage` | `dict` | No | v3.0.0 | No | Usage statistics. Contains `total_tokens` (int) indicating total tokens consumed across all inputs. |

**Methods**

| Method | Return Type | Description |
|--------|-------------|-------------|
| `__len__()` | `int` | Returns the number of embeddings in `data`. |
| `__iter__()` | `Iterator[dict]` | Returns an iterator over the embedding dictionaries in `data`. |
| `__getitem__(index)` | `dict` | Accesses an embedding by integer index into `data`. |
| `__getattr__(attr)` | `Any` | Delegates attribute access to the underlying OpenAPI response object (e.g., `model`, `vector_type`, `usage`). |

**Example**

```python
from pinecone import Pinecone

pc = Pinecone(api_key="sk-example-key-do-not-use")

outputs = pc.inference.embed(
    model="text-embedding-3-small",
    inputs=["Hello", "World"]
)

# Access embeddings by index
first_embedding = outputs[0]
print(f"First embedding: {first_embedding['values'][:5]}...")  # First 5 dimensions

# Iterate over embeddings
for i, embedding in enumerate(outputs):
    print(f"Embedding {i} has {len(embedding['values'])} dimensions")

# Get metadata
print(f"Model: {outputs.model}")
print(f"Total tokens: {outputs.usage['total_tokens']}")
```

---

### `RerankResult`

A result object containing reranked documents returned by the `rerank()` method.

**Source:** `pinecone/inference/models/rerank_result.py:4-19`

**Structure**

The `RerankResult` object wraps the following structure:
```python
{
    "model": "bge-reranker-v2-m3",
    "data": [
        {
            "index": 3,
            "score": 0.9234,
            "document": {"text": "..."}
        },
        {
            "index": 1,
            "score": 0.8123,
            "document": {"text": "..."}
        }
    ],
    "usage": {"rerank_units": 1}
}
```

| Field | Type | Nullable | Since | Deprecated | Description |
|-------|------|----------|-------|------------|-------------|
| `model` | `str` | No | v3.0.0 | No | The name of the reranking model used. |
| `data` | `list[dict]` | No | v3.0.0 | No | List of ranked documents, ordered by relevance (highest first). Each item contains `index` (original position), `score` (relevance score), and optionally `document` (original document text/data). |
| `usage` | `dict` | No | v3.0.0 | No | Usage statistics. Contains `rerank_units` (units consumed for reranking). |

**Example**

```python
from pinecone import Pinecone

pc = Pinecone(api_key="sk-example-key-do-not-use")

result = pc.inference.rerank(
    model="bge-reranker-v2-m3",
    query="best machine learning practices",
    documents=["doc1", "doc2", "doc3", "doc4"],
    top_n=2
)

print(f"Model: {result.model}")
print(f"Rerank units used: {result.usage['rerank_units']}")

# Access ranked documents
for ranked_doc in result.data:
    print(f"Original index: {ranked_doc['index']}, Score: {ranked_doc['score']}")
```

---

### `ModelInfo`

Detailed information about a single inference model, returned by `get_model()` and contained within `ModelInfoList`. Wraps the OpenAPI-generated `ModelInfo` and normalizes the `supported_metrics` field.

**Source:** `pinecone/inference/models/model_info.py:17-52`

**Fields**

| Field | Type | Nullable | Since | Deprecated | Description |
|-------|------|----------|-------|------------|-------------|
| `model` | `str` | No | v3.0.0 | No | The name of the model (e.g., `"multilingual-e5-large"`, `"pinecone-rerank-v0"`). |
| `short_description` | `str` | No | v3.0.0 | No | A brief summary of what the model does. |
| `type` | `str` | No | v3.0.0 | No | The model type: `"embed"` or `"rerank"`. |
| `supported_parameters` | `list[dict]` | No | v3.0.0 | No | List of parameters the model accepts. Each dict contains `parameter` (name), `type` (e.g., `"one_of"`), `value_type` (e.g., `"string"`), `required` (bool), `default` (value), and `allowed_values` (list). |
| `vector_type` | `str` | Yes | v3.0.0 | No | The vector type produced (e.g., `"dense"`, `"sparse"`). Present only for embedding models. |
| `default_dimension` | `int` | Yes | v3.0.0 | No | Default embedding dimension. Range: 1--20000. Present only for embedding models. |
| `supported_dimensions` | `list[int]` | Yes | v3.0.0 | No | List of supported embedding dimensions, if the model allows dimension selection. |
| `modality` | `str` | Yes | v3.0.0 | No | The input modality (e.g., `"text"`). |
| `max_sequence_length` | `int` | Yes | v3.0.0 | No | Maximum number of tokens per input. Minimum: 1. |
| `max_batch_size` | `int` | Yes | v3.0.0 | No | Maximum number of inputs per request. Minimum: 1. |
| `provider_name` | `str` | Yes | v3.0.0 | No | The provider of the model (e.g., `"Pinecone"`). |
| `supported_metrics` | `list[str]` | No | v3.0.0 | No | List of distance metrics supported by the model (e.g., `["cosine", "dotproduct"]`). Empty list if none. Normalized from the API response to always be a list of strings. |

**Methods**

| Method | Return Type | Description |
|--------|-------------|-------------|
| `to_dict()` | `dict` | Returns a plain dictionary representation of the model info with normalized `supported_metrics`. |
| `__getattr__(attr)` | `Any` | Delegates attribute access to the underlying OpenAPI object for any field not directly on the wrapper. |
| `__getitem__(key)` | `Any` | Dictionary-style access; equivalent to attribute access. |
| `__repr__()` | `str` | Returns a JSON-formatted string representation (pretty-printed with 4-space indent). |

**Example**

```python
from pinecone import Pinecone

pc = Pinecone(api_key="sk-example-key-do-not-use")

model_info = pc.inference.get_model(model_name="pinecone-rerank-v0")

# Attribute access
print(model_info.model)                # "pinecone-rerank-v0"
print(model_info.type)                 # "rerank"
print(model_info.max_batch_size)       # 100
print(model_info.supported_metrics)    # []

# Dictionary-style access
print(model_info["short_description"])

# Convert to dict
data = model_info.to_dict()
```

---

### `ModelInfoList`

An iterable collection of `ModelInfo` objects returned by `list_models()`. Wraps the API response and provides convenience accessors.

**Source:** `pinecone/inference/models/model_info_list.py:9-56`

**Fields**

| Field | Type | Nullable | Since | Deprecated | Description |
|-------|------|----------|-------|------------|-------------|
| `models` | `list[ModelInfo]` | No | v3.0.0 | No | The list of model information objects. Accessible via attribute (`models.models`) or key (`models["models"]`). |

**Methods**

| Method | Return Type | Description |
|--------|-------------|-------------|
| `names()` | `list[str]` | Returns a list of model name strings (the `.name` attribute of each `ModelInfo`). |
| `__len__()` | `int` | Returns the number of models in the list. |
| `__iter__()` | `Iterator[ModelInfo]` | Returns an iterator over `ModelInfo` objects. |
| `__getitem__(index)` | `ModelInfo` | Accesses a model by integer index. Also supports `["models"]` key to return the full list. |
| `__getattr__(attr)` | `Any` | Delegates attribute access to the underlying OpenAPI response for future-proofing. |
| `__repr__()` | `str` | Returns a JSON-formatted string representation (pretty-printed, `None` values removed). |

**Example**

```python
from pinecone import Pinecone

pc = Pinecone(api_key="sk-example-key-do-not-use")

models = pc.inference.list_models(type="embed")

# Get all model names
print(models.names())  # ["multilingual-e5-large", "pinecone-sparse-english-v0", ...]

# Iterate
for model in models:
    print(f"{model.model} ({model.type}): {model.short_description}")

# Access by index
first_model = models[0]
print(f"First model: {first_model.model}")

# Check count
print(f"Total embedding models: {len(models)}")
```
