# Test Fixture Factories

This directory contains factory functions for creating OpenAPI model instances used in tests. These factories abstract the construction of models, making tests more maintainable and resilient to OpenAPI spec changes.

## Why Use Factories?

When tests directly instantiate OpenAPI models from `pinecone.core.openapi.*`, they become tightly coupled to the generated code structure. When the OpenAPI spec changes field names or constructor signatures, every test that touches those models breaks.

By using factory functions:
- **Spec changes only require updating factories**, not every test file
- **Tests focus on behavior**, not API structure
- **Consistent defaults** reduce boilerplate in tests
- **Changes can be made incrementally** per test file

## Usage

Import factory functions from `tests.fixtures`:

```python
from tests.fixtures import make_index_model, make_vector, make_list_response

# Use defaults - creates a valid index model with sensible defaults
index = make_index_model()

# Override specific fields as needed
index = make_index_model(name="custom-name", dimension=512)

# Create vectors with sparse values
vector = make_vector(
    id="vec1",
    values=[0.1, 0.2, 0.3],
    sparse_values=make_sparse_values(indices=[0, 2], values=[0.5, 0.7])
)
```

## Available Factories

### Control Plane Models (`db_control_models.py`)

| Factory | Description |
|---------|-------------|
| `make_index_model(**overrides)` | Creates an OpenApiIndexModel with defaults |
| `make_index_status(ready, state)` | Creates an IndexModelStatus |
| `make_schema(fields)` | Creates a Schema with field definitions |
| `make_schema_fields(type, dimension, metric, ...)` | Creates SchemaFields for index schema |
| `make_serverless_deployment(cloud, region)` | Creates a serverless Deployment |
| `make_pod_deployment(environment, pod_type, ...)` | Creates a pod Deployment |
| `make_byoc_deployment(environment)` | Creates a BYOC Deployment |
| `make_index_list(indexes)` | Creates an OpenApiIndexList |
| `make_collection_model(**overrides)` | Creates a CollectionModel |
| `make_collection_list(collections)` | Creates an OpenApiCollectionList |

### Data Plane Models (`db_data_models.py`)

| Factory | Description |
|---------|-------------|
| `make_vector(id, values, metadata, sparse_values)` | Creates an OpenApiVector |
| `make_sparse_values(indices, values)` | Creates an OpenApiSparseValues |
| `make_list_response(vectors, namespace, pagination)` | Creates a ListResponse |
| `make_list_item(id)` | Creates a ListItem |
| `make_pagination(next)` | Creates a Pagination object |
| `make_search_records_vector(values, sparse_indices, sparse_values)` | Creates a SearchRecordsVector |
| `make_vector_values(values)` | Creates a VectorValues wrapper |
| `make_search_records_request_query(top_k, inputs, ...)` | Creates a SearchRecordsRequestQuery |
| `make_search_records_request_rerank(model, rank_fields, ...)` | Creates a SearchRecordsRequestRerank |
| `make_search_records_request(query, fields, rerank)` | Creates a SearchRecordsRequest |

## Adding New Factories

When adding a new factory:

1. **Place it in the appropriate module** (`db_control_models.py` or `db_data_models.py`)
2. **Provide sensible defaults** that create a valid, minimal object
3. **Use `**overrides`** to allow any field to be customized
4. **Add type hints** for parameters and return type
5. **Export from `__init__.py`** so it can be imported from `tests.fixtures`
6. **Add RST docstrings** describing parameters

### Example Factory Template

```python
def make_example_model(
    name: str = "default-name",
    value: int = 42,
    **overrides: Any,
) -> ExampleModel:
    """Create an ExampleModel instance with sensible defaults.

    Args:
        name: The model name
        value: Some integer value
        **overrides: Additional fields to override

    Returns:
        An ExampleModel instance
    """
    kwargs: Dict[str, Any] = {
        "name": name,
        "value": value,
    }
    kwargs.update(overrides)
    return ExampleModel(**kwargs)
```

## Migrating Existing Tests

When migrating a test file to use factories:

1. **Replace imports** - Remove direct OpenAPI model imports, add fixture imports
2. **Update instantiations** - Replace `OpenApiModel(...)` with `make_model(...)`
3. **Keep test assertions** - The factory returns the same model type
4. **Run tests** - Verify behavior is unchanged

### Before

```python
from pinecone.core.openapi.db_data.models import Vector as OpenApiVector

def test_something():
    vec = OpenApiVector(id="1", values=[0.1, 0.2, 0.3])
    assert result == vec
```

### After

```python
from tests.fixtures import make_vector

def test_something():
    vec = make_vector(id="1", values=[0.1, 0.2, 0.3])
    assert result == vec
```
