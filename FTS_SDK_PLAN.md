# Full Text Search SDK Implementation Plan - High Level

## Overview

This document outlines the high-level changes needed in the Python SDK to support Full Text Search (FTS) features from the `2026-01.alpha` API version. The implementation will add support for schema-based index creation and document search operations while maintaining backward compatibility with existing functionality.

## UX Philosophy

**Core Principles:**
1. **Zero Breaking Changes** - All existing code continues to work unchanged
2. **Progressive Enhancement** - New features are additive, not replacements
3. **Developer Experience First** - Reduce boilerplate, provide helpful errors, enable autocomplete
4. **Flexible Access Patterns** - Support multiple ways to accomplish the same task
5. **Smart Defaults** - Minimize required parameters with sensible defaults

**Key UX Strategies:**
- **Builder Patterns** - Fluent APIs for complex objects (schemas, filters)
- **Convenience Methods** - Simple helpers for common use cases
- **Factory Functions** - Easy creation of query objects
- **Flexible Input** - Accept single items or lists, dicts or objects
- **Helpful Errors** - Clear messages with suggestions and migration hints
- **Type Safety** - Extensive type hints for IDE support and error prevention

## Current State

- SDK currently uses `2025-10` stable API version
- OpenAPI client code is generated in `pinecone/core/openapi/`
- No support for alpha API versions yet
- Existing index creation uses `spec`-based approach
- Existing search uses vector-based queries only

## Required Changes

### 1. OpenAPI Code Generation for Alpha API

**What:** Generate Python client code from `2026-01.alpha` API specifications

**Where:**
- `codegen/apis/_build/2026-01.alpha/` contains the OAS files
- Need to generate code to `pinecone/core/openapi/db_control_alpha/` and `pinecone/core/openapi/db_data_alpha/`

**Key Generated Models:**
- `Schema`, `SchemaField` (control plane)
- `DocumentSearchRequest`, `DocumentSearchResponse`, `Document` (data plane)
- `TextQuery`, `VectorQuery`, `ScoreByQuery` (data plane)
- `DocumentUpsertRequest`, `DocumentUpsertResponse` (data plane)

**Action:** Run OpenAPI codegen script to generate alpha API client code

---

### 2. API Version Management

**What:** Add support for using `2026-01.alpha` API version alongside stable version

**Changes Needed:**
- Create alpha API client instances when FTS features are used
- Update API client setup to support version selection
- Ensure alpha API version header (`X-Pinecone-Api-Version: 2026-01.alpha`) is sent

**Files to Modify:**
- `pinecone/utils/setup_openapi_client.py` - Add alpha API version support
- `pinecone/db_control/db_control.py` - Use alpha client for schema operations
- `pinecone/db_data/index.py` - Use alpha client for document operations

---

### 3. Schema Models (Control Plane)

**What:** Create user-friendly Python models for index schema definition

**New Files:**
- `pinecone/db_control/models/schema.py` - Schema container and field models
- `pinecone/db_control/models/schema_field.py` - Field type definitions
- `pinecone/db_control/models/schema_builder.py` - Fluent builder for schemas

**Key Models:**
- `Schema` - Container for field definitions (DictLike for easy conversion)
- `StringField` - String field with `filterable` and `full_text_searchable` options
- `IntegerField` - Integer field with `filterable` option
- `DenseVectorField` - Vector field with dimension and metric
- `SparseVectorField` - Sparse vector field
- `SemanticTextField` - Integrated inference field
- `SchemaBuilder` - Fluent builder for constructing schemas

**UX Features:**
- Builder pattern: `SchemaBuilder().string("title", full_text_searchable=True).build()`
- Dict-like access: `schema["title"]` or `schema.fields["title"]`
- Validation with helpful error messages
- Type hints for IDE autocomplete

**Purpose:** Provide Pythonic interface for defining index schemas with minimal boilerplate

---

### 4. Index Creation Updates (Control Plane)

**What:** Update `create_index()` to support schema-based index creation

**Changes:**
- Add `schema` parameter to `create_index()` methods (optional, for backward compatibility)
- Support new `deployment` structure (replacing `spec` for serverless)
- Handle `read_capacity` as top-level parameter
- Maintain backward compatibility with existing `spec`-based API
- Auto-detect API version: use alpha when `schema` provided, stable otherwise

**Files to Modify:**
- `pinecone/db_control/resources/sync/index.py`
- `pinecone/db_control/resources/asyncio/index.py`
- `pinecone/pinecone.py`
- `pinecone/pinecone_asyncio.py`

**UX Enhancements:**
- **Convenience Methods:**
  - `create_text_index()` - Simplified for text-only indexes
  - `create_hybrid_index()` - Helper for text + vector indexes
- **Smart Defaults:**
  - Default deployment: serverless/aws/us-east-1 if not specified
  - Auto-generated index names if not provided
- **Backward Compatibility:**
  - Existing `create_index(name, spec=..., dimension=...)` calls work unchanged
  - New `create_index(name, schema=...)` calls use alpha API automatically
  - Both patterns can coexist

**Key Consideration:** Zero breaking changes - all existing code continues to work

---

### 5. Document Query Models (Data Plane)

**What:** Create user-friendly models for document search queries

**New Files:**
- `pinecone/db_data/dataclasses/text_query.py` - TextQuery wrapper
- `pinecone/db_data/dataclasses/vector_query.py` - VectorQuery wrapper
- `pinecone/db_data/dataclasses/score_by_query.py` - ScoreByQuery union type
- `pinecone/db_data/query_helpers.py` - Convenience factory functions

**Key Models:**
- `TextQuery` - Text search with `field` and `text_query` string (DictLike)
- `VectorQuery` - Vector search with `field` and `values`/`sparse_values` (DictLike)
- `ScoreByQuery` - Union type for query objects

**UX Enhancements:**
- **Factory Functions:**
  - `text_query(field, query)` - Simple text query creation
  - `text_query.from_phrase(field, phrase)` - Create phrase query
  - `text_query.from_terms(field, required, optional)` - Create with required/optional terms
  - `vector_query(field, values)` - Simple vector query creation
- **Builder Pattern** (optional):
  - `TextQueryBuilder().field("title").query("pink panther").build()`
- **Validation:**
  - Clear errors for v0 limitations (single query only)
  - Helpful suggestions for common mistakes

**Purpose:** Pythonic interface matching API schema structure with convenience helpers

---

### 6. Document Search Implementation (Data Plane)

**What:** Add `search_documents()` method for text and hybrid search

**New Method Signature:**
```python
def search_documents(
    self,
    namespace: str,
    score_by: list[TextQuery | VectorQuery] | TextQuery | VectorQuery,  # Accept single or list
    filter: dict[str, Any] | FilterBuilder | None = None,  # Support FilterBuilder
    include_fields: list[str] | str | None = ["*"],  # Default to all fields
    top_k: int = 10,
) -> DocumentSearchResponse
```

**UX Enhancements:**
- **Flexible Input:**
  - Accept single query: `score_by=TextQuery(...)` (auto-wrap in list)
  - Accept list: `score_by=[TextQuery(...)]`
  - Support `FilterBuilder` for type-safe filter construction
- **Smart Defaults:**
  - `include_fields` defaults to `["*"]` (return all fields)
  - Clear error if `score_by` is empty or has >1 item (v0 limitation)
- **Convenience Overloads:**
  - `search_documents(namespace, text_query="...", field="title")` - Simple text search
  - `search_documents(namespace, vector=[...], field="embedding")` - Simple vector search

**Files to Modify:**
- `pinecone/db_data/index.py` - Add sync method
- `pinecone/db_data/index_asyncio.py` - Add async method
- `pinecone/db_data/interfaces.py` - Add abstract method
- `pinecone/db_data/index_asyncio_interface.py` - Add async abstract method
- `pinecone/db_data/resources/sync/record.py` - Implement endpoint call
- `pinecone/db_data/resources/asyncio/record_asyncio.py` - Async endpoint call
- `pinecone/db_data/request_factory.py` - Add request building logic

**Endpoint:** `POST /namespaces/{namespace}/documents/search` (alpha API)

**Backward Compatibility:**
- Existing `query()` and `search()` methods unchanged
- New method is additive, doesn't replace existing functionality

---

### 7. Document Response Models (Data Plane)

**What:** Create response models for document search results

**New Files:**
- `pinecone/db_data/dataclasses/document_search_response.py` - Response wrapper
- `pinecone/db_data/dataclasses/document.py` - Document model wrapper

**Key Models:**
- `DocumentSearchResponse` - Contains `documents` array and `usage` (DictLike)
- `Document` - Contains `_id`, `score`, and dynamic fields from `include_fields` (DictLike)

**UX Enhancements:**
- **Flexible Access:**
  - Attribute access: `doc.title`, `doc.genre` (with type hints)
  - Dict access: `doc["title"]`, `doc.get("title")`
  - Both patterns work for maximum compatibility
- **Helper Methods:**
  - `response.documents` - List of Document objects
  - `response.usage` - Usage information
  - `response[0]` - Access first document (if needed)
- **Type Safety:**
  - Type hints for common fields
  - IDE autocomplete support
  - Clear error messages for missing fields

**Purpose:** User-friendly access to search results with multiple access patterns

---

### 8. Document Upsert Support (Data Plane)

**What:** Add support for document upsert API (if not already implemented)

**Endpoint:** `POST /namespaces/{namespace}/documents/upsert` (alpha API)

**Changes:**
- May already exist via `upsert_records()` - verify and enhance if needed
- Ensure it works with schema-based indexes

---

### 9. Validation and Error Handling

**What:** Add validation for FTS-specific constraints with helpful error messages

**Validations:**
- V0 limitation: `score_by` array must have exactly 1 item
  - Error: "V0 limitation: score_by must contain exactly 1 query. Found {count}. For multiple queries, wait for v1 support."
- Schema field type validation
  - Error: "Invalid field type '{type}'. Must be one of: string, integer, dense_vector, sparse_vector, semantic_text"
- Field name validation (no reserved names)
  - Error: "Field name '{name}' is reserved. Reserved names: _id, _values, _sparse_values"
- `include_fields` format validation (array or "*" string)
  - Error: "include_fields must be a list of strings or the string '*'. Got: {type}"

**UX Enhancements:**
- **Helpful Error Messages:**
  - Include what was wrong and what's expected
  - Provide suggestions when possible
  - Link to documentation for complex issues
- **Early Validation:**
  - Validate before API call to provide immediate feedback
  - Type checking with mypy-friendly errors
- **Migration Hints:**
  - Suggest alternatives when using deprecated patterns
  - Provide code examples in error messages when helpful

**Files to Create/Modify:**
- `pinecone/db_data/validators.py` - Add FTS validators
- `pinecone/db_control/validators.py` - Add schema validators
- `pinecone/exceptions/fts_exceptions.py` - FTS-specific exceptions

---

### 10. Testing

**What:** Comprehensive test coverage for FTS features

**Integration Tests:**
- Schema-based index creation
- Document upsert with text fields
- Text search queries (phrase, required terms)
- Metadata filtering with text search
- Field selection in responses
- Error cases and edge cases

**Unit Tests:**
- Schema model serialization/deserialization
- Query model validation
- Request factory logic
- Response model parsing

**Files to Create:**
- `tests/integration/rest_sync/db/control/test_create_index_with_schema.py`
- `tests/integration/rest_sync/db/data/test_document_search.py`
- `tests/integration/rest_async/db/data/test_document_search_async.py`
- `tests/unit/db_control/test_schema_models.py`
- `tests/unit/db_data/test_text_query.py`

---

### 11. Documentation

**What:** Update documentation with FTS examples and guides

**Files to Update:**
- `docs/working-with-indexes.rst` - Schema-based index creation
- `docs/rest.rst` - Document search API
- `README.md` - Add FTS examples if needed

**Content:**
- Creating indexes with `full_text_searchable` fields
- Performing text search queries
- Combining text and vector search (v0 limitations)
- Query syntax reference

---

## Implementation Phases

### Phase 1: Foundation (Prerequisites)
1. Run OpenAPI codegen for `2026-01.alpha` API
2. Verify generated models match API schema
3. Set up API version management infrastructure

### Phase 2: Control Plane (Index Creation)
1. Create schema models
2. Update `create_index()` methods
3. Add integration tests

### Phase 3: Data Plane (Search)
1. Create query models
2. Implement `search_documents()` method
3. Create response models
4. Add integration tests

### Phase 4: Polish
1. Add validation and error handling
2. Update documentation
3. Edge case testing
4. Backward compatibility verification

---

## UX Improvements & Backward Compatibility Strategy

### UX Enhancements

1. **Schema Builder Pattern** (Similar to `FilterBuilder`)
   - Fluent API for building schemas: `SchemaBuilder().string("title", full_text_searchable=True).integer("year", filterable=True).build()`
   - Reduces boilerplate and prevents errors
   - Provides autocomplete and type checking

2. **Convenience Methods & Helpers**
   - `create_text_index()` - Simplified method for text-only indexes
   - `create_hybrid_index()` - Helper for indexes with both text and vector fields
   - `text_query()` - Factory function for creating TextQuery objects
   - `vector_query()` - Factory function for creating VectorQuery objects

3. **Smart Defaults**
   - Auto-detect API version based on parameters (use alpha when schema provided)
   - Default `include_fields=["*"]` for document search (return all fields)
   - Sensible defaults for deployment (serverless/aws/us-east-1)

4. **Better Error Messages**
   - Clear validation errors with suggestions
   - Helpful messages for v0 limitations
   - Migration hints when using deprecated patterns

5. **Document Model Access**
   - `Document` objects support attribute access: `doc.title`, `doc.genre`
   - Fallback to dict access: `doc["title"]`
   - Type hints for better IDE support

6. **Query Building Helpers**
   - `TextQuery.from_string()` - Create from simple string
   - `TextQuery.from_phrase()` - Create phrase query
   - `TextQuery.from_terms()` - Create query with required/optional terms

### Backward Compatibility Strategy

1. **Zero Breaking Changes**
   - All existing `create_index()` calls continue to work unchanged
   - Existing `query()` and `search()` methods remain fully functional
   - No changes to existing method signatures

2. **Gradual Migration Path**
   - Old way still works: `create_index(name="idx", spec=ServerlessSpec(...), dimension=1536)`
   - New way available: `create_index(name="idx", schema=Schema(...))`
   - Both can coexist - SDK detects which API version to use

3. **Deprecation Strategy** (Future)
   - Add deprecation warnings only after FTS is stable
   - Provide clear migration guides
   - Long deprecation period (multiple releases)

4. **API Version Selection**
   - Automatic: Use alpha API when schema is provided, stable otherwise
   - Explicit: Allow users to specify API version if needed
   - Transparent: Users don't need to think about API versions

5. **Response Compatibility**
   - `DocumentSearchResponse` has similar structure to existing search responses
   - `Document` objects can be converted to dicts for compatibility
   - Existing code patterns continue to work

## Key Design Decisions

1. **Dual API Version Support**: SDK will use stable API by default, alpha API when FTS features are used
2. **Backward Compatibility**: Existing methods (`create_index()` with `spec`, `query()`, `search()`) remain unchanged
3. **Additive API**: New methods (`search_documents()`) don't replace existing ones
4. **Type Safety**: Extensive use of type hints and TypedDict for API contracts
5. **User-Friendly Wrappers**: Create Pythonic models on top of generated OpenAPI models
6. **Builder Patterns**: Use fluent builder APIs for complex objects (schemas, queries)
7. **Convenience Methods**: Provide simple helpers for common use cases
8. **Smart Defaults**: Reduce boilerplate with sensible defaults

---

## Dependencies

- ✅ API submodule on `jhamon/fts` branch with FTS API definitions
- ⏳ OpenAPI codegen must be run to generate `2026-01.alpha` client code
- ⏳ Backend must support schema-based index creation and document search endpoints

---

## Success Criteria

- ✅ Can create indexes with `full_text_searchable` fields using schema
- ✅ Can perform text search queries with phrase matching and required terms
- ✅ Can combine metadata filtering with text search
- ✅ Can select specific fields in search responses
- ✅ All tests pass (unit + integration)
- ✅ Documentation is complete
- ✅ Backward compatibility maintained (existing code continues to work)

---

## Notes

- V0 limitations: Single text field OR single vector field per query (not both)
- `score_by` array limited to 1 item in v0 (minItems: 1, maxItems: 1)
- Text query syntax is handled by backend; SDK passes through query strings
- `include_fields` supports both array of strings and `"*"` string
- Filter expressions support `$text_match` operator for required/excluded terms

## Migration Examples

### Creating an Index

**Old Way (Still Works):**
```python
pc.create_index(
    name="my-index",
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)
```

**New Way (Schema-Based):**
```python
# Using SchemaBuilder (recommended)
schema = SchemaBuilder() \
    .string("title", full_text_searchable=True) \
    .integer("year", filterable=True) \
    .dense_vector("embedding", dimension=1536, metric="cosine") \
    .build()

pc.create_index(name="my-index", schema=schema)

# Or using convenience method
pc.create_hybrid_index(
    name="my-index",
    text_fields={"title": {"full_text_searchable": True}},
    vector_fields={"embedding": {"dimension": 1536, "metric": "cosine"}}
)
```

### Searching Documents

**Old Way (Still Works):**
```python
results = index.query(vector=[...], top_k=10, filter={"genre": "comedy"})
```

**New Way (Document Search):**
```python
# Simple text search
results = index.search_documents(
    namespace="movies",
    score_by=text_query("title", "pink panther"),
    filter={"genre": "comedy"},
    top_k=10
)

# Using FilterBuilder
filter = FilterBuilder().eq("genre", "comedy").build()
results = index.search_documents(
    namespace="movies",
    score_by=TextQuery(field="title", text_query="pink panther"),
    filter=filter
)

# Access results
for doc in results.documents:
    print(f"{doc.title} ({doc.year}) - Score: {doc.score}")
```

## Breaking Changes Assessment

**Zero Breaking Changes:**
- All existing code continues to work unchanged
- New features are additive only
- No method signatures changed
- No behavior changes to existing methods

**Future Considerations:**
- When FTS moves to stable API, may deprecate old patterns
- Will provide long deprecation period and migration guides
- Will maintain backward compatibility for multiple releases
