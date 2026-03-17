# Spec

Comprehensive reference documentation for Pinecone Python SDK public interfaces, covering all methods, parameters, return types, data models, and error handling.

## Interfaces

### interfaces/sdk/admin_api_key_operations

Documents all API key management operations on the Admin client: `create()`, `list()`, `fetch()` (with `get()` and `describe()` aliases), `update()`, and `delete()`.

→ [`spec/interfaces/sdk/admin_api_key_operations.md`](interfaces/sdk/admin_api_key_operations.md)

### interfaces/sdk/index_management_operations

Documents all index management methods on the Pinecone and PineconeAsyncio clients: listing, describing, deleting, and checking the existence of indexes.

→ [`spec/interfaces/sdk/index_management_operations.md`](interfaces/sdk/index_management_operations.md)

### interfaces/sdk/index_creation_operations

Documents index creation methods on the Pinecone and PineconeAsyncio clients: `create_index()` for general purpose indexes with serverless, pod, or BYOC configurations, and `create_index_for_model()` for serverless indexes with integrated inference.

→ [`spec/interfaces/sdk/index_creation_operations.md`](interfaces/sdk/index_creation_operations.md)

### interfaces/sdk/index_data_operations

Documents vector write operations on the Index and AsyncIndex clients: `upsert()` for inserting or updating vectors with dense and sparse values and metadata, and `delete()` for removing vectors by ID, metadata filter, or bulk deletion.

→ [`spec/interfaces/sdk/index_data_operations.md`](interfaces/sdk/index_data_operations.md)

### interfaces/sdk/index_read_operations

Documents vector read operations on the Index and AsyncIndex clients: `fetch()` for retrieving vectors by ID and `fetch_by_metadata()` for retrieving vectors matching a metadata filter.

→ [`spec/interfaces/sdk/index_read_operations.md`](interfaces/sdk/index_read_operations.md)

### interfaces/sdk/collection_operations

Documents all collection management operations on the Pinecone and PineconeAsyncio clients: creating, listing, deleting, and describing collections.

→ [`spec/interfaces/sdk/collection_operations.md`](interfaces/sdk/collection_operations.md)

### interfaces/sdk/admin_project_operations

Documents all project management operations on the Admin client: creating, listing, retrieving, updating, and deleting projects for Pinecone organizations.

→ [`spec/interfaces/sdk/admin_project_operations.md`](interfaces/sdk/admin_project_operations.md)

### interfaces/sdk/backup_and_restore_operations

Documents all backup and restore management operations on the Pinecone and PineconeAsyncio clients: creating backups, listing and describing backups, deleting backups, and monitoring restore jobs.

→ [`spec/interfaces/sdk/backup_and_restore_operations.md`](interfaces/sdk/backup_and_restore_operations.md)

### interfaces/sdk/index_search_and_query_operations

Documents the `search()` and `query()` methods on the Index and AsyncIndex clients for vector similarity search, metadata filtering, hybrid search with sparse vectors, and optional reranking with integrated inference.

→ [`spec/interfaces/sdk/index_search_and_query_operations.md`](interfaces/sdk/index_search_and_query_operations.md)

### interfaces/sdk/inference_embed_and_rerank_operations

Documents inference operations on the Pinecone client for embedding and reranking: `embed()` for converting text to embeddings using various embedding models, and `rerank()` for reranking search results using reranking models.

→ [`spec/interfaces/sdk/inference_embed_and_rerank_operations.md`](interfaces/sdk/inference_embed_and_rerank_operations.md)

### interfaces/sdk/admin_organization_operations

Documents organization management operations on the Admin client: listing, retrieving, updating, and deleting organizations within a Pinecone account.

→ [`spec/interfaces/sdk/admin_organization_operations.md`](interfaces/sdk/admin_organization_operations.md)

### interfaces/sdk/index_namespace_operations

Documents namespace management operations on the Index and AsyncIndex clients: creating, describing, deleting, and listing namespaces.

→ [`spec/interfaces/sdk/index_namespace_operations.md`](interfaces/sdk/index_namespace_operations.md)

### interfaces/sdk/index_configuration_operations

Documents index configuration modification operations on the Pinecone and PineconeAsyncio clients: `configure_index()` for modifying replicas, pod type, deletion protection, tags, integrated inference settings, and serverless read capacity.

→ [`spec/interfaces/sdk/index_configuration_operations.md`](interfaces/sdk/index_configuration_operations.md)
