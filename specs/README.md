# Specs

Structured specifications of the Pinecone Python SDK's public interfaces (v8.1.0). Each spec documents methods, parameters, return types, error codes, and notable behaviors with exact source attributions.

## Interfaces

### Control Plane — Admin

- [admin_api_key_operations](interfaces/sdk/admin_api_key_operations.md) — API key CRUD on the Admin client (`admin.api_key`)
- [admin_organization_operations](interfaces/sdk/admin_organization_operations.md) — Organization management on the Admin client (`admin.organization`)
- [admin_project_operations](interfaces/sdk/admin_project_operations.md) — Project management on the Admin client (`admin.project`)

### Control Plane — Index Lifecycle

- [index_operations](interfaces/sdk/index_operations.md) — Index creation (`create_index`, `create_index_for_model`) and configuration (`configure_index`)
- [index_management_operations](interfaces/sdk/index_management_operations.md) — List, describe, delete, and check existence of indexes

### Control Plane — Backup & Restore

- [backup_operations](interfaces/sdk/backup_operations.md) — Create and list backups
- [backup_management_operations](interfaces/sdk/backup_management_operations.md) — Describe and delete backups
- [restore_operations](interfaces/sdk/restore_operations.md) — Create index from backup, monitor restore jobs

### Control Plane — Collections

- [collection_operations](interfaces/sdk/collection_operations.md) — Collection CRUD (create, list, describe, delete)

### Data Plane — Index Client

- [index_client_access_operations](interfaces/sdk/index_client_access_operations.md) — Factory methods to obtain Index/IndexAsyncio clients
- [index_data_operations](interfaces/sdk/index_data_operations.md) — Upsert and delete vectors
- [index_read_operations](interfaces/sdk/index_read_operations.md) — Fetch vectors by ID or metadata filter
- [index_search_and_query_operations](interfaces/sdk/index_search_and_query_operations.md) — Search and query vectors (dense, sparse, hybrid, reranking)
- [index_namespace_operations](interfaces/sdk/index_namespace_operations.md) — Namespace CRUD (create, describe, delete, list)

### Inference

- [inference_operations](interfaces/sdk/inference_operations.md) — Embed, rerank, list_models, and get_model operations
