# Spec

This directory contains structured specifications of the Pinecone Python SDK's public interfaces, behaviors, and limits. Each spec documents methods, parameters, return types, error codes, and notable behaviors with exact values and source attributions.

## Interfaces

### interfaces/sdk

Documents the public Python SDK surface: classes, methods, types, and functions exported at the top level or via submodules.

→ [`spec/interfaces/sdk.md`](interfaces/sdk.md)

### interfaces/sdk/index_operations

Documents index creation and configuration operations: the `create()` and `configure()` methods of the `IndexResource` class, plus the enumerations and data models they use.

→ [`spec/interfaces/sdk/index_operations.md`](interfaces/sdk/index_operations.md)

### interfaces/sdk/inference_operations

Documents inference operations for generating embeddings and reranking documents: the `embed()` and `rerank()` methods of the `Inference` class, plus response models and model enumerations.

→ [`spec/interfaces/sdk/inference_operations.md`](interfaces/sdk/inference_operations.md)

### interfaces/sdk/collection_operations

Documents collection operations for creating and querying collections: the `create_collection()` and `describe_collection()` methods available on the `Pinecone` and `PineconeAsyncio` client instances.

→ [`spec/interfaces/sdk/collection_operations.md`](interfaces/sdk/collection_operations.md)

### interfaces/sdk/backup_operations

Documents backup operations for creating and listing index backups: the `create_backup()` and `list_backups()` methods available on the `Pinecone` and `PineconeAsyncio` client instances.

→ [`spec/interfaces/sdk/backup_operations.md`](interfaces/sdk/backup_operations.md)
