# Pinecone SDK - Feature Documentation

> **Purpose**: This document provides a comprehensive catalog of all public interfaces, methods, parameters, validations, edge cases, and performance/reliability/UX attributes of the Pinecone SDK. Use this as a checklist to evaluate feature parity across different Pinecone SDKs and programming languages.

**Last Updated**: 2026-01-16
**Reference Implementation**: Python SDK v7+

---

## Table of Contents

1. [Main Client Classes](#1-main-client-classes)
2. [Index Client Classes (Data Plane)](#2-index-client-classes-data-plane)
3. [Admin API](#3-admin-api)
4. [Inference API](#4-inference-api)
5. [Error Handling](#5-error-handling)
6. [Retry Mechanisms](#6-retry-mechanisms)
7. [Caching](#7-caching)
8. [Configuration & Environment](#8-configuration--environment)
9. [Performance Features](#9-performance-features)
10. [Validation & Edge Cases](#10-validation--edge-cases)

---

## 1. Main Client Classes

### 1.1 Primary Client (Synchronous)

**Purpose**: Main entry point for interacting with Pinecone's control plane APIs

**Initialization Configuration**:
- ☑ **API Key** (string, optional) - API key for authentication; can be read from `PINECONE_API_KEY` environment variable if not provided
- ☑ **Host** (string, optional) - Control plane host; defaults to `api.pinecone.io`
- ☑ **Proxy URL** (string, optional) - URL of proxy server for network requests
- ☑ **Proxy Headers** (dictionary, optional) - Additional headers for proxy authentication
- ☑ **SSL CA Certificates** (string, optional) - Path to SSL CA certificate bundle in PEM format
- ☑ **SSL Verification** (boolean, optional) - Enable/disable SSL certificate verification; defaults to enabled
- ☑ **Additional Headers** (dictionary, optional) - Custom HTTP headers to include in all API calls
- ☑ **Thread Pool Size** (integer, optional) - Size of thread pool for parallel operations; typically defaults to 5 × CPU count

**Accessible Resources**:
- ☑ **Inference API Client** - Lazy-initialized client for embedding and reranking operations
- ☑ **Database Control Client** - Lazy-initialized client for index management
- ☑ **Configuration Object** - Access to internal configuration for advanced use cases

**Index Management Operations**:

#### Create Index
**Purpose**: Create a new vector index with specified configuration

**Parameters**:
  - **Name** (required, string): Index identifier
    - Must contain only lowercase letters, numbers, and hyphens
    - Maximum 45 characters
    - Cannot start or end with hyphen
    - Must be unique within the project
  - **Spec** (required, object): Deployment configuration
    - **ServerlessSpec**: Cloud provider, region, read capacity mode, optional schema
    - **PodSpec**: Environment, pod type, pod count, replicas, shards, optional source collection
    - **ByocSpec**: Bring-your-own-cloud configuration
  - **Dimension** (optional, integer): Vector dimension
    - Required when vector_type is "dense"
    - Should not be provided when vector_type is "sparse"
  - **Metric** (optional, string): Similarity metric for queries
    - Values: "cosine", "dotproduct", or "euclidean"
    - Defaults to "cosine"
  - **Timeout** (optional, integer or null): Wait behavior for index readiness
    - null: Wait indefinitely until ready
    - -1: Return immediately without waiting
    - ≥0: Timeout after specified seconds
  - **Deletion Protection** (optional, string): Protection against accidental deletion
    - Values: "enabled" or "disabled"
    - Defaults to "disabled"
  - **Vector Type** (optional, string): Type of vectors to store
    - Values: "dense" or "sparse"
    - Defaults to "dense"
  - **Tags** (optional, dictionary): Key-value pairs for organization and filtering

**Validations**:
  - Index name must be unique within project
  - Dimension required when vector_type="dense"
  - Dimension should not be passed when vector_type="sparse"

**Returns**: Index model object with complete index details

#### Create Index for Model (Integrated Inference)
**Purpose**: Create an index with integrated embedding generation

**Parameters**:
  - **Name** (required, string): Index name (same validation rules as create_index)
  - **Cloud** (required, string): Cloud provider - "aws", "gcp", or "azure"
  - **Region** (required, string): Deployment region for the chosen cloud provider
  - **Embed Configuration** (required, object): Embedding model configuration
    - **Model**: Embedding model identifier
    - **Metric**: Distance metric supported by the model
    - **Field Map**: Mapping of input fields to embedding targets
  - **Read Capacity** (optional, object): Read capacity mode (OnDemand or Dedicated)
  - **Schema** (optional, object): Metadata schema defining filterable fields
  - **Tags** (optional, dictionary): Key-value pairs for organization
  - **Deletion Protection** (optional, string): "enabled" or "disabled"; defaults to "disabled"
  - **Timeout** (optional, integer or null): Wait behavior for index readiness

**Validations**:
  - Index vector type and dimension must match the embedding model's output
  - Index metric must be supported by the embedding model

**Returns**: Index model object with complete index details

#### Create Index from Backup
**Purpose**: Restore an index from a previously created backup

**Parameters**:
  - **Name** (required, string): Name for the new index
  - **Backup ID** (required, string): Identifier of the backup to restore from
  - **Deletion Protection** (optional, string): "enabled" or "disabled"; defaults to "disabled"
  - **Tags** (optional, dictionary): Key-value pairs for organization
  - **Timeout** (optional, integer or null): Wait behavior for index readiness

**Returns**: Index model object with complete index details

#### Delete Index
**Purpose**: Permanently delete an index and all its data

**Parameters**:
  - **Name** (required, string): Index name to delete
  - **Timeout** (optional, integer or null): Poll timeout for deletion completion

**Validations**:
  - Operation fails if deletion_protection is "enabled"

**Behavior**: Index transitions to "Terminating" state before final deletion

**Returns**: No return value (void)

#### List Indexes
**Purpose**: Retrieve all indexes in the current project

**Parameters**: None

**Returns**: Collection of index model objects
- Should support iteration over all indexes
- Should provide convenience method to extract just index names

#### Describe Index
**Purpose**: Get detailed information about a specific index

**Parameters**:
  - **Name** (required, string): Index name

**Returns**: Index model object containing:
- Name, dimension, metric, host URL
- Status (e.g., "Ready", "Initializing", "Terminating")
- Deployment spec (serverless/pod configuration)
- Vector type, deletion protection status, tags

#### Check Index Exists
**Purpose**: Check if an index exists without fetching full details

**Parameters**:
  - **Name** (required, string): Index name

**Returns**: Boolean - true if index exists, false otherwise

#### Configure Index
**Purpose**: Modify settings of an existing index (scaling, protection, tags)

**Parameters**:
  - **Name** (required, string): Index name
  - **Replicas** (optional, integer): Number of replicas (pod indexes only)
    - Use case: Horizontal scaling for pod-based indexes
  - **Pod Type** (optional, string): Pod hardware configuration (pod indexes only)
    - Use case: Vertical scaling for pod-based indexes
  - **Deletion Protection** (optional, string): "enabled" or "disabled"
  - **Tags** (optional, dictionary): Tags to merge with existing tags
    - Empty string values remove tags
  - **Embed Configuration** (optional, object): Update embedding model settings (integrated indexes only)
  - **Read Capacity** (optional, object): Update capacity mode (serverless indexes only)

**Use Cases**:
  - Horizontal scaling (replicas) for pod indexes
  - Vertical scaling (pod_type) for pod indexes
  - Read capacity configuration for serverless indexes
  - Enable/disable deletion protection
  - Tag management

**Returns**: No return value (void)

**Collection Operations**:

#### Create Collection
**Purpose**: Create a static snapshot of a pod-based index

**Parameters**:
  - **Name** (required, string): Collection name
  - **Source** (required, string): Source pod-based index name

**Limitations**: Only works with pod-based indexes (not serverless)

**Returns**: No return value (void)

#### List Collections
**Purpose**: Retrieve all collections in the project

**Returns**: Collection of collection objects with details
- Should support iteration
- Should provide convenience method to extract just collection names

#### Describe Collection
**Purpose**: Get details about a specific collection

**Parameters**:
  - **Name** (required, string): Collection name

**Returns**: Collection object with name, source index, status, and size

#### Delete Collection
**Purpose**: Permanently delete a collection

**Parameters**:
  - **Name** (required, string): Collection name

**Returns**: No return value (void)

**Backup Operations**:

#### Create Backup
**Purpose**: Create a point-in-time backup of an index

**Parameters**:
  - **Index Name** (required, string): Source index name
  - **Backup Name** (required, string): Name for the backup
  - **Description** (optional, string): Human-readable description

**Returns**: Backup model object with backup details

#### List Backups
**Purpose**: Retrieve backups, optionally filtered by index

**Parameters**:
  - **Index Name** (optional, string): Filter to backups for specific index
  - **Limit** (optional, integer): Maximum results per page; defaults to 10
  - **Pagination Token** (optional, string): Token for next page

**Returns**: Paginated list of backup model objects

#### Describe Backup
**Purpose**: Get details about a specific backup

**Parameters**:
  - **Backup ID** (required, string): Backup identifier

**Returns**: Backup model object with details

#### Delete Backup
**Purpose**: Permanently delete a backup

**Parameters**:
  - **Backup ID** (required, string): Backup identifier

**Returns**: No return value (void)

**Restore Job Operations**:

#### List Restore Jobs
**Purpose**: Retrieve restore operations in progress or completed

**Parameters**:
  - **Limit** (optional, integer): Maximum results per page; defaults to 10
  - **Pagination Token** (optional, string): Token for next page

**Returns**: Paginated list of restore job model objects

#### Describe Restore Job
**Purpose**: Get details about a specific restore operation

**Parameters**:
  - **Job ID** (required, string): Restore job identifier

**Returns**: Restore job model object with status and progress

**Data Plane Client Factory**:

#### Create Index Client (Synchronous)
**Purpose**: Create a client for vector operations on a specific index

**Parameters**:
  - **Name** (optional, string): Index name (triggers automatic host lookup)
  - **Host** (optional, string): Direct host URL (skips control plane lookup)
  - **Thread Pool Size** (optional, integer): Size of thread pool for parallel operations
  - **Connection Pool Max Size** (optional, integer): Maximum connections in pool

**Validations**:
  - Either name or host must be specified
  - Host must be a valid URL format

**Returns**: Index client instance for data operations

#### Create Index Client (Asynchronous)
**Purpose**: Create an async client for vector operations on a specific index

**Parameters**:
  - **Host** (required, string): Index host URL

**Returns**: Async index client instance for data operations

### 1.2 Asynchronous Client

**Purpose**: Async/await variant of the primary client

**Initialization Configuration**:
- Same as synchronous client except:
  - ☑ **Proxy Headers NOT supported** (should raise error if provided)

**Resource Cleanup**:
- ☑ Should support async context managers (async with)
- ☑ Should provide explicit close() method for cleanup

**Methods**: All same operations as synchronous client
- ☑ All methods should be awaitable
- ☑ Same parameters and validations as sync version

### 1.3 GRPC Client

**Purpose**: High-performance variant using gRPC protocol for data operations

**Relationship**: Extends or wraps the primary synchronous client

**Key Differences**:
- ☑ Index client factory returns gRPC-based client instead of REST
- ☑ Uses gRPC protocol for data plane operations (better performance)
- ☑ Control plane operations remain REST-based
- ☑ May require additional dependencies or optional installation

---

## 2. Index Client (Data Plane Operations)

### 2.1 Index Client (Synchronous REST)

**Purpose**: Execute vector operations on a specific index

**Initialization Configuration**:
- **API Key** (required, string): Authentication credential
- **Host** (required, string): Index-specific host URL
- **Thread Pool Size** (optional, integer): Defaults to 5 × CPU count
- **Additional Headers** (optional, dictionary): Custom HTTP headers
- **Connection Pool Max Size** (optional, integer): Maximum concurrent connections

**Resource Cleanup**:
- ☑ Should support context managers (with statement)
- ☑ Should provide explicit close() method to cleanup connections

**Accessible Resources**:
- ☑ **Bulk Import Operations** - Lazy-initialized resource for managing bulk imports
- ☑ **Namespace Operations** - Lazy-initialized resource for namespace management

**Vector Operations**:

#### Upsert Vectors
**Purpose**: Insert or update vectors in the index

**Vector Formats Supported**:
SDKs should accept multiple flexible formats for vectors:
  - **Structured objects** with id, values, optional sparse_values, optional metadata
  - **2-element tuples/arrays**: (id, values)
  - **3-element tuples/arrays**: (id, values, metadata)
  - **Dictionaries/maps**: {"id": string, "values": array, "sparse_values": object, "metadata": object}
  - **Sparse vectors**: Should support sparse_values with indices and values arrays

**Parameters**:
  - **Vectors** (required, array): Collection of vectors to upsert
  - **Namespace** (optional, string): Target namespace; defaults to empty string ""
  - **Batch Size** (optional, integer): Automatically chunk large operations into batches
    - When provided, should enable progress tracking if available
  - **Show Progress** (optional, boolean): Display progress indicators; defaults to true
  - **Async Request** (optional, boolean): Return future/promise for parallel execution

**Validations**:
  - Cannot combine async_req with batch_size (incompatible modes)
  - Vector dimensions must match index dimension

**Returns**: Response object with:
  - **Upserted Count**: Number of vectors successfully upserted
  - **Response Metadata**: Optional timing/request info

#### Upsert from DataFrame/Table
**Purpose**: Convenience method for upserting from tabular data structures

**Parameters**:
  - **DataFrame** (required, data structure): Tabular data with columns:
    - Required: id, values
    - Optional: sparse_values, metadata
  - **Namespace** (optional, string): Target namespace
  - **Batch Size** (optional, integer): Defaults to 500
  - **Show Progress** (optional, boolean): Display progress; defaults to true

**Validations**:
  - May require specific library dependencies (e.g., pandas, polars)
  - Input must be a compatible tabular structure

**Returns**: Response with upserted count

**Note**: This is a convenience method - not all SDKs may implement it

#### Upsert Records (Integrated Inference)
**Purpose**: Upsert raw data that will be automatically embedded

**Parameters**:
  - **Namespace** (required, string): Target namespace
  - **Records** (required, array of objects): Raw data records
    - Must include `id` or `_id` field
    - Must include fields specified in index embed configuration's field_map

**Use Case**: For indexes created with integrated inference (create_index_for_model)

**Behavior**: Service automatically generates embeddings based on field_map configuration

**Returns**: Response with upserted count

#### Query Vectors
**Purpose**: Find similar vectors in the index

**Parameters**:
  - **Top K** (required, integer): Number of results to return; must be ≥ 1
  - **Vector** (optional, array of floats): Query vector (mutually exclusive with id)
  - **ID** (optional, string): Use an existing vector as query (mutually exclusive with vector)
  - **Namespace** (optional, string): Namespace to search; defaults to empty string ""
  - **Filter** (optional, object): Metadata filter expression
  - **Include Values** (optional, boolean): Return vector values in results; defaults to false
  - **Include Metadata** (optional, boolean): Return metadata in results; defaults to false
  - **Sparse Vector** (optional, object): Sparse vector for hybrid search
  - **Async Request** (optional, boolean): Return future/promise for parallel execution

**Validations**:
  - Positional arguments should not be allowed (force named parameters)
  - top_k must be ≥ 1
  - Exactly one of vector or id must be provided (not both, not neither)

**Returns**: Response object with:
  - **Matches**: Array of scored results (id, score, optional values, optional metadata)
  - **Namespace**: Namespace queried
  - **Usage**: Optional resource usage information

#### Query Multiple Namespaces
**Purpose**: Query across multiple namespaces in parallel and merge results

**Parameters**:
  - **Vector** (required, array of floats): Query vector
  - **Namespaces** (required, array of strings): Namespaces to query
  - **Metric** (required, string): Similarity metric for merging results across namespaces
    - Values: "cosine", "euclidean", or "dotproduct"
  - **Top K** (optional, integer): Results per namespace; defaults to 10
  - **Filter** (optional, object): Metadata filter expression
  - **Include Values** (optional, boolean): Return vector values; defaults to false
  - **Include Metadata** (optional, boolean): Return metadata; defaults to false
  - **Sparse Vector** (optional, object): Sparse vector for hybrid search

**Validations**:
  - At least one namespace required
  - Vector cannot be empty

**Performance Considerations**:
  - Should execute queries in parallel when possible
  - Thread/concurrency pool size may affect performance
  - Connection pooling should be tuned for parallel requests

**Returns**: Response with combined and sorted results plus aggregated usage

#### Search Records (Integrated Inference)
**Purpose**: Search using raw input that will be automatically embedded

**Parameters**:
  - **Namespace** (required, string): Namespace to search
  - **Query** (required, object): Search query configuration
    - **Inputs**: Dictionary/object of field values to embed
    - **Top K**: Number of results to return
    - **Match Terms** (optional): For sparse indexes with text matching
      - Strategy: "all" (currently only supported value)
      - Terms: Array of terms to match
  - **Rerank** (optional, object): Rerank configuration
    - Model: Reranking model identifier
    - Rank Fields: Fields to use for reranking
    - Top N: Number of results after reranking
  - **Fields** (optional, array of strings): Fields to return in response
    - Defaults to ["*"] (all fields)

**Use Case**: For indexes created with integrated inference

**Behavior**: Service automatically generates embeddings from input fields

**Returns**: Response with matched records including scores and requested fields

#### Fetch Vectors by ID
**Purpose**: Retrieve specific vectors by their IDs

**Parameters**:
  - **IDs** (required, array of strings): Vector identifiers to fetch
  - **Namespace** (optional, string): Namespace to fetch from; defaults to empty string ""

**Returns**: Response with:
  - **Namespace**: Namespace fetched from
  - **Vectors**: Dictionary/map of ID to vector object (with values and metadata)
  - **Usage**: Optional resource usage information

#### Fetch Vectors by Metadata
**Purpose**: Retrieve vectors matching metadata criteria

**Parameters**:
  - **Filter** (required, object): Metadata filter expression
  - **Namespace** (optional, string): Namespace to fetch from; defaults to empty string ""
  - **Limit** (optional, integer): Maximum vectors to return; defaults to 100
  - **Pagination Token** (optional, string): Token for retrieving next page

**Returns**: Response with:
  - **Namespace**: Namespace fetched from
  - **Vectors**: Dictionary/map of matching vectors
  - **Usage**: Optional resource usage information
  - **Pagination**: Information for retrieving additional results

#### Delete Vectors
**Purpose**: Remove vectors from the index

**Delete Modes** (mutually exclusive):
  1. **By IDs** - Delete specific vectors
     - **IDs** (array of strings): Vector identifiers to delete
  2. **Delete All** - Delete all vectors in namespace
     - **Delete All** (boolean): Set to true
  3. **By Filter** - Delete vectors matching criteria
     - **Filter** (object): Metadata filter expression

**Common Parameters**:
  - **Namespace** (optional, string): Target namespace; defaults to empty string ""

**Validations**:
  - Only one deletion mode allowed per call
  - Exactly one of: ids, delete_all, or filter must be provided

**Behavior**: Does not error if IDs don't exist (idempotent)

**Warning**: Be careful with namespace - wrong namespace won't error but won't delete intended vectors

**Returns**: Acknowledgement (may be empty/void response)

#### Update Vectors
**Purpose**: Modify vector values or metadata

**Update Modes** (mutually exclusive):

1. **Single Vector Update** (by ID):
   - **ID** (required, string): Vector identifier
   - **Values** (optional, array of floats): New vector values
   - **Sparse Values** (optional, object): New sparse vector values
   - **Set Metadata** (optional, object): Metadata updates
     - Merges with existing metadata (doesn't replace entirely)

2. **Bulk Update** (by filter):
   - **Filter** (required, object): Metadata filter expression
   - **Set Metadata** (required, object): Metadata updates to apply to all matching vectors
   - **Dry Run** (optional, boolean): Preview without executing; defaults to false

**Common Parameters**:
  - **Namespace** (optional, string): Target namespace; defaults to empty string ""

**Validations**:
  - Exactly one of: id or filter must be provided (not both)

**Returns**: Response with:
  - **Matched Records**: Count of vectors updated (especially relevant for bulk updates)

#### Describe Index Statistics
**Purpose**: Get statistics about index contents

**Parameters**:
  - **Filter** (optional, object): Only return stats for vectors matching filter

**Returns**: Statistics object with:
  - **Total Vector Count**: Total number of vectors in index
  - **Dimension**: Vector dimensionality
  - **Namespaces**: Dictionary/map of namespace names to their statistics

#### List Vector IDs (Paginated)
**Purpose**: List vector IDs with pagination support

**Parameters**:
  - **Prefix** (optional, string): ID prefix to match; defaults to empty string (all IDs)
  - **Limit** (optional, integer): Maximum IDs per page
  - **Pagination Token** (optional, string): Token for retrieving next page
  - **Namespace** (optional, string): Namespace to list from; defaults to empty string ""

**Returns**: Response with:
  - **Vectors**: Array of vector IDs (strings)
  - **Namespace**: Namespace listed from
  - **Pagination**: Token and information for next page
  - **Usage**: Optional resource usage information

#### List Vector IDs (Iterator)
**Purpose**: Convenience method that auto-handles pagination

**Parameters**: Same as list_paginated()

**Returns**: Iterator/generator that yields batches of IDs
  - Should automatically fetch next page when exhausted
  - Simplifies pagination handling for client code

**Note**: This is a convenience wrapper - implementation details may vary by language

**Bulk Import Operations**:

#### Start Bulk Import
**Purpose**: Initiate import of vectors from cloud storage

**Parameters**:
  - **URI** (required, string): Storage URI (e.g., "s3://bucket/path/data.parquet")
  - **Integration ID** (optional, string): Storage integration identifier for authentication
  - **Error Mode** (optional, string): Behavior when errors occur
    - Values: "CONTINUE" or "ABORT"
    - Defaults to "CONTINUE"

**Returns**: Response with:
  - **Import ID**: Unique identifier for tracking the import operation

#### List Bulk Imports (Iterator)
**Purpose**: List import operations with automatic pagination

**Parameters**:
  - **Limit** (optional, integer): Maximum operations per page
  - **Pagination Token** (optional, string): Token for next page

**Returns**: Iterator/generator yielding import model objects

#### List Bulk Imports (Paginated)
**Purpose**: List import operations with manual pagination

**Parameters**:
  - **Limit** (optional, integer): Maximum operations per page
  - **Pagination Token** (optional, string): Token for next page

**Returns**: Response with:
  - **Data**: Array of import operation models
  - **Pagination**: Token for next page
  - **Usage**: Optional resource usage information

#### Describe Bulk Import
**Purpose**: Get status and details of a specific import

**Parameters**:
  - **Import ID** (required, string): Import operation identifier

**Returns**: Import model with:
  - ID, URI, status
  - Percent complete
  - Records imported count
  - Created at and finished at timestamps

#### Cancel Bulk Import
**Purpose**: Cancel an in-progress import operation

**Parameters**:
  - **Import ID** (required, string): Import operation identifier

**Returns**: Acknowledgement (may be empty/void response)

**Namespace Operations**:

#### Create Namespace
**Purpose**: Explicitly create a namespace

**Parameters**:
  - **Name** (required, string): Namespace name
  - **Schema** (optional, object): Metadata schema for the namespace

**Limitations**: Serverless indexes only

**Returns**: Namespace description with name and vector count

#### Describe Namespace
**Purpose**: Get statistics about a specific namespace

**Parameters**:
  - **Namespace** (required, string): Namespace name

**Returns**: Namespace description with:
  - **Name**: Namespace identifier
  - **Vector Count**: Number of vectors in namespace

#### Delete Namespace
**Purpose**: Delete a namespace and all its vectors

**Parameters**:
  - **Namespace** (required, string): Namespace name

**Returns**: Acknowledgement (may be empty/void response)

#### List Namespaces (Iterator)
**Purpose**: List all namespaces with automatic pagination

**Parameters**:
  - **Limit** (optional, integer): Maximum namespaces per page

**Returns**: Iterator/generator yielding namespace list responses

#### List Namespaces (Paginated)
**Purpose**: List all namespaces with manual pagination

**Parameters**:
  - **Limit** (optional, integer): Maximum namespaces per page
  - **Pagination Token** (optional, string): Token for next page

**Returns**: Response with namespace information and pagination details

### 2.2 Index Client (Asynchronous)

**Purpose**: Async/await variant for vector operations

**Initialization Configuration**: Same as synchronous index client

**Resource Cleanup**:
- ☑ Should support async context managers (async with)
- ☑ Should provide async close() method for cleanup

**Methods**: All same operations as synchronous client
- ☑ All methods should be awaitable
- ☑ Same parameters and validations
- ☑ **DataFrame upsert NOT supported** in async client (implementation complexity)
- ☑ Iterators should return async iterators (async for support)

**Batch Upsert Behavior**:
- Should use parallel async execution when batching
- Order of batch completion may vary
- Should still support progress tracking if available

### 2.3 Index Client (gRPC)

**Purpose**: High-performance variant using gRPC protocol

**Protocol**: gRPC instead of REST/HTTP

**Performance Benefits**:
- Binary protocol (faster than JSON serialization)
- HTTP/2 multiplexing (multiple concurrent requests on single connection)
- Lower latency for high-throughput scenarios
- Smaller payloads

**Configuration Options**:
- **Secure**: Use SSL/TLS encryption; defaults to true
- **Timeout**: Request timeout in seconds; defaults to 20
- **Connection Timeout**: Initial connection timeout; defaults to 1 second
- **Reuse Channel**: Reuse gRPC channels; defaults to true
- **Retry Configuration**: Custom retry policy with max attempts, backoff policy, retryable status codes
- **Channel Options**: gRPC-specific channel arguments
- **Additional Metadata**: gRPC metadata (similar to HTTP headers)

---

## 3. Admin API

**Purpose**: Administrative operations for projects, API keys, and organizations

### 3.1 Admin Client

**Authentication**: Requires service account credentials (client_id and client_secret)

**Initialization Configuration**:
- **Client ID** (optional, string): Service account client ID
  - Can be read from `PINECONE_CLIENT_ID` environment variable
- **Client Secret** (optional, string): Service account secret
  - Can be read from `PINECONE_CLIENT_SECRET` environment variable
- **Additional Headers** (optional, dictionary): Custom HTTP headers

**Accessible Resources**:
- ☑ **Project Management** - Operations for managing projects
- ☑ **API Key Management** - Operations for managing API keys
- ☑ **Organization Management** - Operations for managing organizations

### 3.2 Project Management

**Operations**:

#### Create Project
**Purpose**: Create a new project

**Returns**: Project model object

#### List Projects
**Purpose**: List all accessible projects

**Returns**: Collection of project model objects

#### Get Project
**Purpose**: Get details of a specific project

**Parameters**:
  - **Name** (required, string): Project name

**Returns**: Project model object

#### Delete Project
**Purpose**: Permanently delete a project

**Parameters**:
  - **ID** (required, string): Project identifier

**Returns**: Acknowledgement

### 3.3 API Key Management

**Operations**:

#### Create API Key
**Purpose**: Generate a new API key for a project

**Parameters**:
  - **Name** (required, string): API key name
  - **Project ID** (required, string): Target project
  - **Description** (optional, string): Human-readable description
  - **Roles** (required, array): Array of role identifiers

**Returns**: API key model with key value (only returned once at creation)

#### List API Keys
**Purpose**: List API keys for a project

**Parameters**:
  - **Project ID** (required, string): Project identifier

**Returns**: Collection of API key model objects (without key values)

#### Delete API Key
**Purpose**: Revoke an API key

**Parameters**:
  - **API Key ID** (required, string): API key identifier

**Returns**: Acknowledgement

### 3.4 Organization Management

**Operations**:

#### List Organizations
**Purpose**: List all accessible organizations

**Returns**: Collection of organization model objects

#### Get Organization
**Purpose**: Get details of a specific organization

**Parameters**:
  - **ID** (required, string): Organization identifier

**Returns**: Organization model object

#### Update Organization
**Purpose**: Modify organization settings

**Parameters**:
  - **Organization ID** (required, string): Organization identifier
  - **Name** (optional, string): New organization name

**Returns**: Updated organization model object

---

## 4. Inference API

**Purpose**: Embedding generation and document reranking operations

### 4.1 Inference Client (Synchronous)

**Initialization Configuration**:
- Internal configuration objects required
- **Thread Pool Size** (optional, integer): For parallel operations; defaults to 1

**Accessible Resources**:
- ☑ **Model Operations** - List and describe available models

**Operations**:

#### Generate Embeddings
**Purpose**: Convert text inputs into vector embeddings

**Parameters**:
  - **Model** (required, string): Embedding model identifier or enum
  - **Inputs** (required, string or array): Input text(s) to embed
    - Can be single string
    - Can be array of strings
    - Can be array of structured objects/dictionaries
  - **Parameters** (optional, object): Model-specific parameters
    - input_type: Purpose of embeddings (e.g., "query", "passage")
    - truncate: Truncation strategy
    - Other model-specific options

**Returns**: Embeddings response with:
  - **Model**: Model identifier used
  - **Data**: Array of embedding objects with vectors and indices
  - **Usage**: Token usage information

#### Rerank Documents
**Purpose**: Reorder documents by relevance to a query

**Parameters**:
  - **Model** (required, string): Reranking model identifier or enum
  - **Query** (required, string): Search query
  - **Documents** (required, array): Documents to rerank
    - Can be array of strings
    - Can be array of structured objects/dictionaries
  - **Rank Fields** (optional, array of strings): Fields to rank on; defaults to ["text"]
  - **Return Documents** (optional, boolean): Include document content in response; defaults to true
  - **Top N** (optional, integer): Number of results to return; defaults to all documents
  - **Parameters** (optional, object): Additional model-specific parameters

**Returns**: Reranking response with:
  - **Model**: Model identifier used
  - **Data**: Array of results with index, score, and optional document content
  - **Usage**: Token usage information

#### List Models
**Purpose**: List available embedding and reranking models

**Parameters**:
  - **Type** (optional, string): Filter by model type
    - Values: "embed" or "rerank"
  - **Vector Type** (optional, string): Filter by vector type
    - Values: "dense" or "sparse"

**Returns**: Collection of model information objects

#### Get Model Details
**Purpose**: Get detailed information about a specific model

**Parameters**:
  - **Model Name** (required, string): Model identifier

**Returns**: Model information with:
  - Model details and capabilities
  - Supported parameters
  - Maximum sequence length
  - Other model-specific metadata

### 4.2 Inference Client (Asynchronous)

**Purpose**: Async/await variant of inference client

**Methods**: All same operations as synchronous client
- ☑ Generate Embeddings (async)
- ☑ Rerank Documents (async)
- ☑ List Models (async)
- ☑ Get Model Details (async)

---

## 5. Error Handling

**Purpose**: Structured exception hierarchy for consistent error handling

### Exception Hierarchy

SDKs should implement a hierarchical exception structure:

```
PineconeException (base)
├── Type Validation Errors
├── Value Validation Errors
├── API Errors
│   ├── NotFoundException (404)
│   ├── UnauthorizedException (401)
│   ├── ForbiddenException (403)
│   └── ServiceException (5xx)
├── Protocol Errors
└── Configuration Errors
```

### Exception Categories

#### Base Exception
- **Purpose**: Base class for all SDK-specific exceptions
- **Use**: Allows catch-all error handling for SDK errors

#### Type Validation Errors
- **When Raised**: Type validation failures (wrong data type provided)
- **Should Include**:
  - Path to the invalid field
  - Expected types
  - Actual type received

#### Value Validation Errors
- **When Raised**: Value validation failures (invalid value for correct type)
- **Should Include**:
  - Path to the invalid field
  - Validation rule violated
  - Actual value received

#### API Errors (Base)
- **When Raised**: General API request failures
- **Should Include**:
  - HTTP status code
  - Reason phrase
  - Response body
  - Response headers

#### Not Found Error (404)
- **Inherits**: API Error
- **When Raised**: Requested resource doesn't exist
- **Examples**: Index not found, vector ID not found

#### Unauthorized Error (401)
- **Inherits**: API Error
- **When Raised**: Authentication failure
- **Examples**: Invalid API key, missing credentials

#### Forbidden Error (403)
- **Inherits**: API Error
- **When Raised**: Authorization failure (authenticated but not permitted)
- **Examples**: Insufficient permissions, quota exceeded

#### Service Error (5xx)
- **Inherits**: API Error
- **When Raised**: Server-side errors
- **Examples**: Internal server error, service unavailable

#### Protocol Errors
- **When Raised**: Unexpected network or protocol errors during request/response
- **Examples**: Connection reset, timeout, malformed response

#### Configuration Errors
- **When Raised**: Invalid SDK configuration
- **Examples**: Missing required config, invalid parameter combinations

### Error Context

All errors should provide:
- **Clear error messages** in default string representation
- **Programmatic access** to error details (status codes, response bodies)
- **Stack traces** or equivalent debugging information
- **Request identifiers** when available from API

### Error Conversion

SDKs should wrap/convert underlying errors:
- **gRPC errors** → Convert to appropriate Pinecone exception types
- **HTTP client errors** → Convert to Pinecone protocol or API exceptions
- **Validation errors** → Convert to type/value validation exceptions

---

## 6. Retry Mechanisms

**Purpose**: Automatic retry of failed requests for improved reliability

### 6.1 REST Retry Configuration (Synchronous)

**Default Configuration**:
- **Total Retry Attempts**: 5
- **Backoff Factor**: 0.1 seconds base
- **Retryable Status Codes**: 500, 502, 503, 504
- **Retryable Methods**: All HTTP methods

**Backoff Strategy**:
- **Algorithm**: Exponential backoff with jitter
- **Jitter Amount**: Random value (approximately 0-0.25 seconds)
- **Purpose**: Prevents thundering herd problem when many clients retry simultaneously
- **Formula**: `base_backoff * (2 ^ attempt) + jitter`

### 6.2 REST Retry Configuration (Asynchronous)

**Default Configuration**:
- **Retry Attempts**: 5
- **Initial Timeout**: 0.1 seconds
- **Maximum Timeout**: 3.0 seconds
- **Retryable Status Codes**: 500, 502, 503, 504
- **Retryable Methods**: All HTTP methods
- **Retryable Exceptions**: Client errors, disconnections

**Backoff Strategy**:
- **Algorithm**: Exponential backoff with jitter
- **Jitter Amount**: Random value (0-0.1 seconds)
- **Formula**: `min(initial_timeout * (2 ^ (attempt - 1)) + jitter, max_timeout)`

### 6.3 gRPC Retry Configuration (Interceptor)

**Default Configuration**:
- **Maximum Attempts**: 4
- **Initial Backoff**: 100 milliseconds
- **Maximum Backoff**: 1600 milliseconds
- **Backoff Multiplier**: 2
- **Retryable Status Codes**: UNAVAILABLE

**Backoff Strategy**:
- **Algorithm**: Exponential backoff
- **Formula**: `min(initial_backoff * (multiplier ^ attempt), max_backoff)`
- **Time Unit**: Milliseconds

**Retryable Conditions**:
- Only specific gRPC status codes are retried
- Default: Only UNAVAILABLE status
- Can be configured for additional status codes

**Supported Call Types**:
- ☑ Unary-unary (single request, single response)
- ☑ Unary-stream (single request, streaming response)
- ☑ Stream-unary (streaming request, single response)
- ☑ Stream-stream (bidirectional streaming)

### 6.4 gRPC Native Retry Configuration

**Purpose**: Leverage gRPC's built-in retry mechanism

**Configuration via Service Config**:
- **Per-Method Policies**: Different retry policies for different operations
- **Upsert Operations**:
  - Maximum attempts: 5
  - Initial backoff: 0.1 seconds
  - Maximum backoff: 1 second
  - Backoff multiplier: 2
  - Retryable codes: UNAVAILABLE
- **General Vector Operations**:
  - Same configuration as upsert

**Benefits**:
- Native gRPC retry support
- More efficient than application-level retries
- Applied automatically at channel level

### General Retry Principles

**When to Retry**:
- ☑ Server errors (5xx status codes)
- ☑ Service unavailable errors
- ☑ Network timeout errors
- ☑ Connection failures

**When NOT to Retry**:
- ☑ Client errors (4xx status codes) - typically unrecoverable
- ☑ Authentication errors (401)
- ☑ Authorization errors (403)
- ☑ Not found errors (404)
- ☑ Validation errors (400)

**Configurability**:
SDKs should allow:
- ☑ Disabling retries entirely
- ☑ Configuring max attempts
- ☑ Configuring backoff parameters
- ☑ Configuring retryable status codes/conditions

---

## 7. Caching

**Purpose**: Minimize redundant API calls and optimize connection reuse

### 7.1 Index Host Caching

**Purpose**:
- Cache index host URLs to avoid repeated describe_index() calls
- Share cache across client instances in same process

**Implementation Strategy**:
- Should use singleton pattern or equivalent
- Cache key: Combination of API key and index name
- Scope: Process-wide (survives across multiple client instances)

**Operations**:
- ☑ **Get Host**: Retrieve cached host or fetch via describe_index() if not cached
- ☑ **Set Host**: Store host URL in cache
- ☑ **Delete Host**: Remove from cache (e.g., after index deletion)
- ☑ **Check Existence**: Verify if host is cached

**Behavior**:
- First access to an index triggers describe_index() API call
- Subsequent accesses use cached host (no API call)
- Cache persists for lifetime of process
- Reduces control plane API load

### 7.2 Connection Pooling (REST Synchronous)

**Purpose**: Reuse HTTP connections for better performance

**Configuration**:
- **Connection Pool Max Size** (optional): Maximum connections in pool
- Default pool size determined by underlying HTTP library

**Behavior**:
- Connections reused across requests
- Lazy initialization: connections created on first request
- Reduces TCP handshake overhead

**Best Practices**:
- ☑ Reuse client instances across requests
- ☑ Don't create new client per request (causes connection churn)
- ☑ Configure pool size based on expected concurrency

### 7.3 Connection Pooling (REST Asynchronous)

**Purpose**: Reuse HTTP connections in async environment

**Configuration**:
- **Connection Pool Max Size** (optional): Maximum concurrent connections
- TCP connector with configurable SSL settings

**Behavior**:
- Automatic connection pooling via async HTTP library
- Connections shared across concurrent requests

**Resource Management**:
- ☑ Use async context managers to ensure cleanup
- ☑ Call close() method explicitly if not using context managers
- ☑ Properly cleanup to avoid resource leaks

### 7.4 Connection Pooling (gRPC)

**Purpose**: Reuse gRPC channels for better performance

**Configuration**:
- **Reuse Channel** (boolean): Enable channel reuse; defaults to true
- **Connection Timeout**: Initial connection timeout

**Behavior**:
- Channels reused when enabled and channel is ready
- Health checks determine channel readiness
- Auto-reconnect if channel fails
- Explicit close() releases channel resources

**Benefits**:
- HTTP/2 multiplexing allows multiple concurrent requests per channel
- Reduced connection overhead
- Better resource utilization

---

## 8. Configuration & Environment

**Purpose**: Flexible configuration through environment variables and constructor parameters

### 8.1 Environment Variables

SDKs should support reading configuration from environment:

- ☑ **PINECONE_API_KEY** - Default API key for client authentication
- ☑ **PINECONE_CLIENT_ID** - Service account client ID for admin operations
- ☑ **PINECONE_CLIENT_SECRET** - Service account secret for admin operations
- ☑ **PINECONE_DEBUG_CURL** - Enable debug logging with curl commands
  - ⚠️ **Warning**: Exposes API keys in logs - only for debugging
- ☑ **PINECONE_ADDITIONAL_HEADERS** - Additional headers (JSON/structured format)

**Precedence**: Constructor parameters should override environment variables

### 8.2 Client Configuration Parameters

**Common Configuration** (all clients):
- ☑ **API Key** (string): Authentication credential; overrides environment variable
- ☑ **Host** (string): Control plane host URL; defaults to api.pinecone.io
- ☑ **Additional Headers** (dictionary): Custom HTTP headers for all requests
- ☑ **SSL CA Certificates** (string): Path to CA certificate bundle
- ☑ **SSL Verification** (boolean): Enable/disable SSL certificate verification; defaults to enabled

**REST Client Specific**:
- ☑ **Proxy URL** (string): HTTP/HTTPS proxy server URL
- ☑ **Proxy Headers** (dictionary): Headers for proxy authentication
  - Note: May not be supported in async clients (implementation complexity)
- ☑ **Thread Pool Size** (integer): Size of thread pool for parallel operations
  - Synchronous clients only
  - Typically defaults to 5 × CPU count

**Index Client Specific**:
- ☑ **Connection Pool Max Size** (integer): Maximum connections in pool

**gRPC Client Specific**:
- ☑ **gRPC Configuration Object**: Structured configuration with:
  - Secure mode (SSL/TLS)
  - Timeouts
  - Channel options
  - Retry configuration

### 8.3 Proxy Support

**Capabilities**:
- ☑ **HTTP/HTTPS Proxy**: Route requests through proxy server
- ☑ **Proxy Authentication**: Support for authenticated proxies
- ☑ **Custom SSL Certificates**: Trust custom CAs used by proxy

**Use Cases**:
- Corporate networks with mandatory proxies
- Development environments with traffic inspection
- Security scanning and monitoring

**Configuration Example** (conceptual):
```
Client Configuration:
  api_key: <your-api-key>
  proxy_url: https://proxy.company.com:8080
  proxy_auth: <username:password>
  ssl_ca_certs: /path/to/company-ca-bundle.pem
```

### 8.4 SSL/TLS Configuration

**Options**:
- ☑ **Custom CA Bundle**: Path to trusted certificate bundle
  - Use case: Corporate proxies with custom CAs
  - Default: System's default CA bundle
- ☑ **SSL Verification Toggle**: Enable/disable certificate verification
  - Default: Enabled (recommended)
  - Disable only for: Local development, testing environments
  - ⚠️ **Warning**: Never disable in production

**Certificate Verification**:
- Validates server certificates against trusted CAs
- Prevents man-in-the-middle attacks
- Should be enabled by default for security

---

## 9. Performance Features

**Purpose**: Optimize throughput and minimize latency for vector operations

### 9.1 Automatic Batching

**Purpose**: Chunk large operations into manageable batches

**Upsert Batching**:
- **Batch Size Parameter**: Automatically split large vector lists into chunks
  - Example: 10,000 vectors with batch_size=100 → 100 requests of 100 vectors each
- **Progress Tracking**: Display progress indicators when batching
- **Aggregated Results**: Combine responses from all batches
- **Error Handling**: Continue on error vs abort on first error

**Benefits**:
- Prevents request payload size limits
- Provides progress feedback for long operations
- Manages memory efficiently

**DataFrame/Table Batching**:
- Similar batching strategy for tabular data sources
- Default batch sizes typically 500-1000 vectors
- Optimized for memory efficiency

### 9.2 Parallel Request Execution

**Synchronous Parallel Execution**:
- **Thread Pool**: Configurable thread pool for parallel requests
- **Typical Configuration**: 5 × CPU count threads
- **Use Case**: Execute multiple operations simultaneously
- **Future/Promise Pattern**: Return immediately, retrieve results later
- **Example Use Case**: Parallel upserts to different namespaces

**Asynchronous Parallel Execution**:
- **Native Async/Await**: Use language's async primitives
- **Concurrent Operations**: Execute multiple awaitable operations simultaneously
- **Resource Efficiency**: Better resource utilization than threads
- **Example Patterns**: gather(), as_completed(), or equivalent

**Configuration Considerations**:
- Thread pool size affects concurrent request count
- Connection pool size should accommodate parallel requests
- Balance between throughput and resource consumption

### 9.3 Multi-Namespace Parallel Querying

**Purpose**: Query multiple namespaces simultaneously and merge results

**Behavior**:
- Execute queries to different namespaces in parallel
- Merge and sort results using specified metric
- Aggregate usage statistics across all queries
- Return unified result set

**Performance Tuning**:
- Thread/concurrency pool size affects parallelism
- Connection pooling should support concurrent requests
- Typical configuration: 5 × CPU count for synchronous clients

**Use Cases**:
- Multi-tenant applications with namespace-per-tenant
- Searching across multiple data partitions
- A/B testing with separate namespaces

### 9.4 gRPC Protocol

**Performance Advantages**:
- **Binary Serialization**: Faster than JSON/text-based protocols
- **HTTP/2 Multiplexing**: Multiple concurrent requests on single connection
- **Lower Latency**: Reduced overhead compared to REST/HTTP
- **Smaller Payloads**: Binary format is more compact
- **Streaming Support**: Efficient for large data transfers

**When to Use**:
- ☑ High-throughput applications (thousands of requests/second)
- ☑ Latency-sensitive workloads (milliseconds matter)
- ☑ Large batch operations (many vectors per request)
- ☑ Streaming scenarios

**Trade-offs**:
- May require additional dependencies
- Less ubiquitous than REST
- Debugging may be more complex (binary protocol)

### 9.5 Progress Tracking

**Purpose**: Provide feedback during long-running operations

**Capabilities**:
- ☑ **Progress Indicators**: Display progress bars or percentage complete
- ☑ **Count Tracking**: Show vectors processed / total vectors
- ☑ **Rate Information**: Display processing rate (vectors/second)
- ☑ **Time Estimates**: Estimate remaining time

**Configuration**:
- **Show Progress Parameter**: Enable/disable progress display
- **Auto-Detection**: Detect terminal capabilities
- **Library Integration**: Use standard progress libraries when available

**User Experience**:
- Provides visibility into long operations
- Allows monitoring of batch jobs
- Helps diagnose performance issues

---

## 10. Validation & Edge Cases

**Purpose**: Input validation and handling of edge cases

### 10.1 Vector Format Validation

**Supported Formats**:
SDKs should accept flexible input formats:
1. **Structured Objects**: Objects with id, values, sparse_values, metadata fields
2. **2-Element Tuples/Arrays**: (id, values)
3. **3-Element Tuples/Arrays**: (id, values, metadata)
4. **Dictionaries/Maps**: {"id": string, "values": array, ...}

**Validation Rules**:
- ☑ **ID Required**: String identifier must be provided
- ☑ **Values or Sparse Values Required**: At least one must be present
- ☑ **Dense Values Type**: Must be array of floating-point numbers
- ☑ **Sparse Values Format**: Must contain indices (integers) and values (floats) of same length
- ☑ **Metadata Type**: Must be dictionary/object (optional)

**Validation Errors** (should raise specific exceptions):
- **Missing Required Keys**: ID not provided, or neither values nor sparse_values
- **Excess Keys**: Unknown fields in vector object
- **Invalid Tuple Length**: Tuple/array length not 2 or 3
- **Invalid Sparse Values Type**: Sparse values not in correct format
- **Missing Sparse Components**: Sparse values missing indices or values
- **Type Mismatch**: Expected dictionary/object but received different type

### 10.2 Sparse Vector Validation

**Required Format**:
Sparse vectors must have two components:
- **Indices**: Array of non-negative integers (dimension indices)
- **Values**: Array of floating-point numbers

**Example Representations**:
- Object: `SparseValues(indices=[1, 2, 3], values=[0.1, 0.2, 0.3])`
- Dictionary: `{"indices": [1, 2, 3], "values": [0.1, 0.2, 0.3]}`

**Validation Rules**:
- ☑ **Indices Type**: Must be array of integers
- ☑ **Values Type**: Must be array of floats
- ☑ **Length Match**: Both arrays must have same length
- ☑ **Non-Negative Indices**: Indices should be ≥ 0

### 10.3 Metadata Filter Validation

**Supported Comparison Operators**:
- ☑ **$eq** - Equality
- ☑ **$ne** - Inequality
- ☑ **$gt** - Greater than
- ☑ **$gte** - Greater than or equal
- ☑ **$lt** - Less than
- ☑ **$lte** - Less than or equal
- ☑ **$in** - Value in set
- ☑ **$nin** - Value not in set
- ☑ **$exists** - Field exists

**Logical Operators**:
- ☑ **$and** - Logical AND (all conditions must be true)
- ☑ **$or** - Logical OR (at least one condition must be true)

**Example Filter Expression**:
```
{
  "$and": [
    {"genre": {"$eq": "drama"}},
    {"year": {"$gte": 2000}},
    {"rating": {"$in": [8, 9, 10]}}
  ]
}
```

### 10.4 Namespace Validation

**Rules**:
- ☑ **Empty String Valid**: "" (empty string) is the default namespace
- ☑ **Max Length**: Varies by deployment type (check documentation)
- ☑ **Case Sensitive**: "Namespace" ≠ "namespace"
- ☑ **No Special Validation**: Most strings accepted

### 10.5 Index Name Validation

**Character Rules**:
- ☑ **Allowed Characters**: Lowercase letters (a-z), numbers (0-9), hyphens (-)
- ☑ **Cannot Start with Hyphen**: First character must be letter or number
- ☑ **Cannot End with Hyphen**: Last character must be letter or number
- ☑ **Maximum Length**: 45 characters

**Uniqueness**:
- ☑ **Project Scope**: Must be unique within project
- ☑ **Immutable**: Cannot be changed after creation (must delete and recreate)

### 10.6 Dimension Validation

**Rules**:
- ☑ **Required for Dense Vectors**: Must specify dimension when creating dense vector index
- ☑ **Must Match Model**: For integrated indexes, dimension must match model output
- ☑ **Immutable**: Cannot be changed after index creation
- ☑ **Not Used for Sparse**: Sparse vector indexes don't specify dimension

**Consequences**:
- Upserting vectors with wrong dimension will fail
- Always verify dimension before upserting

### 10.7 Metric Validation

**Supported Values**:
- ☑ **cosine** - Cosine similarity
- ☑ **dotproduct** - Dot product similarity
- ☑ **euclidean** - Euclidean distance

**Rules**:
- ☑ **Immutable**: Cannot be changed after index creation
- ☑ **Affects Results**: Different metrics produce different similarity scores
- ☑ **Model Compatibility**: For integrated indexes, must be supported by embedding model

### 10.8 Mutually Exclusive Parameters

**Delete Operation Modes** (exactly one required):
- **By IDs**: Delete specific vectors by ID
- **Delete All**: Delete all vectors in namespace
- **By Filter**: Delete vectors matching metadata criteria
- ⚠️ **Validation**: Should raise error if multiple modes specified

**Query Vector Source** (exactly one required):
- **Vector**: Provide query vector directly
- **ID**: Use existing vector as query
- ⚠️ **Validation**: Should raise error if both or neither specified

**Update Operation Modes** (exactly one required):
- **By ID**: Update single vector by identifier
- **By Filter**: Bulk update vectors matching criteria
- ⚠️ **Validation**: Should raise error if both specified

### 10.9 Top-K Validation

**Rules**:
- ☑ **Minimum Value**: Must be ≥ 1
- ☑ **Type**: Must be integer (not float or string)
- ☑ **Practical Limits**: Very large values may impact performance

### 10.10 Timeout Validation

**Special Values and Meanings**:
- **null/None**: Wait indefinitely until operation completes
- **-1**: Return immediately without waiting
- **≥ 0**: Wait up to N seconds, then timeout

**Behavior**:
- Used for index creation, deletion, and other long-running operations
- Allows polling vs. blocking vs. fire-and-forget

### 10.11 Important Edge Cases

**Empty Namespace**:
- ☑ **Default is Empty**: Default namespace is empty string ""
- ☑ **No Error on Wrong Namespace**: Operations on non-existent namespace may succeed without error
- ⚠️ **Be Careful**: Always verify namespace when debugging

**Non-Existent Vector IDs**:
- ☑ **Delete is Idempotent**: Deleting non-existent IDs succeeds (no error)
- ☑ **Fetch Returns Empty**: Fetching non-existent IDs returns empty results
- ☑ **Update May Fail**: Updating non-existent ID by ID may error

**Dimension Mismatch**:
- ☑ **Validation on Upsert**: Upserting vectors with wrong dimension raises error
- ☑ **Check Before Upserting**: Always verify vector dimensions match index

**Async Request with Batching**:
- ☑ **Incompatible Modes**: Cannot combine async_req with batch_size
- ☑ **Should Raise Error**: Clear error message explaining incompatibility

**Empty Query Vector**:
- ☑ **Validation Required**: Empty vector array should raise error
- ☑ **Minimum Requirement**: At least vector or sparse_vector must be provided

---

## Appendix A: Data Types

**Purpose**: Standard data structures used across SDK operations

### Vector Types

**Vector Object**:
A vector should be represented as an object/structure with:
- **id** (string, required): Unique identifier
- **values** (array of floats, optional): Dense vector values
- **sparse_values** (sparse values object, optional): Sparse vector representation
- **metadata** (dictionary/object, optional): Associated metadata

At least one of values or sparse_values must be provided.

**Sparse Values Object**:
Sparse vector representation with:
- **indices** (array of integers, required): Non-zero dimension indices
- **values** (array of floats, required): Corresponding values

Both arrays must have the same length.

### Response Types

**Upsert Response**:
Result of upsert operation containing:
- **upserted_count** (integer): Number of vectors successfully upserted
- **response_info** (optional): Request metadata (timing, request ID, etc.)

**Query Response**:
Result of query operation containing:
- **matches** (array of scored vectors): Results ordered by similarity
  - Each match includes: id, score, optional values, optional metadata
- **namespace** (string): Namespace that was queried
- **usage** (optional object): Resource usage information (read units, etc.)
- **response_info** (optional): Request metadata

**Fetch Response**:
Result of fetch operation containing:
- **namespace** (string): Namespace fetched from
- **vectors** (dictionary/map): Vector ID to vector object mapping
- **usage** (optional object): Resource usage information
- **response_info** (optional): Request metadata

**Update Response**:
Result of update operation containing:
- **matched_records** (integer, optional): Number of vectors matched/updated
- **response_info** (optional): Request metadata

### Model Types

**Index Model**:
Represents an index with fields:
- name, dimension, metric, host, status, spec, vector_type, deletion_protection, tags

**Serverless Spec**:
Configuration for serverless index:
- cloud (provider), region, read_capacity mode, optional schema

**Pod Spec**:
Configuration for pod-based index:
- environment, pod_type, pod count, replicas, shards, metadata_config, optional source_collection

---

## Appendix B: Async/Await Support Summary

**Purpose**: Guidelines for async/await implementations in SDKs

### Client Coverage

**Should Have Synchronous Versions**:
- ☑ Main/Primary Client
- ☑ Index Client
- ☑ Inference Client
- ☑ Admin Client (may not have async version)

**Should Have Asynchronous Versions**:
- ☑ Main/Primary Client (async variant)
- ☑ Index Client (async variant)
- ☑ Inference Client (async variant)
- ☑ Admin Client (optional - may not be needed)

### Resource Management Patterns

**Synchronous**:
Should support language-equivalent context managers/resource management:
- Automatic resource cleanup on scope exit
- Explicit close() method available

**Asynchronous**:
Should support async context managers/resource management:
- Async cleanup on scope exit
- Async close() method available
- Proper handling of async resources (sessions, connections)

### Async Implementation Guidelines

**Method Signatures**:
- ☑ All methods should be awaitable
- ☑ Same parameters and validations as sync version
- ☑ Same return types (wrapped in async result)

**Not Supported in Async**:
- ☑ DataFrame/table upsert operations (complexity and library compatibility)
- May be added in future if libraries provide async support

**Iterators**:
- ☑ Should return async iterators
- ☑ Support async for loops
- ☑ Same pagination behavior as sync

**Concurrency**:
- ☑ Multiple operations can be awaited concurrently
- ☑ Better resource utilization than thread-based parallelism
- ☑ Native support for cancellation and timeouts

---

## Appendix C: SDK Feature Evaluation Checklist

**Purpose**: Use this checklist to evaluate feature parity across Pinecone SDKs

### Control Plane Operations (Index Management)
- [ ] Create index with serverless configuration
- [ ] Create index with pod configuration
- [ ] Create index with BYOC (Bring Your Own Cloud) configuration
- [ ] Create index with integrated inference (for model)
- [ ] Create index from backup (restore)
- [ ] Delete index
- [ ] List all indexes
- [ ] Describe specific index (get details)
- [ ] Check if index exists (convenience method)
- [ ] Configure index (scaling, deletion protection, tags)
- [ ] Collection management (create, list, describe, delete)
- [ ] Backup management (create, list, describe, delete)
- [ ] Restore job tracking (list, describe status)

### Data Plane Operations (Vector Operations)
- [ ] Upsert vectors (insert/update)
- [ ] Upsert with multiple vector formats (objects, tuples, dictionaries)
- [ ] Upsert from tabular data (dataframe/table)
- [ ] Upsert records with auto-embedding (integrated inference)
- [ ] Query by vector
- [ ] Query by vector ID
- [ ] Query with hybrid search (dense + sparse)
- [ ] Query multiple namespaces in parallel
- [ ] Search with auto-embedding (integrated inference)
- [ ] Fetch vectors by ID
- [ ] Fetch vectors by metadata filter
- [ ] Delete by vector IDs
- [ ] Delete all vectors in namespace
- [ ] Delete by metadata filter
- [ ] Update single vector by ID
- [ ] Bulk update by metadata filter
- [ ] Dry run for updates (preview without executing)
- [ ] Describe index statistics
- [ ] List vector IDs (paginated)
- [ ] List vector IDs (iterator/generator for auto-pagination)
- [ ] Bulk import from storage (start, monitor, cancel)
- [ ] Namespace management (create, describe, delete, list)

### Inference Operations (Embedding & Reranking)
- [ ] Generate embeddings for single input
- [ ] Generate embeddings for batch of inputs
- [ ] Rerank documents by relevance
- [ ] List available models
- [ ] Filter models by type (embed/rerank)
- [ ] Filter models by vector type (dense/sparse)
- [ ] Get detailed model information

### Admin Operations
- [ ] Create project
- [ ] List projects
- [ ] Get project details
- [ ] Delete project
- [ ] Create API key
- [ ] List API keys
- [ ] Delete API key
- [ ] List organizations
- [ ] Get organization details
- [ ] Update organization

### Error Handling & Exceptions
- [ ] Hierarchical exception types
- [ ] Base exception for all SDK errors
- [ ] Specific exceptions for HTTP status codes (404, 401, 403, 5xx)
- [ ] Type validation errors with details
- [ ] Value validation errors with details
- [ ] Protocol/network errors
- [ ] Configuration errors
- [ ] Detailed error messages
- [ ] Programmatic access to error details
- [ ] Path-to-field for validation errors

### Retry Mechanisms & Reliability
- [ ] Automatic retries for transient failures
- [ ] Exponential backoff for retries
- [ ] Jitter to prevent thundering herd
- [ ] Configurable retry attempts
- [ ] Configurable retryable status codes/conditions
- [ ] REST/HTTP retry support
- [ ] gRPC retry support (if gRPC client exists)
- [ ] Per-operation retry configuration

### Caching & Performance Optimization
- [ ] Index host URL caching
- [ ] Connection pooling (REST/HTTP)
- [ ] Connection pooling (gRPC)
- [ ] Lazy initialization of resources
- [ ] Shared cache across client instances
- [ ] Configurable pool sizes

### Configuration & Authentication
- [ ] Environment variable support for credentials
- [ ] Constructor parameter override of environment
- [ ] API key authentication
- [ ] Service account authentication (for admin operations)
- [ ] Custom control plane host
- [ ] Custom HTTP headers
- [ ] Proxy server support
- [ ] Proxy authentication
- [ ] Custom SSL certificates
- [ ] SSL verification toggle (for development)
- [ ] Thread/concurrency pool configuration

### Performance Features
- [ ] Automatic batching for large operations
- [ ] Configurable batch sizes
- [ ] Progress tracking for long operations
- [ ] Parallel request execution (synchronous)
- [ ] Parallel request execution (asynchronous)
- [ ] Multi-namespace parallel querying
- [ ] gRPC protocol option (high performance)
- [ ] Streaming support (if applicable)

### Developer Experience (UX)
- [ ] Synchronous client API
- [ ] Asynchronous client API (async/await)
- [ ] Context managers for resource cleanup
- [ ] Type hints/annotations (for statically typed languages)
- [ ] Comprehensive API documentation
- [ ] Multiple vector input formats
- [ ] Convenience methods (e.g., has_index, check existence)
- [ ] Iterator/generator-based pagination
- [ ] Enum/constant support for fixed values
- [ ] Clear, idiomatic API design
- [ ] Examples and code samples

### Input Validation
- [ ] Vector format validation
- [ ] Multiple vector format support
- [ ] Sparse vector validation
- [ ] Metadata filter validation
- [ ] Namespace validation
- [ ] Index name validation
- [ ] Dimension validation
- [ ] Metric validation
- [ ] Mutually exclusive parameter validation
- [ ] Required parameter validation
- [ ] Top-K value validation
- [ ] Timeout value validation
- [ ] Clear validation error messages

---

## Document Change Log

- **2026-01-16**: Initial comprehensive documentation created (Python-specific)
- **2026-01-16**: Refactored to be language-agnostic for cross-SDK comparison
