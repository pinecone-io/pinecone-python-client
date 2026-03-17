# Admin API Key Operations

Documents API key management operations available through the Admin client: list, create, fetch, update, and delete API keys for your project.

**Source:** `pinecone/admin/resources/api_key.py:10-332`, `pinecone/admin/admin.py:172-288`

## Overview

The `ApiKeyResource` class provides a complete API key management interface. Access this class through the `Admin` client's `api_key` or `api_keys` property:

```python
from pinecone import Admin

admin = Admin()
api_keys = admin.api_key.list(project_id='my-project-id')
```

All methods require keyword arguments. The class automatically handles requests and responses using the underlying OpenAPI client.

## Methods

### list

List all API keys for a project. The API key secret value is not returned by this method.

**Source:** `pinecone/admin/resources/api_key.py:33-72`

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `project_id` | `str` | Yes | — | The project ID to list API keys for. Find this using `admin.project.get()` or `admin.project.list()`. |

| Return | Type | Description |
|--------|------|-------------|
| `data` | `list[APIKey]` | A list of APIKey objects for the project. Each contains `id`, `name`, `project_id`, and `roles`. |

| Raises | Condition |
|--------|-----------|
| `UnauthorizedException` | Raised when authentication credentials lack sufficient permissions. |

**Example:**

```python
from pinecone import Admin

admin = Admin()
api_keys_response = admin.api_key.list(project_id='my-project-id')
for api_key in api_keys_response.data:
    print(f"ID: {api_key.id}")
    print(f"Name: {api_key.name}")
    print(f"Roles: {api_key.roles}")
```

### create

Create a new API key for a project. The API key secret value is returned only in the create response.

**Source:** `pinecone/admin/resources/api_key.py:195-256`

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `project_id` | `str` | Yes | — | The project ID to create the API key for. Find this using `admin.project.get()` or `admin.project.list()`. |
| `name` | `str` | Yes | — | The name of the API key. Must be 1–80 characters. |
| `roles` | `list[str]` | No | `None` | An optional list of roles to assign to the API key. Available roles are: `ProjectEditor`, `ProjectViewer`, `ControlPlaneEditor`, `ControlPlaneViewer`, `DataPlaneEditor`, `DataPlaneViewer`. If omitted, no roles are assigned. |

| Return | Type | Description |
|--------|------|-------------|
| `key` | `APIKey` | The created API key object with `id`, `name`, `project_id`, and `roles`. |
| `value` | `str` | The secret value of the API key. **This value is only returned here at creation time and cannot be retrieved later.** Store it securely immediately after creation. |

| Raises | Condition |
|--------|-----------|
| `UnauthorizedException` | Raised when authentication credentials lack sufficient permissions. |

**Example:**

```python
from pinecone import Admin

admin = Admin()
response = admin.api_key.create(
    project_id='my-project-id',
    name='ci-testing-key',
    roles=['ProjectEditor', 'DataPlaneEditor']
)

api_key = response.key
secret_value = response.value

print(f"Created API key: {api_key.id}")
print(f"Secret (save this!): {secret_value}")
```

### fetch

Fetch (retrieve) an API key by its ID. This is the primary method for describing an API key. The secret value is not returned.

**Source:** `pinecone/admin/resources/api_key.py:75-108`

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `api_key_id` | `str` | Yes | — | The ID of the API key to fetch. |

| Return | Type | Description |
|--------|------|-------------|
| — | `APIKey` | The API key object with `id`, `name`, `project_id`, and `roles`. |

| Raises | Condition |
|--------|-----------|
| `NotFoundException` | Raised when the API key ID does not exist. |
| `UnauthorizedException` | Raised when authentication credentials lack sufficient permissions. |

**Example:**

```python
from pinecone import Admin

admin = Admin()
api_key = admin.api_key.fetch(api_key_id='my-api-key-id')
print(f"Name: {api_key.name}")
print(f"Roles: {api_key.roles}")
```

### get

Alias for `fetch()`. Retrieves an API key by its ID with identical behavior.

**Source:** `pinecone/admin/resources/api_key.py:111-134`

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `api_key_id` | `str` | Yes | — | The ID of the API key to get. |

| Return | Type | Description |
|--------|------|-------------|
| — | `APIKey` | The API key object with `id`, `name`, `project_id`, and `roles`. |

| Raises | Condition |
|--------|-----------|
| `NotFoundException` | Raised when the API key ID does not exist. |
| `UnauthorizedException` | Raised when authentication credentials lack sufficient permissions. |

**Example:**

```python
from pinecone import Admin

admin = Admin()
api_key = admin.api_key.get(api_key_id='my-api-key-id')
```

### describe

Alias for `fetch()`. Describes an API key by its ID with identical behavior.

**Source:** `pinecone/admin/resources/api_key.py:137-160`

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `api_key_id` | `str` | Yes | — | The ID of the API key to describe. |

| Return | Type | Description |
|--------|------|-------------|
| — | `APIKey` | The API key object with `id`, `name`, `project_id`, and `roles`. |

| Raises | Condition |
|--------|-----------|
| `NotFoundException` | Raised when the API key ID does not exist. |
| `UnauthorizedException` | Raised when authentication credentials lack sufficient permissions. |

**Example:**

```python
from pinecone import Admin

admin = Admin()
api_key = admin.api_key.describe(api_key_id='my-api-key-id')
```

### update

Update an API key's name and/or roles. At least one parameter must be provided.

**Source:** `pinecone/admin/resources/api_key.py:259-331`

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `api_key_id` | `str` | Yes | — | The ID of the API key to update. |
| `name` | `str` | No | `None` | A new name for the API key. Must be 1–80 characters. If omitted, the name is not updated. |
| `roles` | `list[str]` | No | `None` | A new set of roles for the API key. Available roles are: `ProjectEditor`, `ProjectViewer`, `ControlPlaneEditor`, `ControlPlaneViewer`, `DataPlaneEditor`, `DataPlaneViewer`. Existing roles will be completely replaced with the provided list. If omitted, roles are not updated. |

| Return | Type | Description |
|--------|------|-------------|
| — | `APIKey` | The updated API key object with `id`, `name`, `project_id`, and `roles`. |

| Raises | Condition |
|--------|-----------|
| `NotFoundException` | Raised when the API key ID does not exist. |
| `UnauthorizedException` | Raised when authentication credentials lack sufficient permissions. |

**Example:**

```python
from pinecone import Admin

admin = Admin()

# Update the name only
api_key = admin.api_key.update(
    api_key_id='my-api-key-id',
    name='updated-key-name'
)
print(f"Updated name: {api_key.name}")

# Update roles only (replaces existing roles)
api_key = admin.api_key.update(
    api_key_id='my-api-key-id',
    roles=['ProjectViewer', 'DataPlaneViewer']
)
print(f"Updated roles: {api_key.roles}")

# Update both name and roles
api_key = admin.api_key.update(
    api_key_id='my-api-key-id',
    name='new-name',
    roles=['ProjectEditor', 'DataPlaneEditor']
)
```

### delete

Delete an API key. Once deleted, the API key cannot be used.

**Source:** `pinecone/admin/resources/api_key.py:163-192`

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `api_key_id` | `str` | Yes | — | The ID of the API key to delete. |

| Return | Type | Description |
|--------|------|-------------|
| — | `None` | No value is returned. |

| Raises | Condition |
|--------|-----------|
| `NotFoundException` | Raised when the API key ID does not exist. |
| `UnauthorizedException` | Raised when authentication credentials lack sufficient permissions. |

**Example:**

```python
from pinecone import Admin
from pinecone import NotFoundException

admin = Admin()

# Delete an API key
admin.api_key.delete(api_key_id='my-api-key-id')

# Verify it's deleted
try:
    admin.api_key.fetch(api_key_id='my-api-key-id')
except NotFoundException:
    print("API key deleted successfully")
```

## Data Models

### APIKey

Response model returned by `fetch()`, `get()`, `describe()`, `update()`, and `list()` operations.

**Source:** `pinecone/core/openapi/admin/model/api_key.py`

| Field | Type | Nullable | Description |
|-------|------|----------|-------------|
| `id` | `string` | No | The unique identifier of the API key. |
| `name` | `string` | No | The name of the API key. |
| `project_id` | `string` | No | The ID of the project this API key belongs to. |
| `roles` | `array of string` | No | The list of roles assigned to this API key. Each role is one of: `ProjectEditor`, `ProjectViewer`, `ControlPlaneEditor`, `ControlPlaneViewer`, `DataPlaneEditor`, `DataPlaneViewer`. |

### APIKeyWithSecret

Response model returned by the `create()` operation. Contains both the API key metadata and the secret value.

**Source:** `pinecone/core/openapi/admin/model/api_key_with_secret.py`

| Field | Type | Nullable | Description |
|-------|------|----------|-------------|
| `key` | `APIKey` | No | The created API key object with `id`, `name`, `project_id`, and `roles`. |
| `value` | `string` | No | The secret value of the API key. **This is the only time this value is returned. Store it securely immediately; it cannot be retrieved later.** |

### CreateAPIKeyRequest

Request model for the `create()` operation.

**Source:** `pinecone/core/openapi/admin/model/create_api_key_request.py`

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `name` | `string` | Yes | 1–80 characters | The name of the API key. Must be between 1 and 80 characters. |
| `roles` | `array of string` | No | Valid values: `ProjectEditor`, `ProjectViewer`, `ControlPlaneEditor`, `ControlPlaneViewer`, `DataPlaneEditor`, `DataPlaneViewer` | An optional list of roles to assign to the API key. |

### UpdateAPIKeyRequest

Request model for the `update()` operation.

**Source:** `pinecone/core/openapi/admin/model/update_api_key_request.py`

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `name` | `string` | No | 1–80 characters | A new name for the API key. If provided, the name must be between 1 and 80 characters. |
| `roles` | `array of string` | No | Valid values: `ProjectEditor`, `ProjectViewer`, `ControlPlaneEditor`, `ControlPlaneViewer`, `DataPlaneEditor`, `DataPlaneViewer` | A new set of roles for the API key. If provided, existing roles are completely replaced. |

### ListApiKeysResponse

Response model returned by the `list()` operation.

**Source:** `pinecone/core/openapi/admin/model/list_api_keys_response.py`

| Field | Type | Nullable | Description |
|-------|------|----------|-------------|
| `data` | `array of APIKey` | No | A list of APIKey objects. Each contains `id`, `name`, `project_id`, and `roles`. |

## Available Roles

API keys can be assigned the following roles:

| Role | Description |
|------|-------------|
| `ProjectEditor` | Full control over project resources. Can create, update, and delete indexes, collections, and backups. |
| `ProjectViewer` | Read-only access to project resources. Can list and describe indexes, collections, and backups. |
| `ControlPlaneEditor` | Can manage organization, project, and API key settings. |
| `ControlPlaneViewer` | Read-only access to organization and project settings. |
| `DataPlaneEditor` | Can read and write data to indexes (upsert, query, update, delete vectors). |
| `DataPlaneViewer` | Read-only access to index data (query vectors only). |

Multiple roles can be assigned to a single API key. When updating an API key's roles, the entire role list is replaced with the new list (it's not a merge operation).

## Error Handling

API key operations can raise the following exceptions:

| Exception | Condition |
|-----------|-----------|
| `NotFoundException` | Raised when attempting to fetch, update, or delete an API key ID that does not exist. |
| `UnauthorizedException` | Raised when the authentication credentials lack sufficient permissions. |
| `ForbiddenException` | Raised when the authenticated user is not allowed to perform the operation. |
| `ServiceException` | Raised when the Pinecone service encounters an unexpected error. |

**Example Error Handling:**

```python
from pinecone import Admin, NotFoundException, UnauthorizedException

admin = Admin()

try:
    api_key = admin.api_key.fetch(api_key_id='non-existent-id')
except NotFoundException:
    print("API key not found")
except UnauthorizedException:
    print("Authentication failed")

try:
    admin.api_key.delete(api_key_id='my-key-id')
except NotFoundException:
    print("API key already deleted")
```

## Complete Workflow Example

```python
from pinecone import Admin, NotFoundException

admin = Admin()

# Find your project
project = admin.project.get(name='my-project')
print(f"Project ID: {project.id}")

# Create a new API key
print("\n1. Creating API key...")
response = admin.api_key.create(
    project_id=project.id,
    name='ci-testing-key',
    roles=['ProjectEditor', 'DataPlaneEditor']
)
api_key_id = response.key.id
secret_value = response.value
print(f"Created API key: {api_key_id}")
print(f"Secret value: {secret_value}")

# List all API keys
print("\n2. Listing all API keys...")
api_keys_response = admin.api_key.list(project_id=project.id)
print(f"Total API keys: {len(api_keys_response.data)}")

# Fetch a specific API key
print("\n3. Fetching specific API key...")
api_key = admin.api_key.fetch(api_key_id=api_key_id)
print(f"Name: {api_key.name}")
print(f"Roles: {api_key.roles}")

# Update the API key
print("\n4. Updating API key...")
updated_key = admin.api_key.update(
    api_key_id=api_key_id,
    name='ci-testing-key-updated',
    roles=['ProjectViewer', 'DataPlaneViewer']
)
print(f"Updated name: {updated_key.name}")
print(f"Updated roles: {updated_key.roles}")

# Delete the API key
print("\n5. Deleting API key...")
admin.api_key.delete(api_key_id=api_key_id)

# Verify deletion
try:
    admin.api_key.fetch(api_key_id=api_key_id)
except NotFoundException:
    print("API key deleted successfully")
```

## Notable Behaviors

1. **Secret Value Lifecycle:** The API key secret value is returned only during creation. There is no method to retrieve it later. If the secret is lost, a new API key must be created.

2. **Role Replacement:** When updating an API key's roles, the entire role list is replaced. There is no mechanism to add or remove individual roles; you must provide the complete desired role list.

3. **Method Aliases:** The `get()` and `describe()` methods are exact aliases of `fetch()` with identical behavior and exceptions.

4. **Name Constraints:** When updating an API key, the name must be 1–80 characters if provided. The `create()` method has the same constraint.

5. **Required Keyword Arguments:** All methods require keyword arguments. Positional arguments are not supported.

6. **No Partial Updates:** The `update()` method does not perform partial updates. Fields omitted from the call are not modified.
