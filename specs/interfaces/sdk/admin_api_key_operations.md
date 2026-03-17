# Admin API Key Operations

This module documents API key management operations on the Admin client: creating, listing, retrieving, updating, and deleting API keys for Pinecone projects. All operations require a service account and provide project-scoped control plane access to the API key lifecycle.

API key operations are accessed through the Admin client's `api_key` or `api_keys` property:

```python
from pinecone import Admin

admin = Admin(client_id="your-client-id", client_secret="your-client-secret")

# Both are equivalent
admin.api_key.list(project_id="my-project-id")
admin.api_keys.list(project_id="my-project-id")
```

---

## `Admin.api_key.create()`

Creates a new API key for a specified project.

**Source:** `pinecone/admin/resources/api_key.py:195-256`
**Added:** v8.0
**Deprecated:** No
**Idempotency:** Not idempotent — repeated calls will create duplicate keys
**Side effects:** Creates a new API key resource in the Pinecone API

### Signature

```python
def create(
    self,
    project_id: str,
    name: str,
    description: str | None = None,
    roles: list[str] | None = None
) -> APIKeyWithSecret
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `project_id` | `str` | Yes | — | v8.0 | No | The ID of the project in which to create the API key. Obtain this via `admin.project.list()` or `admin.project.get(name="...")`. |
| `name` | `str` | Yes | — | v8.0 | No | The name of the API key. Must be 1-80 characters long. |
| `description` | `str \| None` | No | `None` | v8.0 | No | Optional description of the API key's purpose or usage. |
| `roles` | `list[str] \| None` | No | `None` | v8.0 | No | Optional list of role strings to assign to the key. If omitted, defaults to `["ProjectEditor"]`. See **Available Roles** section below. |

### Returns

**Type:** `APIKeyWithSecret` — A composite object with two properties:
- `key` (APIKey) — The created API key object with `id`, `name`, `project_id`, and `roles` fields.
- `value` (str) — The secret key value in the format `pckey_<public-label>_<unique-key>`. **This value is returned only on creation and cannot be retrieved later.** Store this value securely.

### Raises

| Exception | Condition |
|-----------|-----------|
| `BadRequestException` | Invalid `project_id`, invalid role names, or malformed request. |
| `UnauthorizedException` | The service account credentials are invalid or missing. |
| `NotFoundException` | The specified `project_id` does not exist. |
| `PineconeApiException` | Unexpected server error. |

### Example

```python
from pinecone import Admin

admin = Admin()

# Get a project
project = admin.project.get(name='my-project')

# Create an API key with default role (ProjectEditor)
response = admin.api_key.create(
    project_id=project.id,
    name='ci-automation-key',
    description='Key for CI/CD pipeline'
)

print(f"API Key ID: {response.key.id}")
print(f"API Key Name: {response.key.name}")
print(f"API Key Roles: {response.key.roles}")
print(f"Secret Value: {response.value}")  # Store this securely

# Create an API key with custom roles
response2 = admin.api_key.create(
    project_id=project.id,
    name='data-plane-reader',
    roles=['DataPlaneEditor', 'ProjectViewer']
)
```

### Notes

- The secret value is displayed only once at creation time. There is no way to retrieve it later.
- If no `roles` are specified, the API key defaults to the `ProjectEditor` role.
- The name parameter is subject to 1-80 character validation at the API level.

---

## `Admin.api_key.list()`

Lists all API keys for a specified project.

**Source:** `pinecone/admin/resources/api_key.py:33-72`
**Added:** v8.0
**Deprecated:** No
**Idempotency:** Safe to retry
**Side effects:** None

### Signature

```python
def list(self, project_id: str) -> ListApiKeysResponse
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `project_id` | `str` | Yes | — | v8.0 | No | The ID of the project for which to list API keys. |

### Returns

**Type:** `ListApiKeysResponse` — A response object with a `data` property containing a list of `APIKey` objects.

Structure of returned object:
```python
{
    "data": [
        {
            "id": "api-key-id-1",
            "name": "my-api-key",
            "project_id": "my-project-id",
            "roles": ["ProjectEditor", "DataPlaneEditor"]
        },
        # ... more API keys
    ]
}
```

### Raises

| Exception | Condition |
|-----------|-----------|
| `BadRequestException` | Invalid or missing `project_id`. |
| `UnauthorizedException` | The service account credentials are invalid or missing. |
| `NotFoundException` | The specified `project_id` does not exist. |
| `PineconeApiException` | Unexpected server error. |

### Example

```python
from pinecone import Admin

admin = Admin()

# Get a project
project = admin.project.get(name='my-project')

# List all API keys for the project
response = admin.api_key.list(project_id=project.id)

for api_key in response.data:
    print(f"ID: {api_key.id}")
    print(f"Name: {api_key.name}")
    print(f"Roles: {api_key.roles}")
    print("---")

# Direct iteration is also supported
for api_key in response.data:
    print(f"{api_key.name} ({api_key.id})")
```

### Notes

- The `value` (secret) of the API key is **not** returned in list operations. The secret is only available immediately after creation.
- The response is wrapped in a `data` property; iterate over `response.data` to access the list of keys.

---

## `Admin.api_key.fetch()`

Fetches a specific API key by its ID.

**Source:** `pinecone/admin/resources/api_key.py:75-108`
**Added:** v8.0
**Deprecated:** No
**Idempotency:** Safe to retry
**Side effects:** None

### Signature

```python
def fetch(self, api_key_id: str) -> APIKey
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `api_key_id` | `str` | Yes | — | v8.0 | No | The ID of the API key to fetch. |

### Returns

**Type:** `APIKey` — An object representing the API key with the following properties:
- `id` (str) — The unique identifier of the API key.
- `name` (str) — The name of the API key.
- `project_id` (str) — The ID of the project that owns the API key.
- `roles` (list[str]) — A list of role strings assigned to the API key.

### Raises

| Exception | Condition |
|-----------|-----------|
| `BadRequestException` | Invalid or missing `api_key_id`. |
| `UnauthorizedException` | The service account credentials are invalid or missing. |
| `NotFoundException` | The specified API key does not exist. |
| `PineconeApiException` | Unexpected server error. |

### Example

```python
from pinecone import Admin

admin = Admin()

# Fetch an API key by ID
api_key = admin.api_key.fetch(api_key_id='api-key-12345')

print(f"Name: {api_key.name}")
print(f"Project ID: {api_key.project_id}")
print(f"Roles: {api_key.roles}")
```

### Notes

- The `value` (secret) of the API key is **not** returned. Secrets are only available at creation time.
- This method follows the @require_kwargs decorator, meaning all parameters must be passed as keyword arguments.

---

## `Admin.api_key.get()`

Alias for `fetch()`. Retrieves a specific API key by its ID.

**Source:** `pinecone/admin/resources/api_key.py:111-134`
**Added:** v8.0
**Deprecated:** No
**Idempotency:** Safe to retry
**Side effects:** None

### Signature

```python
def get(self, api_key_id: str) -> APIKey
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `api_key_id` | `str` | Yes | — | v8.0 | No | The ID of the API key to retrieve. |

### Returns

**Type:** `APIKey` — An object with `id`, `name`, `project_id`, and `roles` properties.

### Raises

| Exception | Condition |
|-----------|-----------|
| `BadRequestException` | Invalid or missing `api_key_id`. |
| `UnauthorizedException` | The service account credentials are invalid or missing. |
| `NotFoundException` | The specified API key does not exist. |
| `PineconeApiException` | Unexpected server error. |

### Example

```python
from pinecone import Admin

admin = Admin()

# get() is an alias for fetch()
api_key = admin.api_key.get(api_key_id='api-key-12345')
print(f"API Key Name: {api_key.name}")
```

---

## `Admin.api_key.describe()`

Alias for `fetch()`. Describes a specific API key by its ID.

**Source:** `pinecone/admin/resources/api_key.py:137-160`
**Added:** v8.0
**Deprecated:** No
**Idempotency:** Safe to retry
**Side effects:** None

### Signature

```python
def describe(self, api_key_id: str) -> APIKey
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `api_key_id` | `str` | Yes | — | v8.0 | No | The ID of the API key to describe. |

### Returns

**Type:** `APIKey` — An object with `id`, `name`, `project_id`, and `roles` properties.

### Raises

| Exception | Condition |
|-----------|-----------|
| `BadRequestException` | Invalid or missing `api_key_id`. |
| `UnauthorizedException` | The service account credentials are invalid or missing. |
| `NotFoundException` | The specified API key does not exist. |
| `PineconeApiException` | Unexpected server error. |

### Example

```python
from pinecone import Admin

admin = Admin()

# describe() is an alias for fetch()
api_key = admin.api_key.describe(api_key_id='api-key-12345')
print(f"Described API Key: {api_key.name} with roles {api_key.roles}")
```

---

## `Admin.api_key.update()`

Updates an existing API key's name and/or roles.

**Source:** `pinecone/admin/resources/api_key.py:259-331`
**Added:** v8.0
**Deprecated:** No
**Idempotency:** Idempotent — repeated calls with the same parameters produce the same result
**Side effects:** Modifies the API key resource in the Pinecone API

### Signature

```python
def update(
    self,
    api_key_id: str,
    name: str | None = None,
    roles: list[str] | None = None
) -> APIKey
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `api_key_id` | `str` | Yes | — | v8.0 | No | The ID of the API key to update. |
| `name` | `str \| None` | No | `None` | v8.0 | No | A new name for the API key. If provided, must be 1-80 characters long. If omitted, the name will not be updated. |
| `roles` | `list[str] \| None` | No | `None` | v8.0 | No | A new set of role strings for the API key. If provided, existing roles are completely replaced with the new list. If omitted, the roles will not be updated. See **Available Roles** section below. |

### Returns

**Type:** `APIKey` — The updated API key object with `id`, `name`, `project_id`, and `roles` properties.

### Raises

| Exception | Condition |
|-----------|-----------|
| `BadRequestException` | Invalid `api_key_id`, invalid role names, or malformed request. |
| `UnauthorizedException` | The service account credentials are invalid or missing. |
| `NotFoundException` | The specified API key does not exist. |
| `PineconeApiException` | Unexpected server error. |

### Example

```python
from pinecone import Admin

admin = Admin()

# Update only the name
updated_key = admin.api_key.update(
    api_key_id='api-key-12345',
    name='updated-key-name'
)
print(f"Updated name: {updated_key.name}")
print(f"Roles unchanged: {updated_key.roles}")

# Update only the roles
updated_key = admin.api_key.update(
    api_key_id='api-key-12345',
    roles=['ProjectViewer', 'DataPlaneEditor']
)
print(f"Name unchanged: {updated_key.name}")
print(f"Updated roles: {updated_key.roles}")

# Update both name and roles
updated_key = admin.api_key.update(
    api_key_id='api-key-12345',
    name='new-name',
    roles=['ProjectEditor']
)
print(f"Updated: {updated_key.name} with roles {updated_key.roles}")
```

### Notes

- Either `name`, `roles`, or both may be provided. Omitted fields are not updated.
- When `roles` is provided, it completely replaces the existing role list. There is no partial role update; you must pass the full desired role list.
- The name parameter is subject to 1-80 character validation at the API level.

---

## `Admin.api_key.delete()`

Deletes an API key.

**Source:** `pinecone/admin/resources/api_key.py:163-192`
**Added:** v8.0
**Deprecated:** No
**Idempotency:** Idempotent — deleting an already-deleted key raises `NotFoundException`
**Side effects:** Permanently removes the API key from the Pinecone API

### Signature

```python
def delete(self, api_key_id: str) -> None
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `api_key_id` | `str` | Yes | — | v8.0 | No | The ID of the API key to delete. |

### Returns

**Type:** `None` — This method returns nothing on success.

### Raises

| Exception | Condition |
|-----------|-----------|
| `BadRequestException` | Invalid or missing `api_key_id`. |
| `UnauthorizedException` | The service account credentials are invalid or missing. |
| `NotFoundException` | The specified API key does not exist. |
| `PineconeApiException` | Unexpected server error. |

### Example

```python
from pinecone import Admin
from pinecone.exceptions import NotFoundException

admin = Admin()

# Delete an API key
admin.api_key.delete(api_key_id='api-key-12345')
print("API key deleted")

# Verify deletion by trying to fetch the deleted key
try:
    admin.api_key.fetch(api_key_id='api-key-12345')
except NotFoundException:
    print("API key is confirmed deleted")
```

### Notes

- Deletion is permanent and cannot be undone.
- Attempting to delete a non-existent key raises `NotFoundException`.

---

## Available Roles

API keys can be assigned one or more of the following roles:

| Role | Description |
|------|-------------|
| `ProjectEditor` | Full access to project resources (indexes, API keys, etc.). Can create, read, update, and delete project resources. |
| `ProjectViewer` | Read-only access to project resources. Cannot modify or delete. |
| `ControlPlaneEditor` | Full access to control plane operations (organizations, projects, billing). |
| `ControlPlaneViewer` | Read-only access to control plane operations. |
| `DataPlaneEditor` | Full access to data plane operations (upsert, query, delete on indexes). |
| `DataPlaneViewer` | Read-only access to data plane operations (query, fetch, describe on indexes). |

**Common Role Combinations**

- **`["ProjectEditor"]`** (default) — Full access to a single project.
- **`["ProjectViewer"]`** — Read-only access to a project.
- **`["DataPlaneEditor"]`** — Can upsert and query vectors; used for application data access.
- **`["DataPlaneViewer"]`** — Can only query; minimal permissions for read-only applications.
- **`["ProjectEditor", "DataPlaneEditor"]`** — Can manage project and perform all data operations.

---

## Error Handling

All API key operations may raise the following exceptions:

| Exception | Cause | Handling |
|-----------|-------|----------|
| `PineconeApiException` | Unexpected server error | Implement retry logic with exponential backoff |
| `BadRequestException` | Malformed request, invalid parameter values | Validate inputs before retrying |
| `UnauthorizedException` | Invalid or missing credentials | Verify `PINECONE_CLIENT_ID` and `PINECONE_CLIENT_SECRET` environment variables or constructor arguments |
| `NotFoundException` | Resource (project or API key) does not exist | Confirm resource IDs; list operations to verify existence |

---

## Usage Patterns

### Complete Workflow

```python
from pinecone import Admin

# Initialize Admin client (reads PINECONE_CLIENT_ID and PINECONE_CLIENT_SECRET)
admin = Admin()

# Get a project
project = admin.project.get(name='my-project')

# Create an API key
response = admin.api_key.create(
    project_id=project.id,
    name='app-api-key',
    description='For my application',
    roles=['DataPlaneEditor']
)
print(f"API Key ID: {response.key.id}")
print(f"Secret: {response.value}")  # Store this securely

# List all API keys for the project
keys = admin.api_key.list(project_id=project.id)
for key in keys.data:
    print(f"{key.name}: {key.roles}")

# Update an API key's name
updated = admin.api_key.update(
    api_key_id=response.key.id,
    name='app-api-key-v2'
)

# Delete the API key
admin.api_key.delete(api_key_id=response.key.id)
```

### Error Handling

```python
from pinecone import Admin
from pinecone.exceptions import NotFoundException, UnauthorizedException

try:
    admin = Admin()
    api_key = admin.api_key.fetch(api_key_id='my-key-id')
except UnauthorizedException:
    print("Invalid credentials. Check PINECONE_CLIENT_ID and PINECONE_CLIENT_SECRET.")
except NotFoundException:
    print("API key not found.")
```

---

## Data Models

### `APIKey`

Represents an API key resource with its metadata and roles.

**Source:** `pinecone/core/openapi/admin/model/api_key.py:36-306`

| Field | Type | Nullable | Since | Deprecated | Description |
|-------|------|----------|-------|------------|-------------|
| `id` | `str` | No | v8.0 | No | The unique identifier of the API key. |
| `name` | `str` | No | v8.0 | No | The name of the API key. 1-80 characters. |
| `project_id` | `str` | No | v8.0 | No | The ID of the project that owns this API key. |
| `roles` | `list[str]` | No | v8.0 | No | A list of role strings assigned to this API key. See **Available Roles** section. Example: `["ProjectEditor", "DataPlaneEditor"]`. |

**Example**

```python
api_key = {
    "id": "api-key-id-1",
    "name": "my-api-key",
    "project_id": "my-project-id",
    "roles": ["ProjectEditor", "DataPlaneEditor"]
}
```

---

### `APIKeyWithSecret`

Returned only from the `create()` method. Contains the newly created API key metadata plus the secret value.

**Source:** `pinecone/core/openapi/admin/model/api_key_with_secret.py:47-306`

| Field | Type | Nullable | Since | Deprecated | Description |
|-------|------|----------|-------|------------|-------------|
| `key` | `APIKey` | No | v8.0 | No | The created API key object with `id`, `name`, `project_id`, and `roles`. |
| `value` | `str` | No | v8.0 | No | The secret key value in the format `pckey_<public-label>_<unique-key>`. This is the only time this value is returned. **Store this securely.** |

**Example**

```python
response = admin.api_key.create(
    project_id="my-project-id",
    name="my-new-key"
)

# Access the key metadata and secret
key_id = response.key.id
key_name = response.key.name
secret_value = response.value  # Only available here
```

---

### `CreateAPIKeyRequest`

Request body sent to the API when creating an API key. This is constructed internally but documented for reference.

**Source:** `pinecone/core/openapi/admin/model/create_api_key_request.py:36-294`

| Field | Type | Nullable | Since | Deprecated | Description |
|-------|------|----------|-------|------------|-------------|
| `name` | `str` | No | v8.0 | No | The name of the API key. Must be 1-80 characters long. |
| `roles` | `list[str]` | Yes | v8.0 | No | Optional list of roles to assign. If omitted, defaults to `["ProjectEditor"]`. |

**Validation Rules**

- `name`: Must be between 1 and 80 characters (inclusive).

---

### `UpdateAPIKeyRequest`

Request body sent to the API when updating an API key. This is constructed internally but documented for reference.

**Source:** `pinecone/core/openapi/admin/model/update_api_key_request.py:36-288`

| Field | Type | Nullable | Since | Deprecated | Description |
|-------|------|----------|-------|------------|-------------|
| `name` | `str` | Yes | v8.0 | No | New name for the API key. Must be 1-80 characters if provided. If omitted, name is not updated. |
| `roles` | `list[str]` | Yes | v8.0 | No | New list of roles for the API key. Existing roles are completely replaced if provided. If omitted, roles are not updated. |

**Validation Rules**

- `name`: Must be between 1 and 80 characters (inclusive) if provided.
- At least one field must be provided for the update to have an effect.

---

### `ListApiKeysResponse`

Response object returned by the `list()` method. Contains a list of API keys for a project.

**Source:** `pinecone/core/openapi/admin/model/list_api_keys_response.py:47-300`

| Field | Type | Nullable | Since | Deprecated | Description |
|-------|------|----------|-------|------------|-------------|
| `data` | `list[APIKey]` | No | v8.0 | No | A list of `APIKey` objects representing all API keys in the project. May be an empty list if the project has no API keys. |

**Example**

```python
response = admin.api_key.list(project_id="my-project-id")

# Access the list of API keys
for api_key in response.data:
    print(f"ID: {api_key.id}")
    print(f"Name: {api_key.name}")
    print(f"Roles: {api_key.roles}")
```
