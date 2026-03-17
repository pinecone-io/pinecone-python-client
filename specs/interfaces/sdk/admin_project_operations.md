# Admin Project Operations

Methods for managing projects in a Pinecone organization via the `ProjectResource` class (available on `Admin` client instances as `.project` or `.projects`).

This spec documents the `create()` and `list()` methods — the two primary operations for creating and discovering projects. Alias methods (`get()`, `describe()`, `fetch()`), update operations, and deletion are not covered in this spec and are reserved for future documentation.

---

## ProjectResource.create

Creates a new project in the organization.

**Source:** `pinecone/admin/resources/project.py:278-326`

**Added:** v1.0
**Deprecated:** No

### Signature

```python
def create(
    self,
    *,
    name: str,
    max_pods: int | None = None,
    force_encryption_with_cmek: bool | None = None,
) -> Project:
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|-----------|-------------|
| `name` | `string (1–512 chars)` | Yes | — | v1.0 | No | The name of the project. Must be between 1 and 512 characters. |
| `max_pods` | `integer (int32)` | No | `None` | v1.0 | No | The maximum number of pods allocated to this project. When omitted, no pod quota limit is set. |
| `force_encryption_with_cmek` | `boolean` | No | `None` | v1.0 | No | Whether to force encryption using customer-managed encryption keys (CMEK). When omitted, defaults to standard encryption. |

### Returns

**Type:** `Project`

A project object with the following fields:
- `id`: Unique identifier for the project. Use this ID to reference the project in other API operations.
- `name`: The name assigned to the project.
- `max_pods`: The maximum number of pod indexes that can be created in this project. When `null`, the project has no pod quota limit.
- `force_encryption_with_cmek`: Whether this project enforces encryption using customer-managed keys. If `true`, any index created in this project must use CMEK.
- `organization_id`: The ID of the organization that owns this project.
- `created_at`: The timestamp when the project was created (ISO 8601 format).

### Raises

| Exception | Condition |
|-----------|-----------|
| `PineconeApiValueError` | The `name` parameter is empty or exceeds 512 characters. |
| `PineconeApiException` | A project with the same name already exists in the organization. Returns HTTP 409. |
| `PineconeApiException` | The organization has reached its resource limit. Returns HTTP 429. |
| `UnauthorizedException` | Client credentials (client_id/client_secret) are invalid or expired. Returns HTTP 401. |

### Idempotency

Non-idempotent. Repeated identical calls create multiple projects with the same name.

### Side Effects

- Creates a new project resource in the organization.
- The project is immediately available for use (indexes, collections, backups can be created within it).

### Example

```python
from pinecone import Admin

admin = Admin()

# Create a project with pod quota
project = admin.project.create(
    name="my-production-project",
    max_pods=50,
    force_encryption_with_cmek=True
)

print(project.id)  # "550e8400-e29b-41d4-a716-446655440000"
print(project.name)  # "my-production-project"
print(project.max_pods)  # 50
print(project.created_at)  # "2026-03-17T10:30:00Z"

# Create a project with no pod quota limit
project2 = admin.project.create(
    name="my-dev-project",
)
print(project2.max_pods)  # None
```

### Notable Behavior

- **Kwargs-only:** All parameters are keyword-only (called with `name="..."`, not positional arguments).
- **Partial updates:** When `max_pods` or `force_encryption_with_cmek` are omitted, they default to organization-level settings, not to `null` in the response.
- **Name uniqueness:** Project names are unique within the organization. Attempting to create a project with an existing name raises a 409 conflict.

---

## ProjectResource.list

Lists all projects in the organization.

**Source:** `pinecone/admin/resources/project.py:52-79`

**Added:** v1.0
**Deprecated:** No

### Signature

```python
def list(self) -> ProjectList:
```

### Parameters

None.

### Returns

**Type:** `ProjectList`

An object with a `data` field containing a list of all projects in the organization. Each project object in the list has:
- `id`: Unique identifier for the project. Use this ID to reference the project in other API operations.
- `name`: The name assigned to the project.
- `max_pods`: The maximum number of pod indexes that can be created in this project. When `null`, the project has no pod quota limit.
- `force_encryption_with_cmek`: Whether this project enforces encryption using customer-managed keys. If `true`, any index created in this project must use CMEK.
- `organization_id`: The ID of the organization that owns this project.
- `created_at`: The timestamp when the project was created (ISO 8601 format).

### Raises

| Exception | Condition |
|-----------|-----------|
| `UnauthorizedException` | Client credentials (client_id/client_secret) are invalid or expired. Returns HTTP 401. |
| `PineconeApiException` | The organization API service is unavailable. Returns HTTP 503. |

### Idempotency

Idempotent. Repeated calls return the same list (unless projects are created or deleted between calls).

### Side Effects

None. This is a read-only operation.

### Example

```python
from pinecone import Admin

admin = Admin()

# List all projects in the organization
projects_response = admin.projects.list()

for project in projects_response.data:
    print(f"Project: {project.name}")
    print(f"  ID: {project.id}")
    print(f"  Max Pods: {project.max_pods}")
    print(f"  CMEK Enabled: {project.force_encryption_with_cmek}")
    print(f"  Created: {project.created_at}")

# Check if any projects exist
if projects_response.data:
    print(f"Organization has {len(projects_response.data)} projects")
else:
    print("No projects found")
```

### Notable Behavior

- **No pagination:** The `list()` method returns all projects in the organization in a single response. There is no pagination support or limit parameter.
- **Order:** Projects are returned in the order they are stored in the system (typically creation order, but not guaranteed).
- **Empty list:** If the organization has no projects, `data` is an empty list (not `null`).
- **Consistency:** The list reflects projects as they exist at the moment the API call is made. Projects created or deleted immediately after the call may not be reflected.
