# Admin Project Operations

This module documents project management operations on the Admin client: creating, listing, retrieving, updating, and deleting projects for Pinecone organizations. All operations require a service account and provide organization-scoped control plane access to the project lifecycle.

Project operations are accessed through the Admin client's `project` or `projects` property:

```python
from pinecone import Admin

admin = Admin(client_id="your-client-id", client_secret="your-client-secret")

# Both are equivalent
admin.project.list()
admin.projects.list()
```

---

## `Admin.project.list()`

Lists all projects in the organization.

**Source:** `pinecone/admin/resources/project.py:52-79`

**Added:** v8.0
**Deprecated:** No
**Idempotency:** Safe to retry — the method is read-only and does not modify state
**Side effects:** None

### Signature

```python
def list(self) -> ProjectList:
```

### Parameters

None.

### Returns

**Type:** `ProjectList` — An object containing a `data` list of all projects in the organization.

### Raises

| Exception | Condition |
|-----------|-----------|
| `UnauthorizedException` | The API credentials are invalid or missing. |
| `ForbiddenException` | The API key lacks the required permissions to list projects. |
| `PineconeApiException` | Unexpected server error. |

### Example

```python
from pinecone import Admin

admin = Admin(client_id="your-client-id", client_secret="your-client-secret")

# List all projects in the organization
projects_response = admin.projects.list()
for project in projects_response.data:
    print(f"Project: {project.name} (ID: {project.id})")
    print(f"  Max pods: {project.max_pods}")
    print(f"  CMEK encryption: {project.force_encryption_with_cmek}")
    print(f"  Created: {project.created_at}")
```

---

## `Admin.project.fetch()`

Retrieves a single project by project ID or by name.

**Source:** `pinecone/admin/resources/project.py:81-153`

**Added:** v8.0
**Deprecated:** No
**Idempotency:** Safe to retry — the method is read-only and does not modify state
**Side effects:** None

### Signature

```python
def fetch(
    self,
    project_id: str | None = None,
    name: str | None = None
) -> Project:
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `project_id` | `str` | No | `None` | v8.0 | No | The unique identifier of the project to fetch. Either `project_id` or `name` must be provided, but not both. |
| `name` | `str` | No | `None` | v8.0 | No | The name of the project to fetch. Either `project_id` or `name` must be provided, but not both. When provided, searches all projects in the organization to find a match. Raises `NotFoundException` if no matching project is found. Raises `PineconeException` if multiple projects with the same name are found. |

### Returns

**Type:** `Project` — The requested project object.

### Raises

| Exception | Condition |
|-----------|-----------|
| `ValueError` | Both `project_id` and `name` are provided, or neither is provided. |
| `NotFoundException` | No project exists with the given project ID or name. |
| `PineconeException` | When using `name` parameter, multiple projects with the same name were found. Error message includes all matching project IDs. |
| `UnauthorizedException` | The API credentials are invalid or missing. |
| `ForbiddenException` | The API key lacks the required permissions to fetch projects. |

### Example

```python
from pinecone import Admin

admin = Admin(client_id="your-client-id", client_secret="your-client-secret")

# Fetch a project by project_id
project = admin.projects.fetch(project_id="42ca341d-43bf-47cb-9f27-e645dbfabea6")
print(f"Project: {project.name}")
print(f"Organization ID: {project.organization_id}")
print(f"Max pods: {project.max_pods}")

# Fetch a project by name
project = admin.projects.fetch(name="my-project")
print(f"Project ID: {project.id}")
```

### Notes

- When fetching by name, the operation is not atomic — it first lists all projects and filters by name.
- If multiple projects share the same name (which shouldn't normally occur), the error message includes all matching project IDs for debugging.

---

## `Admin.project.get()`

Alias for `fetch()`. Retrieves a single project by project ID or by name.

**Source:** `pinecone/admin/resources/project.py:155-181`

**Added:** v8.0
**Deprecated:** No
**Idempotency:** Safe to retry — the method is read-only and does not modify state
**Side effects:** None

### Signature

```python
def get(
    self,
    project_id: str | None = None,
    name: str | None = None
) -> Project:
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `project_id` | `str` | No | `None` | v8.0 | No | The unique identifier of the project to get. Either `project_id` or `name` must be provided, but not both. |
| `name` | `str` | No | `None` | v8.0 | No | The name of the project to get. Either `project_id` or `name` must be provided, but not both. |

### Returns

**Type:** `Project` — The requested project object.

### Raises

| Exception | Condition |
|-----------|-----------|
| `ValueError` | Both `project_id` and `name` are provided, or neither is provided. |
| `NotFoundException` | No project exists with the given project ID or name. |
| `PineconeException` | When using `name` parameter, multiple projects with the same name were found. |
| `UnauthorizedException` | The API credentials are invalid or missing. |
| `ForbiddenException` | The API key lacks the required permissions to fetch projects. |

### Example

```python
from pinecone import Admin

admin = Admin(client_id="your-client-id", client_secret="your-client-secret")

# Get a project by project_id
project = admin.project.get(project_id="42ca341d-43bf-47cb-9f27-e645dbfabea6")
print(f"Project: {project.name}")
```

---

## `Admin.project.describe()`

Alias for `fetch()`. Retrieves a single project by project ID or by name.

**Source:** `pinecone/admin/resources/project.py:183-209`

**Added:** v8.0
**Deprecated:** No
**Idempotency:** Safe to retry — the method is read-only and does not modify state
**Side effects:** None

### Signature

```python
def describe(
    self,
    project_id: str | None = None,
    name: str | None = None
) -> Project:
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `project_id` | `str` | No | `None` | v8.0 | No | The unique identifier of the project to describe. Either `project_id` or `name` must be provided, but not both. |
| `name` | `str` | No | `None` | v8.0 | No | The name of the project to describe. Either `project_id` or `name` must be provided, but not both. |

### Returns

**Type:** `Project` — The requested project object.

### Raises

| Exception | Condition |
|-----------|-----------|
| `ValueError` | Both `project_id` and `name` are provided, or neither is provided. |
| `NotFoundException` | No project exists with the given project ID or name. |
| `PineconeException` | When using `name` parameter, multiple projects with the same name were found. |
| `UnauthorizedException` | The API credentials are invalid or missing. |
| `ForbiddenException` | The API key lacks the required permissions to fetch projects. |

### Example

```python
from pinecone import Admin

admin = Admin(client_id="your-client-id", client_secret="your-client-secret")

# Describe a project by project_id
project = admin.project.describe(project_id="42ca341d-43bf-47cb-9f27-e645dbfabea6")
print(f"Project: {project.name}")
print(f"Max pods: {project.max_pods}")
```

---

## `Admin.project.exists()`

Checks whether a project exists by project ID or by name.

**Source:** `pinecone/admin/resources/project.py:211-275`

**Added:** v8.0
**Deprecated:** No
**Idempotency:** Safe to retry — the method is read-only and does not modify state
**Side effects:** None

### Signature

```python
def exists(
    self,
    project_id: str | None = None,
    name: str | None = None
) -> bool:
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `project_id` | `str` | No | `None` | v8.0 | No | The unique identifier of the project to check. Either `project_id` or `name` must be provided, but not both. |
| `name` | `str` | No | `None` | v8.0 | No | The name of the project to check. Either `project_id` or `name` must be provided, but not both. |

### Returns

**Type:** `bool` — `True` if the project exists, `False` otherwise.

### Raises

| Exception | Condition |
|-----------|-----------|
| `ValueError` | Both `project_id` and `name` are provided, or neither is provided. |
| `UnauthorizedException` | The API credentials are invalid or missing. |
| `ForbiddenException` | The API key lacks the required permissions to check project existence. |
| `PineconeException` | When using `name` parameter, multiple projects with the same name were found. |

### Example

```python
from pinecone import Admin

admin = Admin(client_id="your-client-id", client_secret="your-client-secret")

project_name = "my-project"
if admin.project.exists(name=project_name):
    print(f"Project {project_name} exists")
else:
    # Create the project if it doesn't exist
    project = admin.project.create(
        name=project_name,
        max_pods=10,
        force_encryption_with_cmek=False
    )
    print(f"Created project: {project.id}")
```

---

## `Admin.project.create()`

Creates a new project in the organization.

**Source:** `pinecone/admin/resources/project.py:277-326`

**Added:** v8.0
**Deprecated:** No
**Idempotency:** Non-idempotent — repeated calls will create duplicate projects
**Side effects:** Creates a new project resource in the Pinecone API

### Signature

```python
def create(
    self,
    name: str,
    max_pods: int | None = None,
    force_encryption_with_cmek: bool | None = None
) -> Project:
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `name` | `str` | Yes | — | v8.0 | No | The name of the project. Must be 1-512 characters long. |
| `max_pods` | `int` | No | `None` | v8.0 | No | The maximum number of pods that can be created in the project. If omitted, no maximum is enforced. |
| `force_encryption_with_cmek` | `bool` | No | `None` | v8.0 | No | Whether to force encryption with a customer-managed encryption key (CMEK). If omitted, defaults to the organization's default encryption setting. |

### Returns

**Type:** `Project` — The newly created project object.

### Raises

| Exception | Condition |
|-----------|-----------|
| `UnauthorizedException` | The API credentials are invalid or missing. |
| `ForbiddenException` | The API key lacks the required permissions to create projects. |

### Example

```python
from pinecone import Admin

admin = Admin(client_id="your-client-id", client_secret="your-client-secret")

# Create a project with default settings
project = admin.project.create(name="my-project-name")
print(f"Created project: {project.id}")
print(f"Name: {project.name}")
print(f"Organization ID: {project.organization_id}")
print(f"Created at: {project.created_at}")

# Create a project with specific settings
project = admin.project.create(
    name="my-production-project",
    max_pods=20,
    force_encryption_with_cmek=True
)
print(f"Project {project.name} created with max {project.max_pods} pods")
```

---

## `Admin.project.update()`

Updates an existing project.

**Source:** `pinecone/admin/resources/project.py:328-386`

**Added:** v8.0
**Deprecated:** No
**Idempotency:** Idempotent when all parameters remain unchanged — repeated calls with the same parameters produce the same result
**Side effects:** Modifies an existing project resource in the Pinecone API

### Signature

```python
def update(
    self,
    project_id: str,
    name: str | None = None,
    max_pods: int | None = None,
    force_encryption_with_cmek: bool | None = None
) -> Project:
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `project_id` | `str` | Yes | — | v8.0 | No | The unique identifier of the project to update. |
| `name` | `str` | No | `None` | v8.0 | No | A new name for the project. Must be 1-512 characters long. If omitted, the project name is not changed. |
| `max_pods` | `int` | No | `None` | v8.0 | No | A new maximum number of pods for the project. If omitted, the current max pods setting is not changed. |
| `force_encryption_with_cmek` | `bool` | No | `None` | v8.0 | No | Whether to force encryption with a customer-managed encryption key (CMEK). If omitted, the current encryption setting is not changed. |

### Returns

**Type:** `Project` — The updated project object.

### Raises

| Exception | Condition |
|-----------|-----------|
| `NotFoundException` | The project with the given `project_id` does not exist. |
| `UnauthorizedException` | The API credentials are invalid or missing. |
| `ForbiddenException` | The API key lacks the required permissions to update projects. |

### Example

```python
from pinecone import Admin

admin = Admin(client_id="your-client-id", client_secret="your-client-secret")

# Get a project first
project = admin.project.get(name="my-project")

# Update only the max_pods
updated_project = admin.project.update(
    project_id=project.id,
    max_pods=20
)
print(f"Updated max pods to: {updated_project.max_pods}")

# Update only the CMEK encryption setting
updated_project = admin.project.update(
    project_id=project.id,
    force_encryption_with_cmek=True
)
print(f"CMEK encryption enabled: {updated_project.force_encryption_with_cmek}")
```

### Notes

- Only provided parameters are updated; omitted parameters are left unchanged.
- The operation is atomic at the API level, but note that if only some fields are updated, other concurrent updates might observe intermediate state.

---

## `Admin.project.delete()`

Deletes a project from the organization. Projects can only be deleted if they are empty (no indexes, collections, or backups). The method provides optional flags to automatically delete these resources before removing the project.

**Source:** `pinecone/admin/resources/project.py:388-505`

**Added:** v8.0
**Deprecated:** No
**Idempotency:** Non-idempotent — the first call succeeds and deletes the project; subsequent calls raise `NotFoundException`
**Side effects:** Deletes the project and optionally all associated indexes, collections, and backups

### Signature

```python
def delete(
    self,
    project_id: str,
    delete_all_indexes: bool = False,
    delete_all_collections: bool = False,
    delete_all_backups: bool = False
) -> None:
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `project_id` | `str` | Yes | — | v8.0 | No | The unique identifier of the project to delete. |
| `delete_all_indexes` | `bool` | No | `False` | v8.0 | No | If `True`, automatically delete all indexes in the project before deleting the project. If `False` and the project contains indexes, the delete operation fails with an error. **This deletion is permanent.** |
| `delete_all_collections` | `bool` | No | `False` | v8.0 | No | If `True`, automatically delete all collections in the project before deleting the project. If `False` and the project contains collections, the delete operation fails with an error. **This deletion is permanent.** |
| `delete_all_backups` | `bool` | No | `False` | v8.0 | No | If `True`, automatically delete all backups in the project before deleting the project. If `False` and the project contains backups, the delete operation fails with an error. **This deletion is permanent.** |

### Returns

**Type:** `None`

### Raises

| Exception | Condition |
|-----------|-----------|
| `NotFoundException` | The project with the given `project_id` does not exist, or the project has already been deleted. |
| `UnauthorizedException` | The API credentials are invalid or missing. |
| `ForbiddenException` | The API key lacks the required permissions to delete projects. |
| `PineconeException` | The project contains indexes, collections, or backups and none of the `delete_all_*` flags are set to `True`. Error message lists the resources that must be deleted first. |

### Example

```python
from pinecone import Admin

admin = Admin(client_id="your-client-id", client_secret="your-client-secret")

project = admin.project.get(name="my-project")

# Delete an empty project
admin.project.delete(project_id=project.id)
print(f"Project {project.id} deleted")

# Delete a project with all its resources
admin.project.delete(
    project_id=project.id,
    delete_all_indexes=True,
    delete_all_collections=True,
    delete_all_backups=True
)

# Verify deletion
if not admin.project.exists(project_id=project.id):
    print("Project deletion completed successfully")
else:
    print("Project deletion failed")
```

### Notes

- **WARNING: This operation is permanent and cannot be undone.** Deleting a project destroys all associated resources.
- When using the `delete_all_*` flags, the method performs a cascade delete: it first deletes all matching resources, then deletes the project itself.
- If cascade delete encounters transient failures (e.g., network timeouts), the method will retry up to 5 times with 30-second delays between retries before giving up.

---

## Data Models

### `Project`

Represents a project in the Pinecone organization.

**Source:** `pinecone/core/openapi/admin/model/project.py:36-328`

| Field | Type | Nullable | Since | Deprecated | Description |
|-------|------|----------|-------|------------|-------------|
| `id` | `string (uuid)` | No | v8.0 | No | The unique identifier of the project, assigned by the server. |
| `name` | `string (1-512 chars)` | No | v8.0 | No | The name of the project. |
| `max_pods` | `integer (int32)` | No | v8.0 | No | The maximum number of pods that can be created in the project. |
| `force_encryption_with_cmek` | `boolean` | No | v8.0 | No | Whether encryption with a customer-managed encryption key (CMEK) is enforced for this project. |
| `organization_id` | `string (uuid)` | No | v8.0 | No | The unique identifier of the organization that owns this project. |
| `created_at` | `datetime` | No | v8.0 | No | The date and time when the project was created, in ISO 8601 format. |

**Example**

```python
from pinecone import Admin

admin = Admin(client_id="your-client-id", client_secret="your-client-secret")

# Fetch a project and access its fields
project = admin.project.get(name="my-project")

print(f"Project ID: {project.id}")
print(f"Project Name: {project.name}")
print(f"Organization ID: {project.organization_id}")
print(f"Max Pods: {project.max_pods}")
print(f"CMEK Encryption: {project.force_encryption_with_cmek}")
print(f"Created At: {project.created_at}")
```

### `ProjectList`

Represents a paginated list of projects.

**Source:** `pinecone/core/openapi/admin/model/project_list.py:47-300`

| Field | Type | Nullable | Since | Deprecated | Description |
|-------|------|----------|-------|------------|-------------|
| `data` | `array of Project` | No | v8.0 | No | The list of projects in the organization. |

**Example**

```python
from pinecone import Admin

admin = Admin(client_id="your-client-id", client_secret="your-client-secret")

# List all projects and iterate through the results
projects_response = admin.projects.list()

print(f"Total projects: {len(projects_response.data)}")

for project in projects_response.data:
    print(f"  - {project.name} (ID: {project.id})")
    print(f"    Organization: {project.organization_id}")
    print(f"    Max pods: {project.max_pods}")
```
