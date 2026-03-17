# Admin Organization Operations

This module documents organization management operations on the Admin client: listing, retrieving, updating, and deleting organizations within a Pinecone account. All operations require a service account and provide account-scoped control plane access to the organization lifecycle.

## Overview

**Language / runtime:** Python 3.8+
**Package:** `pinecone`
**Module:** `pinecone.admin`
**Class:** `Admin`
**Sub-resource:** `admin.organization` (or `admin.organizations` — alias)
**Version:** v8.1.0
**Breaking change definition:** Changing the return type or return value structure of any method, removing a method, or renaming a parameter.

## Access Pattern

Organization operations are accessed through the Admin client's `organization` or `organizations` property:

```python
from pinecone import Admin

admin = Admin(client_id="your-client-id", client_secret="your-client-secret")

# Both are equivalent
admin.organization.list()
admin.organizations.list()
```

## Methods

### `Admin.organization.list() -> OrganizationList`

Lists all organizations associated with the account.

**Import:** `from pinecone import Admin`
**Source:** `pinecone/admin/resources/organization.py:40-69`
**Added:** v8.0
**Deprecated:** No
**Idempotency:** Safe to retry
**Side effects:** None

**Returns:** `OrganizationList` — A response object with a `data` field containing a list of all organizations associated with the account. Each organization in the list is an `Organization` instance.

**Raises / Throws**

| Exception / Error | Condition |
|-------------------|-----------|
| `PineconeApiException` | An unexpected error occurred while retrieving the organization list. |
| `UnauthorizedException` | The service account credentials are invalid or missing. |
| `ForbiddenException` | The service account credentials lack the required permissions to list organizations. |

**Example**

```python
from pinecone import Admin

admin = Admin(client_id="your-client-id", client_secret="your-client-secret")

# List all organizations
organizations_response = admin.organization.list()
for organization in organizations_response.data:
    print(f"Organization ID: {organization.id}")
    print(f"Organization Name: {organization.name}")
    print(f"Plan: {organization.plan}")
    print(f"Payment Status: {organization.payment_status}")
```

---

### `Admin.organization.fetch(organization_id: str) -> Organization`

Retrieves a single organization by ID.

**Import:** `from pinecone import Admin`
**Source:** `pinecone/admin/resources/organization.py:72-105`
**Added:** v8.0
**Deprecated:** No
**Idempotency:** Safe to retry
**Side effects:** None

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| organization_id | string | Yes | — | The unique ID of the organization to retrieve. Obtain this via `admin.organization.list()`. |

**Returns:** `Organization` — The requested organization object containing its ID, name, plan, payment status, creation timestamp, and support tier.

**Raises / Throws**

| Exception / Error | Condition |
|-------------------|-----------|
| `PineconeApiException` | An unexpected error occurred while retrieving the organization. |
| `UnauthorizedException` | The service account credentials are invalid or missing. |
| `ForbiddenException` | The service account credentials lack the required permissions to describe organizations. |
| `NotFoundException` | The organization with the specified ID does not exist. |

**Example**

```python
from pinecone import Admin

admin = Admin(client_id="your-client-id", client_secret="your-client-secret")

organization = admin.organization.fetch(organization_id="42ca341d-43bf-47cb-9f27-e645dbfabea6")
print(f"Organization: {organization.name}")
print(f"Plan: {organization.plan}")
print(f"Created: {organization.created_at}")
print(f"Support Tier: {organization.support_tier}")
```

---

### `Admin.organization.get(organization_id: str) -> Organization`

Alias for `fetch()`. Retrieves a single organization by ID.

**Import:** `from pinecone import Admin`
**Source:** `pinecone/admin/resources/organization.py:108-131`
**Added:** v8.0
**Deprecated:** No
**Idempotency:** Safe to retry
**Side effects:** None

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| organization_id | string | Yes | — | The unique ID of the organization to retrieve. |

**Returns:** `Organization` — The requested organization object.

**Raises / Throws**

| Exception / Error | Condition |
|-------------------|-----------|
| `PineconeApiException` | An unexpected error occurred while retrieving the organization. |
| `UnauthorizedException` | The service account credentials are invalid or missing. |
| `ForbiddenException` | The service account credentials lack the required permissions to describe organizations. |
| `NotFoundException` | The organization with the specified ID does not exist. |

**Example**

```python
from pinecone import Admin

admin = Admin(client_id="your-client-id", client_secret="your-client-secret")

organization = admin.organization.get(organization_id="42ca341d-43bf-47cb-9f27-e645dbfabea6")
print(f"Organization: {organization.name}")
```

**Notes**

- This method is functionally identical to `fetch()`. Use whichever naming convention you prefer.

---

### `Admin.organization.describe(organization_id: str) -> Organization`

Alias for `fetch()`. Retrieves a single organization by ID.

**Import:** `from pinecone import Admin`
**Source:** `pinecone/admin/resources/organization.py:134-157`
**Added:** v8.0
**Deprecated:** No
**Idempotency:** Safe to retry
**Side effects:** None

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| organization_id | string | Yes | — | The unique ID of the organization to describe. |

**Returns:** `Organization` — The requested organization object.

**Raises / Throws**

| Exception / Error | Condition |
|-------------------|-----------|
| `PineconeApiException` | An unexpected error occurred while retrieving the organization. |
| `UnauthorizedException` | The service account credentials are invalid or missing. |
| `ForbiddenException` | The service account credentials lack the required permissions to describe organizations. |
| `NotFoundException` | The organization with the specified ID does not exist. |

**Example**

```python
from pinecone import Admin

admin = Admin(client_id="your-client-id", client_secret="your-client-secret")

organization = admin.organization.describe(organization_id="42ca341d-43bf-47cb-9f27-e645dbfabea6")
print(f"Organization: {organization.name}")
```

**Notes**

- This method is functionally identical to `fetch()`. Use whichever naming convention you prefer.

---

### `Admin.organization.update(organization_id: str, name: str | None = None) -> Organization`

Updates an organization's details.

**Import:** `from pinecone import Admin`
**Source:** `pinecone/admin/resources/organization.py:160-195`
**Added:** v8.0
**Deprecated:** No
**Idempotency:** Not idempotent — if the exact request is retried, it may produce different results if the organization state changed between calls.
**Side effects:** Modifies the organization resource in the Pinecone API. At minimum, persists changes to the organization name.

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| organization_id | string | Yes | — | The unique ID of the organization to update. Obtain this via `admin.organization.list()` or `admin.organization.get()`. |
| name | string \| None | No | None | The new name for the organization. Must be 1–512 characters long. When omitted or `None`, the name will not be updated. |

**Returns:** `Organization` — The updated organization object with its new state.

**Raises / Throws**

| Exception / Error | Condition |
|-------------------|-----------|
| `PineconeApiException` | The update failed. Verify that `name` is between 1–512 characters long. |
| `UnauthorizedException` | The service account credentials are invalid or missing. |
| `ForbiddenException` | The service account credentials lack the required permissions to update organizations. |
| `NotFoundException` | The organization with the specified ID does not exist. |

**Example**

```python
from pinecone import Admin

admin = Admin(client_id="your-client-id", client_secret="your-client-secret")

organization = admin.organization.update(
    organization_id="42ca341d-43bf-47cb-9f27-e645dbfabea6",
    name="updated-organization-name"
)
print(f"Updated organization name: {organization.name}")
```

**Notes**

- Only the `name` field can be updated through this method. Other organization properties (ID, plan, payment status, creation timestamp, support tier) are managed by Pinecone and cannot be changed.
- Passing `None` for `name` (the default) skips the name update entirely — it does not clear the name.

---

### `Admin.organization.delete(organization_id: str) -> None`

Deletes an organization permanently.

**Import:** `from pinecone import Admin`
**Source:** `pinecone/admin/resources/organization.py:198-232`
**Added:** v8.0
**Deprecated:** No
**Idempotency:** Idempotent — deleting a non-existent organization returns success (no error).
**Side effects:** Permanently deletes the organization resource and all associated data from the Pinecone API. This operation is irreversible. All projects, indexes, assistants, backups, and collections within the organization must be deleted before the organization itself can be deleted.

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| organization_id | string | Yes | — | The unique ID of the organization to delete. Obtain this via `admin.organization.list()` or `admin.organization.get()`. |

**Returns:** `None` — This method returns nothing on success.

**Raises / Throws**

| Exception / Error | Condition |
|-------------------|-----------|
| `PineconeApiException` | The organization cannot be deleted because it still contains associated projects, indexes, assistants, backups, or collections. Delete these first. |
| `UnauthorizedException` | The service account credentials are invalid or missing. |
| `ForbiddenException` | The service account credentials lack the required permissions to delete organizations. |

**Example**

```python
from pinecone import Admin

admin = Admin(client_id="your-client-id", client_secret="your-client-secret")

# Delete an organization
admin.organization.delete(organization_id="42ca341d-43bf-47cb-9f27-e645dbfabea6")
print("Organization deleted")
```

**Notes**

- Deleting an organization is a **permanent and irreversible operation**. All data associated with the organization will be lost.
- Before deleting an organization, you must delete all projects (including indexes, assistants, backups, and collections) associated with the organization. If you attempt to delete an organization with associated resources, the deletion will fail with a `PineconeApiException`.
- Use this method with extreme caution.

---

## Types and Models

### `OrganizationList`

A response wrapper containing a list of organizations.

**Source:** `pinecone/core/openapi/admin/model/organization_list.py`

| Field | Type | Nullable | Since | Deprecated | Description |
|-------|------|----------|-------|------------|-------------|
| data | array of Organization | No | v8.0 | No | The list of organizations associated with the account. Each item is an `Organization` object. |

**Example**

```python
from pinecone import Admin

admin = Admin(client_id="your-client-id", client_secret="your-client-secret")

response = admin.organization.list()
# Access the organizations list via the data field
for org in response.data:
    print(f"ID: {org.id}, Name: {org.name}")
```

---

### `Organization`

Represents a single organization in Pinecone.

**Source:** `pinecone/core/openapi/admin/model/organization.py`

| Field | Type | Nullable | Since | Deprecated | Description |
|-------|------|----------|-------|------------|-------------|
| id | string (uuid) | No | v8.0 | No | The unique identifier of the organization. |
| name | string (1–512 chars) | No | v8.0 | No | The name of the organization. Can be updated via `admin.organization.update()`. Must be between 1 and 512 characters in length. |
| plan | string | No | v8.0 | No | The current billing plan the organization is on. |
| payment_status | string | No | v8.0 | No | The current payment status of the organization (e.g., "active", "past_due"). |
| created_at | string (date-time) | No | v8.0 | No | The ISO 8601 timestamp when the organization was created. |
| support_tier | string | No | v8.0 | No | The support tier level of the organization. |
