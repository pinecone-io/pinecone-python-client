# Admin Organization Operations

Documents organization management operations available through the Admin client: list, fetch, update, and delete organizations in your Pinecone account.

**Source:** `pinecone/admin/resources/organization.py:10-232`, `pinecone/admin/admin.py:287-344`

## Overview

**Language / runtime:** Python 3.8+
**Package:** `pinecone`
**Module:** `pinecone.admin`
**Class:** `Admin`
**Sub-resource:** `admin.organization` (or `admin.organizations` — alias)
**Breaking change definition:** Changing the return type or return value structure of any method, removing a method, or renaming a parameter.

The `OrganizationResource` class provides a complete organization management interface. Access this class through the `Admin` client's `organization` or `organizations` property:

```python
from pinecone import Admin

admin = Admin()
organizations = admin.organization.list()
```

All methods require keyword arguments. The class automatically handles requests and responses using the underlying OpenAPI client.

## Methods

### `Admin.organization.list() -> OrganizationList`

List all organizations associated with the account.

**Source:** `pinecone/admin/resources/organization.py:40-69`

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| (none) | — | — | — | This method takes no parameters. |

| Return | Type | Description |
|--------|------|-------------|
| `data` | `list[Organization]` | A list of Organization objects for all organizations accessible to the authenticated service account. Each contains `id`, `name`, `plan`, `payment_status`, `created_at`, and `support_tier`. |

| Raises | Condition |
|--------|-----------|
| `UnauthorizedException` | Raised when client credentials (client_id/client_secret) are invalid or expired. |
| `PineconeApiException` | Raised when the organization API service is unavailable. |

**Example:**

```python
from pinecone import Admin

admin = Admin()

# List all organizations
organizations_response = admin.organization.list()
for org in organizations_response.data:
    print(f"Organization: {org.name}")
    print(f"  ID: {org.id}")
    print(f"  Plan: {org.plan}")
    print(f"  Payment Status: {org.payment_status}")
    print(f"  Support Tier: {org.support_tier}")
    print(f"  Created: {org.created_at}")
```

### `Admin.organization.fetch(organization_id: str) -> Organization`

Retrieve details about a specific organization by its ID.

**Source:** `pinecone/admin/resources/organization.py:72-105`

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `organization_id` | `str` | Yes | — | The unique identifier of the organization to retrieve. |

| Return | Type | Description |
|--------|------|-------------|
| — | `Organization` | The organization object with `id`, `name`, `plan`, `payment_status`, `created_at`, and `support_tier`. |

| Raises | Condition |
|--------|-----------|
| `NotFoundException` | Raised when the organization ID does not exist. |
| `UnauthorizedException` | Raised when client credentials are invalid or the service account lacks permission to access this organization. |

**Example:**

```python
from pinecone import Admin

admin = Admin()

# Fetch a specific organization
organization = admin.organization.fetch(
    organization_id="42ca341d-43bf-47cb-9f27-e645dbfabea6"
)
print(f"Organization: {organization.name}")
print(f"  Plan: {organization.plan}")
print(f"  Created: {organization.created_at}")
```

### `Admin.organization.get(organization_id: str) -> Organization`

Alias for `fetch()`. Retrieves an organization by its ID with identical behavior.

**Source:** `pinecone/admin/resources/organization.py:108-131`

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `organization_id` | `str` | Yes | — | The unique identifier of the organization to get. |

| Return | Type | Description |
|--------|------|-------------|
| — | `Organization` | The organization object with `id`, `name`, `plan`, `payment_status`, `created_at`, and `support_tier`. |

| Raises | Condition |
|--------|-----------|
| `NotFoundException` | Raised when the organization ID does not exist. |
| `UnauthorizedException` | Raised when client credentials are invalid or the service account lacks permission to access this organization. |

**Example:**

```python
from pinecone import Admin

admin = Admin()

# Get an organization by ID
organization = admin.organization.get(
    organization_id="42ca341d-43bf-47cb-9f27-e645dbfabea6"
)
print(f"Organization: {organization.name}")
```

### `Admin.organization.describe(organization_id: str) -> Organization`

Alias for `fetch()`. Describes an organization by its ID with identical behavior.

**Source:** `pinecone/admin/resources/organization.py:134-157`

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `organization_id` | `str` | Yes | — | The unique identifier of the organization to describe. |

| Return | Type | Description |
|--------|------|-------------|
| — | `Organization` | The organization object with `id`, `name`, `plan`, `payment_status`, `created_at`, and `support_tier`. |

| Raises | Condition |
|--------|-----------|
| `NotFoundException` | Raised when the organization ID does not exist. |
| `UnauthorizedException` | Raised when client credentials are invalid or the service account lacks permission to access this organization. |

**Example:**

```python
from pinecone import Admin

admin = Admin()

# Describe an organization by ID
organization = admin.organization.describe(
    organization_id="42ca341d-43bf-47cb-9f27-e645dbfabea6"
)
print(f"Organization: {organization.name}")
print(f"  Support Tier: {organization.support_tier}")
```

### `Admin.organization.update(organization_id: str, name: str | None = None) -> Organization`

Update an organization's properties.

**Source:** `pinecone/admin/resources/organization.py:160-195`

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `organization_id` | `str` | Yes | — | The unique identifier of the organization to update. |
| `name` | `str` | No | `None` | The new display name for the organization. Must be 1–512 characters. If omitted, the name is not updated. |

| Return | Type | Description |
|--------|------|-------------|
| — | `Organization` | The updated organization object with `id`, `name`, `plan`, `payment_status`, `created_at`, and `support_tier`. |

| Raises | Condition |
|--------|-----------|
| `ValueError` | Raised when `name` is empty or exceeds 512 characters. |
| `NotFoundException` | Raised when the organization ID does not exist. |
| `UnauthorizedException` | Raised when client credentials are invalid or the service account lacks permission to update this organization. |

**Example:**

```python
from pinecone import Admin

admin = Admin()

# Update an organization's name
organization = admin.organization.update(
    organization_id="42ca341d-43bf-47cb-9f27-e645dbfabea6",
    name="updated-organization-name"
)
print(f"Updated organization name: {organization.name}")
```

### `Admin.organization.delete(organization_id: str) -> None`

Delete an organization and all its associated configuration.

**Source:** `pinecone/admin/resources/organization.py:198-232`

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `organization_id` | `str` | Yes | — | The unique identifier of the organization to delete. |

| Return | Type | Description |
|--------|------|-------------|
| — | `None` | No value is returned. |

| Raises | Condition |
|--------|-----------|
| `BadRequestException` | Raised when the organization still contains active projects, indexes, backups, or collections that must be deleted first. |
| `UnauthorizedException` | Raised when client credentials are invalid or the service account lacks permission to delete this organization. |

**Example:**

```python
from pinecone import Admin

admin = Admin()

# Delete an organization
admin.organization.delete(
    organization_id="42ca341d-43bf-47cb-9f27-e645dbfabea6"
)
print("Organization deleted successfully")
```

## Data Models

### Organization

Response model returned by `fetch()`, `get()`, `describe()`, `update()`, and `list()` operations.

**Source:** `pinecone/core/openapi/admin/model/organization.py`

| Field | Type | Nullable | Description |
|-------|------|----------|-------------|
| `id` | `string` | No | The unique identifier of the organization. |
| `name` | `string` | No | The display name of the organization. Must be 1–512 characters. |
| `plan` | `string` | No | The subscription plan type (e.g., `starter`, `pro`, `enterprise`). |
| `payment_status` | `string` | No | Current payment status of the organization (e.g., `active`, `overdue`, `unpaid`). |
| `created_at` | `string (date-time)` | No | ISO 8601 timestamp when the organization was created. |
| `support_tier` | `string` | No | The support tier level (e.g., `free`, `pro`, `enterprise`). |

### OrganizationList

Response model returned by the `list()` operation. Wraps a list of organizations.

**Source:** `pinecone/core/openapi/admin/model/organization_list.py`

| Field | Type | Nullable | Description |
|-------|------|----------|-------------|
| `data` | `array of Organization` | No | Array of all organizations accessible to the authenticated service account. Empty if no organizations exist. |

### UpdateOrganizationRequest

Request model for the `update()` operation.

**Source:** `pinecone/core/openapi/admin/model/update_organization_request.py`

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `name` | `string` | No | 1–512 characters | The new display name for the organization. Must be between 1 and 512 characters. |
