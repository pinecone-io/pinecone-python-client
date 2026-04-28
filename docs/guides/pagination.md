# Pagination

Some SDK operations return results in pages. The SDK provides `Paginator` (sync) and
`AsyncPaginator` (async) to iterate over those pages lazily — only fetching the next
page when you ask for it.

## Paginator and AsyncPaginator

Both classes share the same interface:

| Member | Description |
|--------|-------------|
| `__iter__` / `__aiter__` | Iterate over individual items across all pages |
| `.pages()` | Iterate over `Page` objects rather than individual items |
| `.to_list()` | Fetch all items into a list in one call |
| `.pagination_token` | Token for the next page; `None` when all pages have been consumed |

### Iterating over items

::::{tabs}
:::{tab} Sync
```python
from pinecone import Pinecone

pc = Pinecone()

for assistant in pc.assistants.list():
    print(assistant.name)
```
:::
:::{tab} Async
```python
import asyncio
from pinecone import AsyncPinecone

async def main() -> None:
    async with AsyncPinecone() as pc:
        async for assistant in pc.assistants.list():
            print(assistant.name)

asyncio.run(main())
```
:::
::::

### Iterating page by page

::::{tabs}
:::{tab} Sync
```python
for page in pc.assistants.list().pages():
    print(f"Page has {len(page.items)} items")
    for assistant in page.items:
        print(assistant.name)
    if not page.has_more:
        break
```
:::
:::{tab} Async
```python
async with AsyncPinecone() as pc:
    async for page in pc.assistants.list().pages():
        print(f"Page has {len(page.items)} items")
        for assistant in page.items:
            print(assistant.name)
```
:::
::::

### Collecting all results at once

::::{tabs}
:::{tab} Sync
```python
all_assistants = pc.assistants.list().to_list()
```
:::
:::{tab} Async
```python
async with AsyncPinecone() as pc:
    all_assistants = await pc.assistants.list().to_list()
```
:::
::::

## The Page Object

Each page yielded by `.pages()` has two attributes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `items` | `list[T]` | The items on this page |
| `pagination_token` | `str \| None` | Opaque token to fetch the next page; `None` on the last page |
| `has_more` | `bool` | `True` if more pages are available |

## Limiting Results

Pass `limit` to cap the total number of items returned across all pages:

```python
# Return at most 50 assistants total
for assistant in pc.assistants.list(limit=50):
    print(assistant.name)
```

## Resuming Pagination

Save `pagination_token` to resume iteration later:

```python
paginator = pc.assistants.list()
first_page = next(paginator.pages())

# Store the token
token = first_page.pagination_token

# Later: resume from where you left off
from pinecone.models.pagination import Paginator

# (pass initial_token when constructing directly)
```

## Paginating Vector IDs

`Index.list_paginated()` returns a `ListResponse` for a single page of vector IDs.
Use it directly when you need fine-grained control over pagination tokens:

```python
from pinecone import Pinecone

pc = Pinecone()
desc = pc.indexes.describe("product-search")
index = pc.index(host=desc.host)

# Fetch the first page
page = index.list_paginated(prefix="product#", limit=100)
for item in page.vectors:
    print(item.id)

# Fetch subsequent pages
while page.pagination is not None and page.pagination.next is not None:
    page = index.list_paginated(
        prefix="product#",
        limit=100,
        pagination_token=page.pagination.next,
    )
    for item in page.vectors:
        print(item.id)
```

## Non-Paginated Responses

Not every list operation uses a paginator. `pc.indexes.list()` returns an `IndexList`
directly — it contains all indexes in a single response with no pagination token.

```python
result = pc.indexes.list()
for index in result.indexes:
    print(index.name)
```
