# Fetch, update, delete, and list documents

Search returns the best matches. The methods on this page work on documents you can name:
by ID, by ID prefix, or by a filter on their fields. The examples use the `article-search`
index from [Declare a schema and upsert documents](schema-and-upsert.md), with `category`
and `year` as metadata.

Updates and deletes apply asynchronously, like upserts: the call returns once the server
accepts the change, and a read made right afterward may still show the old documents.

## Fetch documents

{meth}`~pinecone.client.documents.Documents.fetch` takes either `ids` or `filter`. Fetching
by ID returns `response.documents`, a dict from ID to
{class}`~pinecone.models.documents.document.Document`. An ID that does not exist is left
out rather than raising an error:

```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-api-key")
index = pc.index("article-search")

response = index.documents.fetch(
    namespace="articles-en",
    ids=["article-101", "article-102"],
    include_fields=["title", "year"],
)
for doc_id, doc in response.documents.items():
    print(doc_id, doc.title, doc.year)
```

Unlike search, fetch returns every stored field when you omit `include_fields`.

A fetch by filter returns every matching document a page at a time. Read each page, then
pass `response.pagination.next` back as `pagination_token` until `pagination` is `None`:

```python
pagination_token = None
while True:
    response = index.documents.fetch(
        namespace="articles-en",
        filter={"category": {"$eq": "history"}},
        pagination_token=pagination_token,
    )
    for doc_id, doc in response.documents.items():
        print(doc_id, doc.title)
    if response.pagination is None:
        break
    pagination_token = response.pagination.next
```

Filters for fetch take the same operators as search filters, including the text-match
operators on full-text fields; see [Filter the search](search.md#filter-the-search).

## Update documents

{meth}`~pinecone.client.documents.Documents.update` changes only the fields you name and
keeps the rest. An upsert, by contrast, replaces the whole document, so use `update` to
change one field without resending the vector. Patch documents by ID with `documents`; in
each patch, every key other than `_id` sets a field, and `_remove_fields` lists fields to
drop:

```python
index.documents.update(
    namespace="articles-en",
    documents=[
        {"_id": "article-101", "category": "engineering"},
        {"_id": "article-102", "_remove_fields": ["year"]},
    ],
)
```

To set or remove the same fields on every document that matches a filter, pass `filter`
with `set_fields`, `remove_fields`, or both:

```python
response = index.documents.update(
    namespace="articles-en",
    filter={"year": {"$lt": 2020}},
    set_fields={"category": "archive"},
)
print(response.matched_records)
```

A patch naming an ID that does not exist does nothing. To remove a field, list it in
`_remove_fields` or `remove_fields`; setting it to `None` fails with a `400`. Changing the
text of an `embed` or `sparse_embed` field has Pinecone embed the new text, and
`response.usage` reports the tokens embedded.

## Delete documents

{meth}`~pinecone.client.documents.Documents.delete` takes exactly one of `ids`, `filter`, or
`delete_all=True`. Deleting an ID that does not exist is not an error.

```python
index.documents.delete(namespace="articles-en", ids=["article-101"])

response = index.documents.delete(
    namespace="articles-en",
    filter={"category": {"$eq": "archive"}},
)
print(response.matched_records)

index.documents.delete(namespace="drafts", delete_all=True)
```

The filters of `update` and `delete` reject the text-match operators (`$match_phrase`,
`$match_all`, `$match_any`), so select documents by metadata or by ID. For a filtered
update or delete, `matched_records` counts the documents that matched when the server
accepted the request; it is `None` for the other forms.

## List document IDs

{meth}`~pinecone.client.documents.Documents.list` returns the IDs in a namespace, sorted,
as a {class}`~pinecone.models.pagination.Paginator` that fetches pages as you iterate. Give
your IDs a shared prefix, such as `article-`, and you can list one group of documents
without a filter:

```python
for record in index.documents.list(namespace="articles-en", prefix="article-"):
    print(record.id)
```

Each item carries only `id`. Pass a batch of IDs to `fetch` to read their fields. To
checkpoint a long walk, iterate `.pages()` and save each page's `pagination_token`, then
resume later with `list(..., pagination_token=token)`. See
[Pagination](../../guides/pagination.md) for the paginator's other methods.
