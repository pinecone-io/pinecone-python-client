AsyncIndex
==========

``AsyncIndex`` is the asynchronous twin of :class:`~pinecone.index.Index`: the
same data-plane methods, each an ``async def``. Use it inside an ``async with``
block. Get one by awaiting :meth:`pinecone.AsyncPinecone.index`, which resolves
the host for you:

.. code-block:: python

   from pinecone import AsyncPinecone

   pc = AsyncPinecone(api_key="your-api-key")

   async with await pc.index("my-index") as idx:
       stats = await idx.describe_index_stats()

Or construct one directly with the index host and an API key, no client needed:

.. code-block:: python

   from pinecone import AsyncIndex

   async with AsyncIndex(
       host="my-index-abc123.svc.pinecone.io",
       api_key="your-api-key",
   ) as idx:
       stats = await idx.describe_index_stats()

Leaving the ``async with`` block, or calling
:meth:`~pinecone.async_client.async_index.AsyncIndex.close`, releases the HTTP
connection pool.

**Method groups:**

- **Documents**: :attr:`~pinecone.async_client.async_index.AsyncIndex.documents`, an
  :class:`~pinecone.async_client.documents.AsyncDocuments` namespace for
  schema-based indexes (``upsert``, ``search``, ``fetch``, ``delete``, ``update``,
  ``list``, ``batch_upsert``). See `AsyncDocuments`_ below.
- **Vectors**: :meth:`~pinecone.async_client.async_index.AsyncIndex.upsert`,
  :meth:`~pinecone.async_client.async_index.AsyncIndex.upsert_from_dataframe`,
  :meth:`~pinecone.async_client.async_index.AsyncIndex.upsert_records`,
  :meth:`~pinecone.async_client.async_index.AsyncIndex.query`,
  :meth:`~pinecone.async_client.async_index.AsyncIndex.query_namespaces`,
  :meth:`~pinecone.async_client.async_index.AsyncIndex.fetch`,
  :meth:`~pinecone.async_client.async_index.AsyncIndex.fetch_by_metadata`,
  :meth:`~pinecone.async_client.async_index.AsyncIndex.update`,
  :meth:`~pinecone.async_client.async_index.AsyncIndex.delete`,
  :meth:`~pinecone.async_client.async_index.AsyncIndex.list`,
  :meth:`~pinecone.async_client.async_index.AsyncIndex.list_paginated`
- **Stats**: :meth:`~pinecone.async_client.async_index.AsyncIndex.describe_index_stats`
- **Legacy Records API** (legacy integrated indexes): :meth:`~pinecone.async_client.async_index.AsyncIndex.search`,
  :meth:`~pinecone.async_client.async_index.AsyncIndex.search_records`
- **Namespaces**: :meth:`~pinecone.async_client.async_index.AsyncIndex.create_namespace`,
  :meth:`~pinecone.async_client.async_index.AsyncIndex.describe_namespace`,
  :meth:`~pinecone.async_client.async_index.AsyncIndex.delete_namespace`,
  :meth:`~pinecone.async_client.async_index.AsyncIndex.list_namespaces`,
  :meth:`~pinecone.async_client.async_index.AsyncIndex.list_namespaces_paginated`
- **Bulk import**: :meth:`~pinecone.async_client.async_index.AsyncIndex.start_import`,
  :meth:`~pinecone.async_client.async_index.AsyncIndex.describe_import`,
  :meth:`~pinecone.async_client.async_index.AsyncIndex.cancel_import`,
  :meth:`~pinecone.async_client.async_index.AsyncIndex.list_imports`,
  :meth:`~pinecone.async_client.async_index.AsyncIndex.list_imports_paginated`
- **Lifecycle**: :attr:`~pinecone.async_client.async_index.AsyncIndex.host`,
  :meth:`~pinecone.async_client.async_index.AsyncIndex.close`

.. autoclass:: pinecone.async_client.async_index.AsyncIndex
   :members:
   :undoc-members: False
   :show-inheritance:
   :special-members: __init__, __aenter__, __aexit__
   :member-order: bysource


AsyncDocuments
--------------

.. autoclass:: pinecone.async_client.documents.AsyncDocuments
   :members:
   :undoc-members: False
   :show-inheritance:
