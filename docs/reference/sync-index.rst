Index
=====

``Index`` is the synchronous data-plane client for one index. Which methods apply
depends on how the index was created: a schema-based (document) index is read and
written through :attr:`~pinecone.index.Index.documents`, an index created with
top-level ``dimension``/``metric`` through the vector methods, and a legacy
integrated index through the Legacy Records API. Get one from
:meth:`pinecone.Pinecone.index`.

.. code-block:: python

   from pinecone import Pinecone

   pc = Pinecone(api_key="your-api-key")

   # Resolve host automatically by index name
   idx = pc.index("my-index")

   # Or connect directly with a host URL
   idx = pc.index(host="my-index-abc123.svc.pinecone.io")

**Method groups:**

- **Documents**: :attr:`~pinecone.index.Index.documents`, a
  :class:`~pinecone.client.documents.Documents` namespace for schema-based indexes
  (``upsert``, ``search``, ``fetch``, ``delete``, ``update``, ``list``,
  ``batch_upsert``). See `Documents`_ below.
- **Vectors**: :meth:`~pinecone.index.Index.upsert`,
  :meth:`~pinecone.index.Index.upsert_from_dataframe`,
  :meth:`~pinecone.index.Index.upsert_records`,
  :meth:`~pinecone.index.Index.query`,
  :meth:`~pinecone.index.Index.query_namespaces`,
  :meth:`~pinecone.index.Index.fetch`,
  :meth:`~pinecone.index.Index.fetch_by_metadata`,
  :meth:`~pinecone.index.Index.update`,
  :meth:`~pinecone.index.Index.delete`,
  :meth:`~pinecone.index.Index.list`,
  :meth:`~pinecone.index.Index.list_paginated`
- **Stats**: :meth:`~pinecone.index.Index.describe_index_stats`
- **Legacy Records API** (legacy integrated indexes): :meth:`~pinecone.index.Index.search`,
  :meth:`~pinecone.index.Index.search_records`
- **Namespaces**: :meth:`~pinecone.index.Index.create_namespace`,
  :meth:`~pinecone.index.Index.describe_namespace`,
  :meth:`~pinecone.index.Index.delete_namespace`,
  :meth:`~pinecone.index.Index.list_namespaces`,
  :meth:`~pinecone.index.Index.list_namespaces_paginated`
- **Bulk import**: :meth:`~pinecone.index.Index.start_import`,
  :meth:`~pinecone.index.Index.describe_import`,
  :meth:`~pinecone.index.Index.cancel_import`,
  :meth:`~pinecone.index.Index.list_imports`,
  :meth:`~pinecone.index.Index.list_imports_paginated`
- **Lifecycle**: :attr:`~pinecone.index.Index.host`,
  :meth:`~pinecone.index.Index.close`

.. autoclass:: pinecone.index.Index
   :members:
   :undoc-members: False
   :show-inheritance:
   :special-members: __init__, __enter__, __exit__
   :member-order: bysource


Documents
---------

.. autoclass:: pinecone.client.documents.Documents
   :members:
   :undoc-members: False
   :show-inheritance:
