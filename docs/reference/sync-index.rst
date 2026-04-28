Index
=====

Obtain an ``Index`` instance via :meth:`pinecone.Pinecone.index`.

.. code-block:: python

   from pinecone import Pinecone

   pc = Pinecone(api_key="your-api-key")

   # Resolve host automatically by index name
   idx = pc.index("my-index")

   # — or — connect directly with a host URL
   idx = pc.index(host="my-index-abc123.svc.pinecone.io")

**Method groups:**

- **Vectors** — :meth:`~pinecone.index.Index.upsert`,
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
- **Stats** — :meth:`~pinecone.index.Index.describe_index_stats`
- **Integrated Inference** — :meth:`~pinecone.index.Index.search`,
  :meth:`~pinecone.index.Index.search_records`
- **Namespaces** — :meth:`~pinecone.index.Index.create_namespace`,
  :meth:`~pinecone.index.Index.describe_namespace`,
  :meth:`~pinecone.index.Index.delete_namespace`,
  :meth:`~pinecone.index.Index.list_namespaces`,
  :meth:`~pinecone.index.Index.list_namespaces_paginated`
- **Bulk Import** — :meth:`~pinecone.index.Index.start_import`,
  :meth:`~pinecone.index.Index.describe_import`,
  :meth:`~pinecone.index.Index.cancel_import`,
  :meth:`~pinecone.index.Index.list_imports`,
  :meth:`~pinecone.index.Index.list_imports_paginated`
- **Lifecycle** — :meth:`~pinecone.index.Index.close`

.. autoclass:: pinecone.index.Index
   :members:
   :undoc-members: False
   :show-inheritance:
   :special-members: __init__, __enter__, __exit__
   :member-order: bysource
