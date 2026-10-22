GrpcIndex
=========

``GrpcIndex`` runs the data-plane methods of :class:`~pinecone.index.Index` over
gRPC, and adds ``*_async()`` methods that
return a :class:`~pinecone.grpc.future.PineconeFuture`. It has no ``documents``
namespace: use :class:`~pinecone.index.Index` or
:class:`~pinecone.async_client.async_index.AsyncIndex` for a schema-based index.
Get one from :meth:`pinecone.Pinecone.index` with ``grpc=True``, or construct one
directly.

.. code-block:: python

   from pinecone import Pinecone

   pc = Pinecone(api_key="your-api-key")

   # Resolve host automatically by index name
   idx = pc.index("my-index", grpc=True)

   # Or construct directly with a host URL
   from pinecone.grpc import GrpcIndex
   idx = GrpcIndex(host="my-index-abc123.svc.pinecone.io", api_key="your-api-key")

**Method groups:**

- **Vectors**: :meth:`~pinecone.grpc.GrpcIndex.upsert`,
  :meth:`~pinecone.grpc.GrpcIndex.upsert_from_dataframe`,
  :meth:`~pinecone.grpc.GrpcIndex.upsert_records`,
  :meth:`~pinecone.grpc.GrpcIndex.query`,
  :meth:`~pinecone.grpc.GrpcIndex.query_namespaces`,
  :meth:`~pinecone.grpc.GrpcIndex.fetch`,
  :meth:`~pinecone.grpc.GrpcIndex.fetch_by_metadata`,
  :meth:`~pinecone.grpc.GrpcIndex.update`,
  :meth:`~pinecone.grpc.GrpcIndex.delete`,
  :meth:`~pinecone.grpc.GrpcIndex.list`,
  :meth:`~pinecone.grpc.GrpcIndex.list_paginated`
- **Stats**: :meth:`~pinecone.grpc.GrpcIndex.describe_index_stats`
- **Legacy Records API** (legacy integrated indexes): :meth:`~pinecone.grpc.GrpcIndex.search`,
  :meth:`~pinecone.grpc.GrpcIndex.search_records`
- **Namespaces**: :meth:`~pinecone.grpc.GrpcIndex.create_namespace`,
  :meth:`~pinecone.grpc.GrpcIndex.describe_namespace`,
  :meth:`~pinecone.grpc.GrpcIndex.delete_namespace`,
  :meth:`~pinecone.grpc.GrpcIndex.list_namespaces`,
  :meth:`~pinecone.grpc.GrpcIndex.list_namespaces_paginated`
- **Bulk import**: :meth:`~pinecone.grpc.GrpcIndex.start_import`,
  :meth:`~pinecone.grpc.GrpcIndex.describe_import`,
  :meth:`~pinecone.grpc.GrpcIndex.cancel_import`,
  :meth:`~pinecone.grpc.GrpcIndex.list_imports`,
  :meth:`~pinecone.grpc.GrpcIndex.list_imports_paginated`
- **Async variants**: :meth:`~pinecone.grpc.GrpcIndex.upsert_async`,
  :meth:`~pinecone.grpc.GrpcIndex.query_async`,
  :meth:`~pinecone.grpc.GrpcIndex.query_namespaces_async`,
  :meth:`~pinecone.grpc.GrpcIndex.fetch_async`,
  :meth:`~pinecone.grpc.GrpcIndex.update_async`,
  :meth:`~pinecone.grpc.GrpcIndex.delete_async`
- **Lifecycle**: :attr:`~pinecone.grpc.GrpcIndex.host`,
  :meth:`~pinecone.grpc.GrpcIndex.close`

.. autoclass:: pinecone.grpc.GrpcIndex
   :members:
   :undoc-members: False
   :show-inheritance:
   :special-members: __init__, __enter__, __exit__
   :member-order: bysource

PineconeFuture
--------------

The ``*_async()`` methods on :class:`~pinecone.grpc.GrpcIndex` return a
:class:`~pinecone.grpc.future.PineconeFuture`, which works with
:func:`concurrent.futures.as_completed` and :func:`concurrent.futures.wait`.

.. autoclass:: pinecone.grpc.future.PineconeFuture
   :members:
   :undoc-members: False
   :show-inheritance:
