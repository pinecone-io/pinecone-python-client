AsyncPinecone
=============

:class:`AsyncPinecone` is the asynchronous control-plane client — use it inside an
``async with`` block to manage indexes, collections, backups, and related resources.
Sub-clients for each resource type are accessed as namespace properties (e.g.
``pc.indexes``, ``pc.collections``) and are lazily initialised on first access.

.. code-block:: python

   from pinecone import AsyncPinecone

   async with AsyncPinecone(api_key="your-api-key") as pc:
       desc = await pc.indexes.describe("my-index")
       index = pc.index(host=desc.host)
       async with index:
           results = await index.query(
               vector=[0.012, -0.087, 0.153],
               top_k=10,
           )

.. note::

   Unlike :class:`~pinecone.Pinecone`, ``AsyncPinecone.index()`` cannot auto-resolve an
   index host by name.  Call ``await pc.indexes.describe(name)`` first to populate the
   cache, then create the data-plane client::

       desc = await pc.indexes.describe("my-index")
       idx = pc.index("my-index")          # uses cached host
       # — or —
       idx = pc.index(host=desc.host)       # explicit host

.. autoclass:: pinecone.async_client.pinecone.AsyncPinecone
   :members:
   :undoc-members: False
   :show-inheritance:
   :special-members: __init__, __aenter__, __aexit__
   :canonical: pinecone.AsyncPinecone


AsyncIndexes Namespace
----------------------

.. autoclass:: pinecone.async_client.indexes.AsyncIndexes
   :members:
   :undoc-members: False
   :show-inheritance:


AsyncCollections Namespace
--------------------------

.. autoclass:: pinecone.async_client.collections.AsyncCollections
   :members:
   :undoc-members: False
   :show-inheritance:


AsyncBackups Namespace
----------------------

.. autoclass:: pinecone.async_client.backups.AsyncBackups
   :members:
   :undoc-members: False
   :show-inheritance:


AsyncRestoreJobs Namespace
--------------------------

.. autoclass:: pinecone.async_client.restore_jobs.AsyncRestoreJobs
   :members:
   :undoc-members: False
   :show-inheritance:


AsyncInference Namespace
------------------------

.. autoclass:: pinecone.async_client.inference.AsyncInference
   :members:
   :undoc-members: False
   :show-inheritance:


AsyncAssistants Namespace
-------------------------

.. autoclass:: pinecone.async_client.assistants.AsyncAssistants
   :members:
   :undoc-members: False
   :show-inheritance:
