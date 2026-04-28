Pinecone
========

:class:`Pinecone` is the synchronous control-plane client — use it to manage indexes,
collections, backups, and related resources. Sub-clients for each resource type are
accessed as namespace properties (e.g. ``pc.indexes``, ``pc.collections``) and are
lazily initialized on first access.

.. autoclass:: pinecone.Pinecone
   :members:
   :undoc-members: False
   :show-inheritance:
   :special-members: __init__


Indexes Namespace
-----------------

.. autoclass:: pinecone.client.indexes.Indexes
   :members:
   :undoc-members: False
   :show-inheritance:


Collections Namespace
---------------------

.. autoclass:: pinecone.client.collections.Collections
   :members:
   :undoc-members: False
   :show-inheritance:


Backups Namespace
-----------------

.. autoclass:: pinecone.client.backups.Backups
   :members:
   :undoc-members: False
   :show-inheritance:


RestoreJobs Namespace
---------------------

.. autoclass:: pinecone.client.restore_jobs.RestoreJobs
   :members:
   :undoc-members: False
   :show-inheritance:


Inference Namespace
-------------------

.. autoclass:: pinecone.client.inference.Inference
   :members:
   :undoc-members: False
   :show-inheritance:


Assistants Namespace
--------------------

.. autoclass:: pinecone.client.assistants.Assistants
   :members:
   :undoc-members: False
   :show-inheritance:
