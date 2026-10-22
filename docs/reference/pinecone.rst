Pinecone
========

:class:`Pinecone` is the synchronous control-plane client: it creates and manages
indexes, collections, backups, and assistants, and runs inference. Each resource
type is a namespace property (``pc.indexes``, ``pc.collections``, ...) documented
in its own section below. For reading and writing data, :meth:`~pinecone.Pinecone.index`
returns an :doc:`Index <sync-index>`.

.. autoclass:: pinecone.Pinecone
   :members:
   :undoc-members: False
   :show-inheritance:
   :member-order: bysource
   :special-members: __init__


Indexes
-------

.. autoclass:: pinecone.client.indexes.Indexes
   :members:
   :undoc-members: False
   :show-inheritance:
   :member-order: bysource


Collections
-----------

.. autoclass:: pinecone.client.collections.Collections
   :members:
   :undoc-members: False
   :show-inheritance:
   :member-order: bysource


Backups
-------

.. autoclass:: pinecone.client.backups.Backups
   :members:
   :undoc-members: False
   :show-inheritance:
   :member-order: bysource


BackupSchedules
---------------

.. autoclass:: pinecone.client.backup_schedules.BackupSchedules
   :members:
   :undoc-members: False
   :show-inheritance:
   :member-order: bysource


RestoreJobs
-----------

.. autoclass:: pinecone.client.restore_jobs.RestoreJobs
   :members:
   :undoc-members: False
   :show-inheritance:
   :member-order: bysource


Inference
---------

.. autoclass:: pinecone.client.inference.Inference
   :members:
   :undoc-members: False
   :show-inheritance:
   :member-order: bysource

.. autoclass:: pinecone.client.inference.ModelResource
   :members:
   :undoc-members: False
   :show-inheritance:
   :member-order: bysource


Assistants
----------

.. autoclass:: pinecone.client.assistants.Assistants
   :members:
   :undoc-members: False
   :show-inheritance:
   :member-order: bysource


Retry configuration
-------------------

``RetryConfig`` is a constructor argument on :class:`~pinecone.Pinecone` and
:class:`~pinecone.AsyncPinecone`.  See :doc:`/guides/retries` for which calls it
governs and which run on fixed policy.

.. autoclass:: pinecone.RetryConfig
   :members:
   :undoc-members: False
   :show-inheritance:
   :member-order: bysource


Resolved configuration
----------------------

``pc.config`` returns the :class:`~pinecone.PineconeConfig` a client resolved at
construction, with environment variables and defaults folded in. Read it to
confirm which host and settings a client uses; pass settings to the
:class:`~pinecone.Pinecone` constructor rather than building one yourself.

.. autoclass:: pinecone.PineconeConfig
   :members:
   :undoc-members: False
   :show-inheritance:
   :member-order: bysource
