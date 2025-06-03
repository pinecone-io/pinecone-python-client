========
Pinecone
========

.. autoclass:: pinecone::Pinecone

.. automethod:: pinecone::Pinecone.__init__

.. automethod:: pinecone::Pinecone.Index

.. automethod:: pinecone::Pinecone.IndexAsyncio

DB Control Plane
================

Indexes
-------

.. automethod:: pinecone::Pinecone.create_index

.. automethod:: pinecone::Pinecone.create_index_for_model

.. automethod:: pinecone::Pinecone.create_index_from_backup

.. automethod:: pinecone::Pinecone.list_indexes

.. automethod:: pinecone::Pinecone.describe_index

.. automethod:: pinecone::Pinecone.configure_index

.. automethod:: pinecone::Pinecone.delete_index

.. automethod:: pinecone::Pinecone.has_index

Backups
-------

.. automethod:: pinecone::Pinecone.create_backup

.. automethod:: pinecone::Pinecone.list_backups

.. automethod:: pinecone::Pinecone.describe_backup

.. automethod:: pinecone::Pinecone.delete_backup

Collections
-----------

.. automethod:: pinecone::Pinecone.create_collection

.. automethod:: pinecone::Pinecone.list_collections

.. automethod:: pinecone::Pinecone.describe_collection

.. automethod:: pinecone::Pinecone.delete_collection

Restore Jobs
------------

.. automethod:: pinecone::Pinecone.list_restore_jobs

.. automethod:: pinecone::Pinecone.describe_restore_job

DB Data Plane
=============

.. autoclass:: pinecone.db_data::Index

.. automethod:: pinecone.db_data::Index.__init__

.. automethod:: pinecone.db_data::Index.describe_index_stats

Vectors
-------

.. automethod:: pinecone.db_data::Index.upsert

.. automethod:: pinecone.db_data::Index.query

.. automethod:: pinecone.db_data::Index.query_namespaces

.. automethod:: pinecone.db_data::Index.delete

.. automethod:: pinecone.db_data::Index.fetch

.. automethod:: pinecone.db_data::Index.list

.. automethod:: pinecone.db_data::Index.list_paginated


Bulk Import
-----------

.. automethod:: pinecone.db_data::Index.start_import

.. automethod:: pinecone.db_data::Index.list_imports

.. automethod:: pinecone.db_data::Index.list_imports_paginated

.. automethod:: pinecone.db_data::Index.describe_import

.. automethod:: pinecone.db_data::Index.cancel_import


Records
-------

If you have created an index using integrated inference, you can use the following methods to
search and retrieve records.

.. automethod:: pinecone.db_data::Index.search

.. automethod:: pinecone.db_data::Index.search_records



Inference
=========

.. automethod:: pinecone.inference::Inference.embed

.. automethod:: pinecone.inference::Inference.rerank

.. automethod:: pinecone.inference::Inference.list_models

.. automethod:: pinecone.inference::Inference.get_model
