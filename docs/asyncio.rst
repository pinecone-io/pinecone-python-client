===============
PineconeAsyncio
===============

.. autoclass:: pinecone::PineconeAsyncio

.. automethod:: pinecone::PineconeAsyncio.__init__

DB Control Plane
================

Indexes
-------

.. automethod:: pinecone::PineconeAsyncio.create_index

.. automethod:: pinecone::PineconeAsyncio.create_index_for_model

.. automethod:: pinecone::PineconeAsyncio.create_index_from_backup

.. automethod:: pinecone::PineconeAsyncio.list_indexes

.. automethod:: pinecone::PineconeAsyncio.describe_index

.. automethod:: pinecone::PineconeAsyncio.configure_index

.. automethod:: pinecone::PineconeAsyncio.delete_index

.. automethod:: pinecone::PineconeAsyncio.has_index

Backups
-------

.. automethod:: pinecone::PineconeAsyncio.create_backup

.. automethod:: pinecone::PineconeAsyncio.list_backups

.. automethod:: pinecone::PineconeAsyncio.describe_backup

.. automethod:: pinecone::PineconeAsyncio.delete_backup

Collections
-----------

.. automethod:: pinecone::PineconeAsyncio.create_collection

.. automethod:: pinecone::PineconeAsyncio.list_collections

.. automethod:: pinecone::PineconeAsyncio.describe_collection

.. automethod:: pinecone::PineconeAsyncio.delete_collection

Restore Jobs
------------

.. automethod:: pinecone::PineconeAsyncio.list_restore_jobs

.. automethod:: pinecone::PineconeAsyncio.describe_restore_job

DB Data Plane
=============

.. autoclass:: pinecone.db_data::IndexAsyncio

.. automethod:: pinecone.db_data::IndexAsyncio.__init__

.. automethod:: pinecone.db_data::IndexAsyncio.describe_index_stats

Vectors
-------

.. automethod:: pinecone.db_data::IndexAsyncio.upsert

.. automethod:: pinecone.db_data::IndexAsyncio.query

.. automethod:: pinecone.db_data::IndexAsyncio.query_namespaces

.. automethod:: pinecone.db_data::IndexAsyncio.delete

.. automethod:: pinecone.db_data::IndexAsyncio.fetch

.. automethod:: pinecone.db_data::IndexAsyncio.list

.. automethod:: pinecone.db_data::IndexAsyncio.list_paginated

Records
-------

If you have created an index using integrated inference, you can use the following methods to
search and retrieve records.

.. automethod:: pinecone.db_data::IndexAsyncio.search

.. automethod:: pinecone.db_data::IndexAsyncio.search_records



Inference
=========

.. automethod:: pinecone.inference::AsyncioInference.embed

.. automethod:: pinecone.inference::AsyncioInference.rerank

.. automethod:: pinecone.inference::AsyncioInference.list_models

.. automethod:: pinecone.inference::AsyncioInference.get_model
