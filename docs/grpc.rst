===============
PineconeGRPC
===============

.. autoclass:: pinecone.grpc::PineconeGRPC

.. automethod:: pinecone.grpc::PineconeGRPC.Index

DB Control Plane
================

Indexes
-------

.. automethod:: pinecone.grpc::PineconeGRPC.create_index

.. automethod:: pinecone.grpc::PineconeGRPC.create_index_for_model

.. automethod:: pinecone.grpc::PineconeGRPC.create_index_from_backup

.. automethod:: pinecone.grpc::PineconeGRPC.list_indexes

.. automethod:: pinecone.grpc::PineconeGRPC.describe_index

.. automethod:: pinecone.grpc::PineconeGRPC.configure_index

.. automethod:: pinecone.grpc::PineconeGRPC.delete_index

.. automethod:: pinecone.grpc::PineconeGRPC.has_index

Backups
-------

.. automethod:: pinecone.grpc::PineconeGRPC.create_backup

.. automethod:: pinecone.grpc::PineconeGRPC.list_backups

.. automethod:: pinecone.grpc::PineconeGRPC.describe_backup

.. automethod:: pinecone.grpc::PineconeGRPC.delete_backup

Collections
-----------

.. automethod:: pinecone.grpc::PineconeGRPC.create_collection

.. automethod:: pinecone.grpc::PineconeGRPC.list_collections

.. automethod:: pinecone.grpc::PineconeGRPC.describe_collection

.. automethod:: pinecone.grpc::PineconeGRPC.delete_collection

Restore Jobs
------------

.. automethod:: pinecone.grpc::PineconeGRPC.list_restore_jobs

.. automethod:: pinecone.grpc::PineconeGRPC.describe_restore_job

DB Data Plane
=============

.. autoclass:: pinecone.grpc::GRPCIndex

.. automethod:: pinecone.grpc::GRPCIndex.__init__

.. automethod:: pinecone.grpc::GRPCIndex.describe_index_stats

Vectors
-------

.. automethod:: pinecone.grpc::GRPCIndex.upsert

.. automethod:: pinecone.grpc::GRPCIndex.query

.. automethod:: pinecone.grpc::GRPCIndex.query_namespaces

.. automethod:: pinecone.grpc::GRPCIndex.delete

.. automethod:: pinecone.grpc::GRPCIndex.fetch

.. automethod:: pinecone.grpc::GRPCIndex.list

.. automethod:: pinecone.grpc::GRPCIndex.list_paginated

.. automethod:: pinecone.grpc::GRPCIndex.fetch_by_metadata

.. automethod:: pinecone.grpc::GRPCIndex.update

.. automethod:: pinecone.grpc::GRPCIndex.upsert_from_dataframe

Namespaces
----------

.. automethod:: pinecone.grpc::GRPCIndex.create_namespace

.. automethod:: pinecone.grpc::GRPCIndex.describe_namespace

.. automethod:: pinecone.grpc::GRPCIndex.delete_namespace

.. automethod:: pinecone.grpc::GRPCIndex.list_namespaces

.. automethod:: pinecone.grpc::GRPCIndex.list_namespaces_paginated
