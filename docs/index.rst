Pinecone Python SDK
===================

The Pinecone Python SDK manages indexes with :class:`~pinecone.Pinecone` and
reads and writes data with the :class:`~pinecone.Index` that ``pc.index("name")``
returns. New here? Start with the quickstart. Looking up a method? Go to the
reference section.

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   getting-started/installation
   getting-started/quickstart
   getting-started/authentication

.. toctree::
   :maxdepth: 1
   :caption: Concepts and guides

   guides/concepts
   guides/sync-vs-async
   guides/error-handling
   guides/retries
   guides/pagination
   guides/performance
   guides/bulk-ingest
   guides/grpc

.. toctree::
   :maxdepth: 1
   :caption: Indexes

   how-to/indexes/serverless
   how-to/indexes/pod
   how-to/indexes/backups-and-restore
   how-to/collections
   how-to/vectors/namespaces

.. toctree::
   :maxdepth: 1
   :caption: Documents

   how-to/documents/schema-and-upsert
   how-to/documents/search
   how-to/documents/manage

.. toctree::
   :maxdepth: 1
   :caption: Vectors

   how-to/vectors/upsert-and-query
   how-to/vectors/upsert-from-dataframe
   how-to/vectors/bulk-import

.. toctree::
   :maxdepth: 1
   :caption: Inference

   how-to/inference/embeddings
   how-to/inference/reranking

.. toctree::
   :maxdepth: 1
   :caption: Assistant

   how-to/assistant

.. toctree::
   :maxdepth: 1
   :caption: Legacy integrated indexes

   how-to/integrated-records

.. toctree::
   :maxdepth: 1
   :caption: Migration and FAQ

   migration/v10-1-upgrade
   migration/v10-migration
   migration/v10-grpc-partial-failures
   migration/v9-migration
   faq

.. toctree::
   :maxdepth: 1
   :caption: Reference

   reference/pinecone
   reference/sync-index
   reference/async-pinecone
   reference/async-index
   reference/grpc
   reference/admin
   reference/schema-builder
   reference/models
   reference/exceptions
