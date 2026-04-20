"""Backwards-compatibility shim for :mod:`pinecone.core`.

Re-exports legacy OpenAPI-generated model paths that existed before the
``python-sdk2`` rewrite. Preserved to keep pre-rewrite callers working.
New code should import from canonical modules.

:meta private:
"""
