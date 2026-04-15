"""Backwards-compatibility shim for :mod:`pinecone.admin.organizations`.

Re-exports :class:`pinecone.admin.organizations.Organizations` as
``OrganizationResource`` for pre-rewrite callers. Preserved to keep
legacy ``from pinecone.admin.resources.organization import OrganizationResource``
imports working. New code should use
``from pinecone.admin.organizations import Organizations``.

:meta private:
"""
from __future__ import annotations

from pinecone.admin.organizations import Organizations as OrganizationResource

__all__ = ["OrganizationResource"]
