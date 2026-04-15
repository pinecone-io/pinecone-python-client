"""Backwards-compatibility shim for :mod:`pinecone.admin.resources`.

Re-exports legacy resource classes from their canonical locations in the
new SDK. Preserved to keep pre-rewrite callers working. New code should
import from the canonical modules directly.

:meta private:
"""

from __future__ import annotations

from pinecone.admin.resources.api_key import ApiKeyResource
from pinecone.admin.resources.organization import OrganizationResource
from pinecone.admin.resources.project import ProjectResource

__all__ = ["ApiKeyResource", "OrganizationResource", "ProjectResource"]
