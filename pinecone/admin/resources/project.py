"""Backwards-compatibility shim for :mod:`pinecone.admin.projects`.

Re-exports :class:`pinecone.admin.projects.Projects` as
``ProjectResource`` for pre-rewrite callers. Preserved to keep
legacy ``from pinecone.admin.resources.project import ProjectResource``
imports working. New code should use
``from pinecone.admin.projects import Projects``.

:meta private:
"""
from __future__ import annotations

from pinecone.admin.projects import Projects as ProjectResource

__all__ = ["ProjectResource"]
