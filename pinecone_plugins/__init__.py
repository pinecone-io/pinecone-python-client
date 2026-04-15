"""Backwards-compatibility shim package for the legacy ``pinecone-plugin-assistant`` distribution.

Re-exports the classes that used to ship in the separate
``pinecone-plugin-assistant`` wheel. Preserved so that pre-rewrite
callers continue to resolve imports like
``from pinecone_plugins.assistant.models.chat import ChatResponse``.

:meta private:
"""

from __future__ import annotations
