"""Backwards-compatibility shim for :mod:`pinecone.client.assistants`.

Re-exports classes that used to live at :mod:`pinecone_plugins.assistant` before
the `python-sdk2` rewrite. Preserved to keep pre-rewrite callers working.
New code should import from the canonical module.

:meta private:
"""

__all__: list[str] = []
