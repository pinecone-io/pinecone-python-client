"""Backwards-compatibility shim for :mod:`pinecone.client.assistants`.

Re-exports classes that used to live at :mod:`pinecone_plugins.assistant` before
the `python-sdk2` rewrite. Preserved to keep pre-rewrite callers working.
New code should import from the canonical module.

:meta private:
"""

try:
    import pinecone  # noqa: F401
except ImportError as e:
    raise ImportError(
        "The pinecone-plugin-assistant package requires the pinecone package to be installed. "
        "Install it with: pip install pinecone"
    ) from e

from pinecone_plugin_interface import PluginMetadata

from pinecone_plugins.assistant.assistant.assistant import Assistant

__installables__ = [
    PluginMetadata(
        target_object="Pinecone",
        namespace="assistant",
        implementation_class=Assistant,
    )
]

__all__ = ["__installables__"]
