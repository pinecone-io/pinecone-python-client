"""ModelInfoList wrapper for listing inference models."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, overload

from pinecone.models.inference.models import ModelInfo


class ModelInfoList:
    """Wrapper around a list of ModelInfo with convenience methods.

    Supports integer indexing, string key access (``["models"]``),
    iteration, ``len()``, and a ``.names()`` convenience method.

    Attributes:
        models: The underlying list of :class:`ModelInfo` instances.
    """

    def __init__(self, models: list[ModelInfo]) -> None:
        """Initialize a ModelInfoList.

        Args:
            models: List of :class:`ModelInfo` instances.
        """
        self._models = models

    @property
    def models(self) -> list[ModelInfo]:
        """Return the underlying list of models."""
        return self._models

    def names(self) -> list[str]:
        """Return a list of model identifiers.

        Returns:
            list[str]: Model identifiers from each :class:`ModelInfo`.

        Examples:
            List available model names:

            >>> from pinecone import Pinecone
            >>> pc = Pinecone(api_key="your-api-key")
            >>> models = pc.inference.list_models()
            >>> models.names()
            ['multilingual-e5-large', 'pinecone-sparse-english-v0']
        """
        return [m.model for m in self._models]

    @overload
    def __getitem__(self, key: int) -> ModelInfo: ...

    @overload
    def __getitem__(self, key: str) -> list[ModelInfo]: ...

    def __getitem__(self, key: int | str) -> Any:
        """Support integer indexing and string key access.

        Args:
            key: An integer index into the models list, or the string
                ``"models"`` to get the full list.

        Returns:
            A :class:`ModelInfo` for integer keys, or ``list[ModelInfo]``
            for the ``"models"`` key.
        """
        if isinstance(key, int):
            return self._models[key]
        if key == "models":
            return self._models
        raise KeyError(key)

    def __len__(self) -> int:
        return len(self._models)

    def __iter__(self) -> Iterator[ModelInfo]:
        return iter(self._models)

    def __repr__(self) -> str:
        summaries = ", ".join(f"<model={m.model!r}, type={m.type!r}>" for m in self._models)
        return f"ModelInfoList([{summaries}])"
