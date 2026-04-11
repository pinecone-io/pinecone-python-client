"""Embedding response models for the Inference API."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Union, overload

from msgspec import Struct


class EmbedUsage(Struct, kw_only=True):
    """Token usage information for an embedding request.

    Attributes:
        total_tokens: Total number of tokens processed.
    """

    total_tokens: int

    def __getitem__(self, key: str) -> Any:
        """Support bracket access (e.g. usage['total_tokens'])."""
        if key not in self.__struct_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        """Support ``in`` operator (e.g. ``'total_tokens' in usage``)."""
        return key in self.__struct_fields__


class DenseEmbedding(Struct, kw_only=True):
    """A dense embedding vector.

    Attributes:
        values: The embedding values as a list of floats.
        vector_type: The type of embedding, always ``"dense"``.
    """

    values: list[float]
    vector_type: str = "dense"

    def __getitem__(self, key: str) -> Any:
        """Support bracket access (e.g. embedding['values'])."""
        if key not in self.__struct_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        """Support ``in`` operator (e.g. ``'values' in embedding``)."""
        return key in self.__struct_fields__

    def __repr__(self) -> str:
        if len(self.values) > 5:
            preview = ", ".join(repr(v) for v in self.values[:3])
            values_str = f"[{preview}, ...{len(self.values) - 3} more]"
        else:
            values_str = repr(self.values)
        return f"DenseEmbedding(values={values_str}, vector_type={self.vector_type!r})"


class SparseEmbedding(Struct, kw_only=True):
    """A sparse embedding vector.

    Attributes:
        sparse_values: The non-zero values of the sparse embedding.
        sparse_indices: The indices of the non-zero values.
        sparse_tokens: Optional token strings corresponding to each index.
        vector_type: The type of embedding, always ``"sparse"``.
    """

    sparse_values: list[float]
    sparse_indices: list[int]
    sparse_tokens: list[str] | None = None
    vector_type: str = "sparse"

    def __getitem__(self, key: str) -> Any:
        """Support bracket access (e.g. embedding['sparse_values'])."""
        if key not in self.__struct_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        """Support ``in`` operator (e.g. ``'sparse_values' in embedding``)."""
        return key in self.__struct_fields__

    def __repr__(self) -> str:
        if len(self.sparse_indices) > 5:
            idx_preview = ", ".join(repr(v) for v in self.sparse_indices[:3])
            indices_str = f"[{idx_preview}, ...{len(self.sparse_indices) - 3} more]"
        else:
            indices_str = repr(self.sparse_indices)
        if len(self.sparse_values) > 5:
            val_preview = ", ".join(repr(v) for v in self.sparse_values[:3])
            values_str = f"[{val_preview}, ...{len(self.sparse_values) - 3} more]"
        else:
            values_str = repr(self.sparse_values)
        parts = [
            f"sparse_indices={indices_str}",
            f"sparse_values={values_str}",
            f"vector_type={self.vector_type!r}",
        ]
        if self.sparse_tokens is not None:
            parts.insert(2, f"sparse_tokens={self.sparse_tokens!r}")
        return f"SparseEmbedding({', '.join(parts)})"


Embedding = Union[DenseEmbedding, SparseEmbedding]


class EmbeddingsList(Struct, kw_only=True):
    """Response from the embed endpoint.

    Supports integer indexing, iteration, and ``len()`` over the
    embedded data items, as well as bracket access for field names.

    Attributes:
        model: The model used to generate embeddings.
        vector_type: The type of embeddings returned (``"dense"`` or ``"sparse"``).
        data: The list of embedding objects.
        usage: Token usage information.
    """

    model: str
    vector_type: str
    data: list[DenseEmbedding] | list[SparseEmbedding]
    usage: EmbedUsage

    @overload
    def __getitem__(self, key: int) -> DenseEmbedding | SparseEmbedding: ...

    @overload
    def __getitem__(self, key: str) -> Any: ...

    def __getitem__(self, key: int | str) -> Any:
        """Support integer indexing into data and string bracket access.

        Args:
            key: An integer index into ``data``, or a string field name.

        Returns:
            The embedding at the given index, or the field value.
        """
        if isinstance(key, int):
            return self.data[key]
        if key not in self.__struct_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        """Support ``in`` for field names (str) and embedding membership."""
        if isinstance(key, str):
            return key in self.__struct_fields__
        return key in self.data

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> Iterator[DenseEmbedding | SparseEmbedding]:
        return iter(self.data)

    def __repr__(self) -> str:
        return (
            f"EmbeddingsList("
            f"model={self.model!r}, "
            f"vector_type={self.vector_type!r}, "
            f"count={len(self.data)}, "
            f"usage={self.usage!r})"
        )
