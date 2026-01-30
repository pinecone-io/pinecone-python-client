"""Document class for document search responses."""

from __future__ import annotations

from typing import Any, Iterator


class Document:
    """A document returned from a document search operation.

    Documents have standard fields (``id`` and ``score``) plus dynamic fields
    that can be accessed via attribute access, dict-style access, or the
    ``get()`` method.

    :param id: The unique identifier for the document.
    :param score: The relevance score for the document.
    :param fields: Additional dynamic fields from the document.

    Example usage::

        # Assuming results from index.search_documents()
        for doc in results.documents:
            print(doc.id)              # Standard field
            print(doc.score)           # Standard field
            print(doc.title)           # Attribute access to dynamic field
            print(doc["title"])        # Dict access to dynamic field
            print(doc.get("title"))    # Safe access (returns None if missing)
            print(doc.get("title", "N/A"))  # Safe access with default
    """

    __slots__ = ("_id", "_score", "_fields")

    _id: str
    _score: float
    _fields: dict[str, Any]

    def __init__(self, id: str, score: float, **fields: Any) -> None:
        """Initialize a Document.

        :param id: The unique identifier for the document.
        :param score: The relevance score for the document.
        :param fields: Additional dynamic fields from the document.
        """
        object.__setattr__(self, "_id", id)
        object.__setattr__(self, "_score", score)
        object.__setattr__(self, "_fields", fields)

    @property
    def id(self) -> str:
        """The unique identifier for the document."""
        return self._id

    @property
    def score(self) -> float:
        """The relevance score for the document."""
        return self._score

    def __getattr__(self, name: str) -> Any:
        """Allow attribute access to dynamic fields.

        :param name: The field name to access.
        :returns: The field value.
        :raises AttributeError: If the field does not exist.
        """
        try:
            return self._fields[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __getitem__(self, key: str) -> Any:
        """Allow dict-style access to fields.

        :param key: The field name to access.
        :returns: The field value.
        :raises KeyError: If the field does not exist.
        """
        if key == "id":
            return self._id
        if key == "score":
            return self._score
        if key in self._fields:
            return self._fields[key]
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        """Safe access with default value.

        :param key: The field name to access.
        :param default: The default value if the field does not exist.
        :returns: The field value or the default.
        """
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self) -> list[str]:
        """Return all field names.

        :returns: List of all field names including id and score.
        """
        return ["id", "score"] + list(self._fields.keys())

    def values(self) -> list[Any]:
        """Return all field values.

        :returns: List of all field values.
        """
        return [self._id, self._score] + list(self._fields.values())

    def items(self) -> list[tuple[str, Any]]:
        """Return all field name-value pairs.

        :returns: List of (name, value) tuples.
        """
        return [("id", self._id), ("score", self._score)] + list(self._fields.items())

    def __contains__(self, key: str) -> bool:
        """Check if a field exists.

        :param key: The field name to check.
        :returns: True if the field exists, False otherwise.
        """
        return key in ("id", "score") or key in self._fields

    def __iter__(self) -> Iterator[str]:
        """Iterate over field names.

        :returns: Iterator over field names.
        """
        yield "id"
        yield "score"
        yield from self._fields

    def __len__(self) -> int:
        """Return the number of fields.

        :returns: Total number of fields including id and score.
        """
        return 2 + len(self._fields)

    def __repr__(self) -> str:
        """Return a string representation.

        :returns: String representation of the document.
        """
        field_str = ", ".join(f"{k}={v!r}" for k, v in self._fields.items())
        if field_str:
            return f"Document(id={self._id!r}, score={self._score!r}, {field_str})"
        return f"Document(id={self._id!r}, score={self._score!r})"

    def __eq__(self, other: object) -> bool:
        """Check equality with another Document.

        :param other: The object to compare with.
        :returns: True if equal, False otherwise.
        """
        if not isinstance(other, Document):
            return NotImplemented
        return (
            self._id == other._id and self._score == other._score and self._fields == other._fields
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary.

        :returns: Dictionary representation of the document.
        """
        return {"id": self._id, "score": self._score, **self._fields}
