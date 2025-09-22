from typing import Any, Dict, Optional
from dataclasses import dataclass, field


# -----------------------
# Model
# -----------------------
@dataclass(frozen=True)
class Document:
    """
    Lightweight model for a Repository document.

    _id is optional: it will be present for fetched/listed documents,
    but not required when creating/upserting.
    """

    _id: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def _extract_id(payload: Dict[str, Any]) -> Optional[str]:
        return payload.get("_id")

    @classmethod
    def from_api(cls, payload: Dict[str, Any]) -> "Document":
        if not isinstance(payload, dict):
            raise TypeError("Document.from_api payload must be a dict")

        doc_id = cls._extract_id(payload)
        if not doc_id:
            raise ValueError("Document.from_api payload missing an '_id' field")

        return cls(_id=str(doc_id), raw=payload)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Document":
        """
        Create a Document directly from a dict (e.g., user-supplied).
        The dict must contain an '_id' field.
        """
        if not isinstance(payload, dict):
            raise TypeError("payload must be a dict.")

        doc_id = cls._extract_id(payload)  # may be None
        return cls(_id=str(doc_id), raw=dict(payload))

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the Document into a dict suitable for upsert.
        """
        return dict(self.raw)

    def __str__(self) -> str:
        truncate = 50
        as_str = self.raw.__str__()
        return as_str[:truncate] + (" ... " if len(as_str) > truncate else "")

    def __repr__(self) -> str:
        # Make a shallow copy of raw without the _id key (if present)
        raw_copy = {k: v for k, v in self.raw.items() if k != "_id"}

        truncate = 50
        as_str = repr(raw_copy)
        return (
            f"Document(_id={self._id!r}, raw={as_str[:truncate]}"
            f"{' ... ' if len(as_str) > truncate else ''})"
        )

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    # attribute-style passthrough into raw
    # e.g., doc.title instead of doc.raw["title"]
    def __getattr__(self, attr: str) -> Any:
        # __getattr__ is only called if normal attribute lookup fails
        raw = object.__getattribute__(self, "raw")
        if attr in raw:
            return raw[attr]
        raise AttributeError(f"{self.__class__.__name__!s} has no attribute {attr!r}")
