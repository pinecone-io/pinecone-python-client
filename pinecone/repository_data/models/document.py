from typing import Any, Dict, Optional
from dataclasses import dataclass, field


# -----------------------
# Model
# -----------------------
@dataclass(frozen=True)
class Document:
    """
    Lightweight model for a Repository document.

    Fields are intentionally permissive to accommodate any document shape.
    """

    _id: str
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

        doc_id = cls._extract_id(payload)
        if not doc_id:
            raise ValueError("Document dict must include an '_id' field")

        return cls(_id=str(doc_id), raw=dict(payload))

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the Document into a dict suitable for upsert.
        """
        return dict(self.raw)
