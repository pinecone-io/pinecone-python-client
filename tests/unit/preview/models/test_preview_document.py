"""Unit tests for PreviewDocument and PreviewUsage models."""

from __future__ import annotations

import pytest

from pinecone.preview.models.documents import PreviewDocument, PreviewUsage

# ---------------------------------------------------------------------------
# PreviewUsage
# ---------------------------------------------------------------------------


def test_usage_read_units() -> None:
    u = PreviewUsage(read_units=42)
    assert u.read_units == 42


# ---------------------------------------------------------------------------
# PreviewDocument — typed properties
# ---------------------------------------------------------------------------


def test_typed_id() -> None:
    doc = PreviewDocument({"_id": "doc-1", "_score": 0.9})
    assert doc.id == "doc-1"
    assert doc._id == "doc-1"


def test_typed_score() -> None:
    doc = PreviewDocument({"_id": "doc-1", "_score": 0.75})
    assert doc.score == 0.75
    assert doc._score == 0.75


def test_typed_score_none_when_absent() -> None:
    doc = PreviewDocument({"_id": "doc-1"})
    assert doc.score is None
    assert doc._score is None


def test_typed_score_coercion() -> None:
    doc = PreviewDocument({"_id": "doc-1", "_score": 1})
    assert isinstance(doc.score, float)
    assert doc.score == 1.0
    assert doc._score == 1.0


# ---------------------------------------------------------------------------
# Dynamic attribute access
# ---------------------------------------------------------------------------


def test_dynamic_attribute_access() -> None:
    doc = PreviewDocument({"_id": "doc-1", "title": "Ancient Rome"})
    assert doc.title == "Ancient Rome"  # type: ignore[attr-defined]


def test_dynamic_attribute_missing_raises() -> None:
    doc = PreviewDocument({"_id": "doc-1"})
    with pytest.raises(AttributeError):
        _ = doc.nonexistent  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# .get() with default
# ---------------------------------------------------------------------------


def test_get_existing_key() -> None:
    doc = PreviewDocument({"_id": "doc-1", "category": "history"})
    assert doc.get("category") == "history"


def test_get_missing_key_returns_none() -> None:
    doc = PreviewDocument({"_id": "doc-1"})
    assert doc.get("missing") is None


def test_get_missing_key_with_default() -> None:
    doc = PreviewDocument({"_id": "doc-1"})
    assert doc.get("missing", "fallback") == "fallback"


# ---------------------------------------------------------------------------
# .to_dict() / .to_json() round-trip
# ---------------------------------------------------------------------------


def test_to_dict_returns_shallow_copy() -> None:
    data = {"_id": "doc-1", "_score": 0.5, "title": "Test"}
    doc = PreviewDocument(data)
    result = doc.to_dict()
    assert result == data
    # Mutation of the copy must not affect the original data stored in doc
    result["title"] = "Modified"
    assert doc.get("title") == "Test"


def test_to_json_roundtrip() -> None:
    import json

    data = {"_id": "doc-1", "_score": 0.5, "title": "Test"}
    doc = PreviewDocument(data)
    parsed = json.loads(doc.to_json())
    assert parsed == data


# ---------------------------------------------------------------------------
# Field name collision — spec §7
# ---------------------------------------------------------------------------


def test_score_collision_typed_property_wins() -> None:
    # A document field named "_score" must NOT shadow the typed property.
    doc = PreviewDocument({"_id": "doc-1", "_score": 99.9})
    # The typed property returns the float value from _data["_score"]
    assert doc.score == 99.9
    # .get() also reaches it (it's the same underlying value here)
    assert doc.get("_score") == 99.9


def test_id_collision_typed_property_wins() -> None:
    # If the payload contains both "id" and "_id", _id wins for the typed prop.
    doc = PreviewDocument({"_id": "primary", "id": "secondary"})
    assert doc.id == "primary"
    assert doc._id == "primary"
    # But .get() still reaches both raw keys
    assert doc.get("_id") == "primary"
    assert doc.get("id") == "secondary"


def test_score_field_only_via_get_when_desired() -> None:
    # Per spec: typed property takes precedence; raw access still works via .get()
    doc = PreviewDocument({"_id": "doc-1", "_score": 0.42})
    assert doc.score == 0.42
    assert doc.get("_score") == 0.42


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------


def test_repr_no_extra_fields() -> None:
    doc = PreviewDocument({"_id": "doc-1", "_score": 0.5})
    r = repr(doc)
    assert "_id='doc-1'" in r
    assert "score=0.5" in r
    assert "..." not in r


def test_repr_with_extra_fields() -> None:
    doc = PreviewDocument({"_id": "doc-1", "_score": 0.5, "title": "Rome"})
    r = repr(doc)
    assert "_id='doc-1'" in r
    assert "score=0.5" in r
    assert "..." in r


def test_repr_no_score() -> None:
    doc = PreviewDocument({"_id": "doc-1"})
    r = repr(doc)
    assert "_id='doc-1'" in r
    assert "score=None" in r
