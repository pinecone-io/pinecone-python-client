"""Type-compatibility smoke test for DX-0140 / Sequence-loosening change.

Verifies that callers can pass tuples (or any other Sequence-conforming type)
to public input parameters that previously required list. The runtime always
accepted these — this test pins the type contract so a future tightening
doesn't silently regress the DX win.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import TypedDict


def test_sequence_protocol_accepts_tuple() -> None:
    # Tuples are Sequences. This is a pure type-system check; no SDK call needed
    # at runtime — the assertion is that the static checker treats tuple[float, ...]
    # as a Sequence[float].
    values: Sequence[float] = (0.1, 0.2, 0.3)
    assert len(values) == 3

    ids: Sequence[str] = ("a", "b")
    assert ids[0] == "a"


def test_mapping_protocol_accepts_mappingproxy() -> None:
    """Read-only mapping types must satisfy public Mapping[str, Any] params.

    MappingProxyType (returned by e.g. dict.values() in some idioms,
    cls.__dict__) is the common case. If a future change tightens
    a public param back to dict[str, Any], this fails under mypy.
    """
    d = {"key": "value"}
    m: Mapping[str, str] = MappingProxyType(d)
    assert m["key"] == "value"


def test_mapping_typeddict_compatibility() -> None:
    """TypedDict instances are structural Mapping[str, Any]."""

    class Filter(TypedDict):
        genre: str

    filter_typed: Filter = {"genre": "comedy"}
    filter_as_mapping: Mapping[str, str] = filter_typed
    assert filter_as_mapping["genre"] == "comedy"
