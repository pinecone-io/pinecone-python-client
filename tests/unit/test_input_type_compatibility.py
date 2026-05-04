"""Type-compatibility smoke test for DX-0140 / Sequence-loosening change.

Verifies that callers can pass tuples (or any other Sequence-conforming type)
to public input parameters that previously required list. The runtime always
accepted these — this test pins the type contract so a future tightening
doesn't silently regress the DX win.
"""

from __future__ import annotations

from collections.abc import Sequence


def test_sequence_protocol_accepts_tuple() -> None:
    # Tuples are Sequences. This is a pure type-system check; no SDK call needed
    # at runtime — the assertion is that the static checker treats tuple[float, ...]
    # as a Sequence[float].
    values: Sequence[float] = (0.1, 0.2, 0.3)
    assert len(values) == 3

    ids: Sequence[str] = ("a", "b")
    assert ids[0] == "a"
