from __future__ import annotations

import pytest

from pinecone._internal.kwargs_aliases import (
    reject_unknown_kwargs,
    remap_legacy_kwargs,
)
from pinecone.errors.exceptions import PineconeValueError


def test_remap_translates_legacy_to_canonical() -> None:
    out = remap_legacy_kwargs(
        {"assistant_name": "foo"},
        aliases={"assistant_name": "name"},
        method_name="create",
    )
    assert out == {"name": "foo"}


def test_remap_preserves_canonical_when_only_canonical_passed() -> None:
    out = remap_legacy_kwargs(
        {"name": "foo"},
        aliases={"assistant_name": "name"},
        method_name="create",
    )
    assert out == {"name": "foo"}


def test_remap_raises_when_both_legacy_and_canonical_passed() -> None:
    with pytest.raises(PineconeValueError) as exc:
        remap_legacy_kwargs(
            {"assistant_name": "foo", "name": "bar"},
            aliases={"assistant_name": "name"},
            method_name="create",
        )
    assert "both" in str(exc.value)
    assert "assistant_name" in str(exc.value)
    assert "name" in str(exc.value)


def test_reject_unknown_raises_on_extra_keys() -> None:
    with pytest.raises(PineconeValueError) as exc:
        reject_unknown_kwargs(
            {"name": "foo", "bogus": 1},
            allowed={"name", "instructions"},
            method_name="create",
        )
    assert "bogus" in str(exc.value)


def test_reject_unknown_passes_when_all_allowed() -> None:
    # No exception
    reject_unknown_kwargs(
        {"name": "foo"},
        allowed={"name", "instructions"},
        method_name="create",
    )
