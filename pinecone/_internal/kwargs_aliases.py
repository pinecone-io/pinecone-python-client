"""Helper for accepting legacy parameter names via **kwargs.

Used by backwards-compatibility shims where a parameter was renamed
(e.g. assistant_name -> name) but legacy callers must keep working.
"""

from __future__ import annotations

from typing import Any

from pinecone.errors.exceptions import PineconeValueError


def remap_legacy_kwargs(
    kwargs: dict[str, Any],
    *,
    aliases: dict[str, str],
    method_name: str,
) -> dict[str, Any]:
    """Rewrite legacy parameter names to their canonical equivalents.

    Args:
        kwargs: The ``**kwargs`` dict received by the caller method.
        aliases: Mapping of ``legacy_name -> canonical_name``.
        method_name: Name of the calling method, used in error messages.

    Returns:
        A new dict with all legacy names replaced by their canonical names.
        Canonical-named keys that were passed directly are preserved as-is.

    Raises:
        PineconeValueError: If ``kwargs`` contains both the legacy name and the
            canonical name for the same parameter, or if ``kwargs`` contains a
            key that is neither a legacy alias nor expected to be passed
            through (unknown kwarg).

    Example:
        >>> remap_legacy_kwargs(
        ...     {"assistant_name": "foo"},
        ...     aliases={"assistant_name": "name"},
        ...     method_name="create",
        ... )
        {'name': 'foo'}
    """
    result: dict[str, Any] = {}
    seen_canonical: set[str] = set()

    for key, value in kwargs.items():
        if key in aliases:
            canonical = aliases[key]
            if canonical in kwargs:
                raise PineconeValueError(
                    f"{method_name}() received both {key!r} (legacy) and "
                    f"{canonical!r} (current) for the same parameter. "
                    f"Pass only one — prefer {canonical!r}."
                )
            if canonical in seen_canonical:
                raise PineconeValueError(
                    f"{method_name}() received multiple legacy aliases that map to {canonical!r}."
                )
            result[canonical] = value
            seen_canonical.add(canonical)
        else:
            # Not a legacy alias. The calling method is responsible for
            # validating this against its own accepted parameter set.
            result[key] = value

    return result


def reject_unknown_kwargs(
    kwargs: dict[str, Any],
    *,
    allowed: set[str],
    method_name: str,
) -> None:
    """Raise ``PineconeValueError`` if *kwargs* contains any key outside *allowed*.

    Call this after :func:`remap_legacy_kwargs` so that legacy aliases have
    already been translated to canonical names.
    """
    unknown = set(kwargs) - allowed
    if unknown:
        unknown_list = ", ".join(repr(k) for k in sorted(unknown))
        allowed_list = ", ".join(repr(k) for k in sorted(allowed))
        raise PineconeValueError(
            f"{method_name}() received unexpected keyword argument(s): "
            f"{unknown_list}. Expected one of: {allowed_list}."
        )
