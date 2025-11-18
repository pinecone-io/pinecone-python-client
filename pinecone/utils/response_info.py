"""Response information utilities for extracting LSN headers from API responses."""

from typing import Any, TypedDict

# Exclude timing-dependent headers that cause test flakiness
# Defined at module level to avoid recreation on every function call
_TIMING_HEADERS = frozenset(
    (
        "x-envoy-upstream-service-time",
        "date",
        "x-request-id",  # Request IDs are unique per request
    )
)


class ResponseInfo(TypedDict):
    """Response metadata including raw headers.

    Attributes:
        raw_headers: Dictionary of all response headers (normalized to lowercase).
    """

    raw_headers: dict[str, str]


def extract_response_info(headers: dict[str, Any] | None) -> ResponseInfo:
    """Extract raw headers from response headers.

    Extracts and normalizes response headers from API responses.
    Header names are normalized to lowercase keys.

    Args:
        headers: Dictionary of response headers, or None.

    Returns:
        ResponseInfo dictionary with raw_headers containing all
        headers normalized to lowercase keys.

    Examples:
        >>> headers = {"x-pinecone-request-lsn": "12345", "Content-Type": "application/json"}
        >>> info = extract_response_info(headers)
        >>> info["raw_headers"]["content-type"]
        'application/json'
        >>> info["raw_headers"]["x-pinecone-request-lsn"]
        '12345'
    """
    if not headers:
        return {"raw_headers": {}}

    # Optimized: use dictionary comprehension for better performance
    # Pre-compute lowercase keys and filter in one pass
    raw_headers = {}
    for key, value in headers.items():
        key_lower = key.lower()
        if key_lower not in _TIMING_HEADERS:
            # Optimize value conversion: check most common types first
            if isinstance(value, list) and value:
                raw_headers[key_lower] = str(value[0])
            elif isinstance(value, tuple) and value:
                raw_headers[key_lower] = str(value[0])
            elif isinstance(value, str):
                # Already a string, no conversion needed
                raw_headers[key_lower] = value
            else:
                raw_headers[key_lower] = str(value)

    return {"raw_headers": raw_headers}
