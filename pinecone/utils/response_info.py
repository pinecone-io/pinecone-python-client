"""Response information utilities for extracting headers from API responses."""

from typing import Any, TypedDict


class ResponseInfo(TypedDict):
    """Response metadata including raw headers.

    Attributes:
        raw_headers: Dictionary of all response headers (normalized to lowercase).
    """

    raw_headers: dict[str, str]


def extract_response_info(headers: dict[str, Any] | None) -> ResponseInfo:
    """Extract raw headers from response headers.

    Extracts and normalizes all response headers from API responses.
    Header names are normalized to lowercase keys. All headers are included
    without filtering.

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

    # Optimized: normalize keys to lowercase and convert values to strings
    # Check string type first (most common case) for better performance
    raw_headers = {}
    for key, value in headers.items():
        key_lower = key.lower()
        if isinstance(value, str):
            # Already a string, no conversion needed
            raw_headers[key_lower] = value
        elif isinstance(value, list) and value:
            raw_headers[key_lower] = str(value[0])
        elif isinstance(value, tuple) and value:
            raw_headers[key_lower] = str(value[0])
        else:
            raw_headers[key_lower] = str(value)

    return {"raw_headers": raw_headers}
