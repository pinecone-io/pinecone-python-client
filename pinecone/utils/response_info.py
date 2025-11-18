"""Response information utilities for extracting LSN headers from API responses."""

from typing import Any, TypedDict


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
    if headers is None:
        return {"raw_headers": {}}

    raw_headers: dict[str, str] = {}
    for key, value in headers.items():
        key_lower = key.lower()
        if isinstance(value, (list, tuple)) and len(value) > 0:
            # Handle headers that may be lists
            raw_headers[key_lower] = str(value[0])
        else:
            raw_headers[key_lower] = str(value)

    return {"raw_headers": raw_headers}
