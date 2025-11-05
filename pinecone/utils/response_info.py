"""Response information utilities for extracting LSN headers from API responses."""

from typing import Dict, Any, Optional, TypedDict


class ResponseInfo(TypedDict, total=False):
    """Response metadata including LSN values and raw headers.

    Attributes:
        lsn_committed: Committed LSN from write operations (upsert, delete).
        lsn_reconciled: Reconciled LSN from read operations (query).
        raw_headers: Dictionary of all response headers (normalized to lowercase).
    """

    lsn_committed: int
    lsn_reconciled: int
    raw_headers: Dict[str, str]


def extract_response_info(headers: Optional[Dict[str, Any]]) -> ResponseInfo:
    """Extract LSN values and raw headers from response headers.

    Extracts LSN (Log Sequence Number) headers from API response headers
    and returns them in a structured format. Header names are matched
    case-insensitively.

    Args:
        headers: Dictionary of response headers, or None.

    Returns:
        ResponseInfo dictionary with optional LSN values and raw headers.
        The raw_headers dictionary is always present and contains all
        headers normalized to lowercase keys.

    Examples:
        >>> headers = {"x-pinecone-request-lsn": "12345", "Content-Type": "application/json"}
        >>> info = extract_response_info(headers)
        >>> info["lsn_committed"]
        12345
        >>> info["raw_headers"]["content-type"]
        'application/json'
    """
    if headers is None:
        headers = {}

    # Normalize headers to lowercase keys
    # Exclude timing-dependent headers that cause test flakiness
    timing_headers = {
        "x-envoy-upstream-service-time",
        "date",
        "x-request-id",  # Request IDs are unique per request
    }
    raw_headers: Dict[str, str] = {}
    for key, value in headers.items():
        key_lower = key.lower()
        if key_lower not in timing_headers:
            if isinstance(value, (list, tuple)) and len(value) > 0:
                # Handle headers that may be lists
                raw_headers[key_lower] = str(value[0])
            else:
                raw_headers[key_lower] = str(value)

    result: ResponseInfo = {"raw_headers": raw_headers}

    # Extract committed LSN (from write operations)
    committed_header = None
    for key in raw_headers:
        if key == "x-pinecone-request-lsn":
            committed_header = raw_headers[key]
            break

    if committed_header:
        try:
            result["lsn_committed"] = int(committed_header)
        except (ValueError, TypeError):
            pass

    # Extract reconciled LSN (from read operations)
    reconciled_header = None
    for key in raw_headers:
        if key == "x-pinecone-max-indexed-lsn":
            reconciled_header = raw_headers[key]
            break

    if reconciled_header:
        try:
            result["lsn_reconciled"] = int(reconciled_header)
        except (ValueError, TypeError):
            pass

    return result
