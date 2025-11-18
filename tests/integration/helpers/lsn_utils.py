"""Utilities for extracting and comparing LSN (Log Sequence Number) values from API response headers.

LSN headers are used to determine data freshness without polling describe_index_stats.
These headers are not part of the official OpenAPI spec, so this module handles
them defensively with fallbacks.

This is a test utility and not part of the public API.
"""

from typing import Any, Optional


# Possible header names for LSN values (case-insensitive matching)
# Based on actual API responses discovered via scripts/inspect_lsn_headers.py:
# - x-pinecone-request-lsn: Appears in write operations (upsert, delete) - committed LSN
# - x-pinecone-max-indexed-lsn: Appears in query operations - reconciled/max indexed LSN
#
# Note: These headers are not part of the OpenAPI spec and are undocumented behavior.
# The implementation is defensive and falls back gracefully if headers are missing.
LSN_RECONCILED_HEADERS = [
    "x-pinecone-max-indexed-lsn"  # Actual header name from API (discovered via inspection)
]

LSN_COMMITTED_HEADERS = [
    "x-pinecone-request-lsn"  # Actual header name from API (discovered via inspection)
]


def _get_header_value(headers: dict[str, Any], possible_names: list[str]) -> Optional[int]:
    """Extract a header value by trying multiple possible header names.

    Args:
        headers: Dictionary of response headers (case-insensitive matching)
        possible_names: List of possible header names to try

    Returns:
        Integer value of the header if found, None otherwise
    """
    if not headers:
        return None

    # Normalize headers to lowercase for case-insensitive matching
    headers_lower = {k.lower(): v for k, v in headers.items()}

    for name in possible_names:
        value = headers_lower.get(name.lower())
        if value is not None:
            try:
                # Try to convert to int
                return int(value)
            except (ValueError, TypeError):
                # If conversion fails, try parsing as string
                try:
                    return int(str(value).strip())
                except (ValueError, TypeError):
                    continue

    return None


def extract_lsn_reconciled(headers: dict[str, Any]) -> Optional[int]:
    """Extract the reconciled LSN value from response headers.

    The reconciled LSN represents the latest log sequence number that has been
    reconciled and is available for reads.

    Args:
        headers: Dictionary of response headers from an API call

    Returns:
        The reconciled LSN value as an integer, or None if not found
    """
    return _get_header_value(headers, LSN_RECONCILED_HEADERS)


def extract_lsn_committed(headers: dict[str, Any]) -> Optional[int]:
    """Extract the committed LSN value from response headers.

    The committed LSN represents the log sequence number that was committed
    for a write operation.

    Args:
        headers: Dictionary of response headers from an API call

    Returns:
        The committed LSN value as an integer, or None if not found
    """
    return _get_header_value(headers, LSN_COMMITTED_HEADERS)


def extract_lsn_values(headers: dict[str, Any]) -> tuple[Optional[int], Optional[int]]:
    """Extract both reconciled and committed LSN values from headers.

    Args:
        headers: Dictionary of response headers from an API call

    Returns:
        Tuple of (reconciled_lsn, committed_lsn). Either or both may be None.
    """
    reconciled = extract_lsn_reconciled(headers)
    committed = extract_lsn_committed(headers)
    return (reconciled, committed)


def is_lsn_reconciled(target_lsn: int, current_reconciled_lsn: Optional[int]) -> bool:
    """Check if a target LSN has been reconciled.

    Args:
        target_lsn: The LSN value to check (typically from a write operation)
        current_reconciled_lsn: The current reconciled LSN from a read operation

    Returns:
        True if target_lsn <= current_reconciled_lsn, False otherwise.
        Returns False if current_reconciled_lsn is None (header not available).
    """
    if current_reconciled_lsn is None:
        return False
    return target_lsn <= current_reconciled_lsn


def get_headers_from_response(response: Any) -> Optional[dict[str, Any]]:
    """Extract headers from various response types.

    This function handles different response formats:
    - Tuple from _return_http_data_only=False: (data, status, headers)
    - RESTResponse object with getheaders() method
    - Dictionary of headers

    Args:
        response: Response object that may contain headers

    Returns:
        Dictionary of headers, or None if headers cannot be extracted
    """
    # Handle tuple response from _return_http_data_only=False
    if isinstance(response, tuple) and len(response) == 3:
        _, _, headers = response
        return headers if isinstance(headers, dict) else None

    # Handle RESTResponse object
    if hasattr(response, "getheaders"):
        headers = response.getheaders()
        if isinstance(headers, dict):
            return headers

    # Handle dictionary directly
    if isinstance(response, dict):
        return response

    return None
