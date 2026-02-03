"""Shared utilities for adapter implementations.

This module provides helper functions used across different adapters
for common operations like extracting response metadata.
"""

from typing import Any

from pinecone.utils.response_info import ResponseInfo, extract_response_info


def extract_response_metadata(response: Any) -> ResponseInfo:
    """Extract response metadata from an OpenAPI response object.

    Extracts the _response_info attribute from an OpenAPI response if present,
    otherwise returns an empty ResponseInfo with empty headers.

    Args:
        response: An OpenAPI response object that may have _response_info.

    Returns:
        ResponseInfo with extracted headers, or empty headers if not present.
    """
    response_info = None
    if hasattr(response, "_response_info"):
        response_info = response._response_info

    if response_info is None:
        response_info = extract_response_info({})

    return response_info
