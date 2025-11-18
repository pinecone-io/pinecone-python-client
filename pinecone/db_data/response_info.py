"""Response information from API calls.

DEPRECATED: This module has been moved to pinecone.utils.response_info.
This file exists only for backwards compatibility during worktree operations.

Please import from pinecone.utils.response_info instead.
"""

import warnings

# Re-export from the new location
from pinecone.utils.response_info import ResponseInfo, extract_response_info

__all__ = ["ResponseInfo", "extract_response_info"]

warnings.warn(
    "pinecone.db_data.response_info is deprecated. "
    "Please import from pinecone.utils.response_info instead.",
    DeprecationWarning,
    stacklevel=2,
)
