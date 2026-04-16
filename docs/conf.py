# Sphinx configuration stub.
# This file is intentionally minimal — it will be expanded when Sphinx docs
# are fully introduced. The exclude_patterns list is required now to keep
# backwards-compatibility shim modules out of autodoc output.

from __future__ import annotations

project = "pinecone"
extensions: list[str] = []

exclude_patterns = [
    "pinecone/db_data.py",
    "pinecone/db_data/**",
    "pinecone/db_control/**",
    "pinecone/admin/resources/**",
    "pinecone/config/**",
    "pinecone/utils/response_info.py",
    "pinecone_plugins/**",
]
