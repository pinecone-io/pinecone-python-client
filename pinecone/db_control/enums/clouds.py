"""Backwards-compatibility shim for :mod:`pinecone.models.enums`.

Re-exports classes that used to live at :mod:`pinecone.db_control.enums.clouds` before
the `python-sdk2` rewrite. Preserved to keep pre-rewrite callers working.
New code should import from the canonical module.

:meta private:
"""

from __future__ import annotations

from pinecone.models.enums import AwsRegion, AzureRegion, GcpRegion

__all__ = ["AwsRegion", "AzureRegion", "GcpRegion"]
