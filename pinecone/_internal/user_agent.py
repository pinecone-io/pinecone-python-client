"""User-Agent string construction for the Pinecone SDK."""

from __future__ import annotations


def build_user_agent(version: str, source_tag: str | None = None) -> str:
    """Build the User-Agent header value.

    Args:
        version: SDK version string (e.g. "0.1.0").
        source_tag: Optional pre-normalized source tag to append.

    Returns:
        User-Agent string in the format ``python-client-{version}``
        with an optional ``source_tag={tag}`` suffix.
    """
    ua = f"python-client-{version}"
    if source_tag:
        ua = f"{ua} source_tag={source_tag}"
    return ua
