"""Tests for User-Agent string construction."""

from __future__ import annotations

from pinecone import __version__
from pinecone._internal.config import PineconeConfig
from pinecone._internal.http_client import _build_headers
from pinecone._internal.user_agent import build_user_agent


class TestBuildUserAgent:
    """Tests for the build_user_agent function."""

    def test_default_format(self) -> None:
        result = build_user_agent("0.1.0")
        assert result == "python-client-0.1.0"

    def test_uses_provided_version(self) -> None:
        result = build_user_agent("2.3.4")
        assert result == "python-client-2.3.4"

    def test_source_tag_appended(self) -> None:
        result = build_user_agent("0.1.0", source_tag="my_tag")
        assert result == "python-client-0.1.0 source_tag=my_tag"

    def test_none_source_tag_no_suffix(self) -> None:
        result = build_user_agent("0.1.0", source_tag=None)
        assert result == "python-client-0.1.0"

    def test_empty_source_tag_no_suffix(self) -> None:
        result = build_user_agent("0.1.0", source_tag="")
        assert result == "python-client-0.1.0"

    def test_colons_preserved_in_source_tag(self) -> None:
        result = build_user_agent("0.1.0", source_tag="foo:bar")
        assert result == "python-client-0.1.0 source_tag=foo:bar"


class TestSourceTagNormalizationIntegration:
    """Tests that source tag normalization (done in PineconeConfig) works
    correctly when combined with build_user_agent via _build_headers."""

    def test_uppercase_lowered(self) -> None:
        config = PineconeConfig(api_key="test-key", source_tag="MyTag")
        assert config.source_tag == "mytag"
        result = build_user_agent("0.1.0", source_tag=config.source_tag)
        assert result == "python-client-0.1.0 source_tag=mytag"

    def test_spaces_become_underscores(self) -> None:
        config = PineconeConfig(api_key="test-key", source_tag="my tag")
        assert config.source_tag == "my_tag"

    def test_special_chars_stripped(self) -> None:
        config = PineconeConfig(api_key="test-key", source_tag="my@tag!#$%")
        assert config.source_tag == "mytag"

    def test_colons_preserved(self) -> None:
        config = PineconeConfig(api_key="test-key", source_tag="foo:bar")
        assert config.source_tag == "foo:bar"

    def test_mixed_normalization(self) -> None:
        config = PineconeConfig(api_key="test-key", source_tag="My App:Version 2!")
        assert config.source_tag == "my_app:version_2"


class TestBuildHeadersUserAgent:
    """Tests that _build_headers includes the correct User-Agent."""

    def test_default_user_agent(self) -> None:
        config = PineconeConfig(api_key="test-key")
        headers = _build_headers(config, "2025-10")
        assert headers["User-Agent"] == f"python-client-{__version__}"

    def test_user_agent_with_source_tag(self) -> None:
        config = PineconeConfig(api_key="test-key", source_tag="my_tag")
        headers = _build_headers(config, "2025-10")
        assert headers["User-Agent"] == f"python-client-{__version__} source_tag=my_tag"

    def test_user_agent_no_source_tag_suffix_when_empty(self) -> None:
        config = PineconeConfig(api_key="test-key", source_tag="")
        headers = _build_headers(config, "2025-10")
        assert headers["User-Agent"] == f"python-client-{__version__}"
        assert "source_tag" not in headers["User-Agent"]

    def test_user_agent_set_on_client_instance(self) -> None:
        """User-Agent is a default header on the HTTPClient, not per-request."""
        from pinecone._internal.http_client import HTTPClient

        config = PineconeConfig(
            api_key="test-key",
            host="https://test.pinecone.io",
            source_tag="test_tag",
        )
        client = HTTPClient(config, "2025-10")
        expected_ua = f"python-client-{__version__} source_tag=test_tag"
        assert client._headers["User-Agent"] == expected_ua
        client.close()

    def test_api_version_header_present(self) -> None:
        config = PineconeConfig(api_key="test-key")
        headers = _build_headers(config, "2025-10")
        assert headers["X-Pinecone-Api-Version"] == "2025-10"
