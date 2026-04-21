"""Display method tests for chat models."""
from __future__ import annotations

from pinecone.models.assistant.chat import ChatUsage


class TestChatUsage:
    def test_repr(self) -> None:
        r = repr(ChatUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30))
        assert "10" in r
        assert "20" in r
        assert "30" in r

    def test_repr_html(self) -> None:
        h = ChatUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)._repr_html_()
        assert "<div" in h

    def test_repr_pretty(self) -> None:
        from IPython.lib.pretty import pretty

        assert "10" in pretty(ChatUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30))

    def test_safe_on_malformed(self) -> None:
        u = ChatUsage(prompt_tokens=1, completion_tokens=1, total_tokens=1)
        u.prompt_tokens = object()  # type: ignore[assignment]
        assert isinstance(repr(u), str)
        assert isinstance(u._repr_html_(), str)
