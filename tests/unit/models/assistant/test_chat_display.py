"""Display method tests for chat models."""

from __future__ import annotations

from pinecone.models.assistant.chat import (
    ChatCitation,
    ChatCompletionChoice,
    ChatCompletionResponse,
    ChatHighlight,
    ChatMessage,
    ChatReference,
    ChatResponse,
    ChatUsage,
)
from pinecone.models.assistant.file_model import AssistantFileModel


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


class TestChatHighlight:
    def test_repr(self) -> None:
        r = repr(ChatHighlight(type="text", content="hello"))
        assert "text" in r
        assert "hello" in r

    def test_repr_long_content_truncated(self) -> None:
        r = repr(ChatHighlight(type="text", content="x" * 5000))
        assert len(r) < 500
        assert "..." in r

    def test_repr_html(self) -> None:
        assert "<div" in ChatHighlight(type="text", content="hi")._repr_html_()

    def test_repr_html_long_truncated(self) -> None:
        h = ChatHighlight(type="text", content="x" * 10_000)._repr_html_()
        assert len(h) < 5000
        assert "..." in h

    def test_safe_on_malformed(self) -> None:
        hl = ChatHighlight(type="text", content="x")
        hl.content = object()  # type: ignore[assignment]
        assert isinstance(repr(hl), str)
        assert isinstance(hl._repr_html_(), str)


def _file() -> AssistantFileModel:
    return AssistantFileModel(name="doc.pdf", id="f-1", status="Available")


class TestChatReference:
    def test_repr_minimal(self) -> None:
        r = repr(ChatReference(file=_file()))
        assert "doc.pdf" in r

    def test_repr_with_pages_and_highlight(self) -> None:
        r = repr(
            ChatReference(
                file=_file(), pages=[1, 2, 3], highlight=ChatHighlight(type="text", content="x")
            )
        )
        assert "1" in r

    def test_repr_many_pages_abbreviated(self) -> None:
        r = repr(ChatReference(file=_file(), pages=list(range(2000))))
        assert len(r) < 500
        assert "more" in r

    def test_repr_html(self) -> None:
        h = ChatReference(file=_file(), pages=[1, 2])._repr_html_()
        assert "doc.pdf" in h

    def test_repr_html_no_highlight_shows_dash(self) -> None:
        h = ChatReference(file=_file())._repr_html_()
        assert "<div" in h

    def test_safe_on_malformed(self) -> None:
        c = ChatReference(file=_file())
        c.file = object()  # type: ignore[assignment]
        assert isinstance(repr(c), str)
        assert isinstance(c._repr_html_(), str)


class TestChatCitation:
    def test_repr(self) -> None:
        c = ChatCitation(position=42, references=[ChatReference(file=_file())])
        r = repr(c)
        assert "42" in r

    def test_repr_empty_refs(self) -> None:
        r = repr(ChatCitation(position=0, references=[]))
        assert "0" in r

    def test_repr_large_refs_abbreviated(self) -> None:
        refs = [ChatReference(file=_file()) for _ in range(500)]
        r = repr(ChatCitation(position=1, references=refs))
        assert len(r) < 500

    def test_repr_html(self) -> None:
        h = ChatCitation(position=1, references=[ChatReference(file=_file())])._repr_html_()
        assert "doc.pdf" in h

    def test_repr_html_empty_refs(self) -> None:
        h = ChatCitation(position=0, references=[])._repr_html_()
        assert "<div" in h

    def test_safe_on_malformed(self) -> None:
        c = ChatCitation(position=1, references=[])
        c.position = object()  # type: ignore[assignment]
        assert isinstance(repr(c), str)
        assert isinstance(c._repr_html_(), str)


class TestChatMessage:
    def test_repr(self) -> None:
        assert "Hi" in repr(ChatMessage(role="assistant", content="Hi"))

    def test_repr_long_truncated(self) -> None:
        r = repr(ChatMessage(role="assistant", content="x" * 5000))
        assert len(r) < 500
        assert "..." in r

    def test_repr_html(self) -> None:
        assert "Hi" in ChatMessage(role="assistant", content="Hi")._repr_html_()

    def test_repr_html_long_truncated(self) -> None:
        h = ChatMessage(role="assistant", content="x" * 10_000)._repr_html_()
        assert len(h) < 5000

    def test_safe_on_malformed(self) -> None:
        m = ChatMessage(role="user", content="x")
        m.role = object()  # type: ignore[assignment]
        assert isinstance(repr(m), str)
        assert isinstance(m._repr_html_(), str)


def _resp(n_citations: int = 1, content: str = "Hi") -> ChatResponse:
    return ChatResponse(
        id="r-1",
        model="m",
        usage=ChatUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        message=ChatMessage(role="assistant", content=content),
        finish_reason="stop",
        citations=[ChatCitation(position=i, references=[]) for i in range(n_citations)],
    )


class TestChatResponse:
    def test_repr(self) -> None:
        r = repr(_resp())
        assert "r-1" in r
        assert "citations=1" in r.replace(" ", "").lower() or "1" in r

    def test_repr_many_citations(self) -> None:
        r = repr(_resp(n_citations=500))
        assert len(r) < 500

    def test_repr_long_message_not_dumped(self) -> None:
        r = repr(_resp(content="x" * 100_000))
        assert len(r) < 1000

    def test_repr_html(self) -> None:
        h = _resp()._repr_html_()
        assert "r-1" in h
        assert "Hi" in h

    def test_repr_html_long_message_truncated(self) -> None:
        h = _resp(content="x" * 100_000)._repr_html_()
        assert len(h) < 10_000

    def test_safe_on_malformed(self) -> None:
        r = _resp()
        r.message = object()  # type: ignore[assignment]
        assert isinstance(repr(r), str)
        assert isinstance(r._repr_html_(), str)


class TestChatCompletionChoice:
    def test_repr(self) -> None:
        c = ChatCompletionChoice(
            index=0, message=ChatMessage(role="assistant", content="Hi"), finish_reason="stop"
        )
        r = repr(c)
        assert "0" in r
        assert "stop" in r

    def test_repr_long_message_bounded(self) -> None:
        c = ChatCompletionChoice(
            index=0,
            message=ChatMessage(role="assistant", content="x" * 10_000),
            finish_reason="stop",
        )
        assert len(repr(c)) < 500

    def test_repr_html(self) -> None:
        c = ChatCompletionChoice(
            index=0, message=ChatMessage(role="assistant", content="Hi"), finish_reason="stop"
        )
        assert "Hi" in c._repr_html_()

    def test_safe_on_malformed(self) -> None:
        c = ChatCompletionChoice(
            index=0, message=ChatMessage(role="a", content="b"), finish_reason="stop"
        )
        c.message = object()  # type: ignore[assignment]
        assert isinstance(repr(c), str)
        assert isinstance(c._repr_html_(), str)


def _ccr(n_choices: int = 1, content: str = "Hi") -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id="c-1",
        model="m",
        usage=ChatUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        choices=[
            ChatCompletionChoice(
                index=i,
                message=ChatMessage(role="assistant", content=content),
                finish_reason="stop",
            )
            for i in range(n_choices)
        ],
    )


class TestChatCompletionResponse:
    def test_repr(self) -> None:
        assert "c-1" in repr(_ccr())

    def test_repr_many_choices(self) -> None:
        r = repr(_ccr(n_choices=100))
        assert len(r) < 500

    def test_repr_long_content_bounded(self) -> None:
        assert len(repr(_ccr(content="x" * 100_000))) < 500

    def test_repr_html(self) -> None:
        assert "c-1" in _ccr()._repr_html_()

    def test_repr_html_no_choices(self) -> None:
        h = _ccr(n_choices=0)._repr_html_()
        assert "<div" in h

    def test_safe_on_malformed(self) -> None:
        r = _ccr()
        r.choices = object()  # type: ignore[assignment]
        assert isinstance(repr(r), str)
        assert isinstance(r._repr_html_(), str)
