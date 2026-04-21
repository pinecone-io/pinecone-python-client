from __future__ import annotations

from collections.abc import Iterator

from pinecone.models.assistant.chat import ChatCitation, ChatReference, ChatUsage
from pinecone.models.assistant.file_model import AssistantFileModel
from pinecone.models.assistant.streaming import (
    ChatCompletionStreamChoice,
    ChatCompletionStreamChunk,
    ChatCompletionStreamDelta,
    ChatStream,
    ChatStreamChunk,
    StreamCitationChunk,
    StreamContentChunk,
    StreamContentDelta,
    StreamMessageEnd,
    StreamMessageStart,
)


class TestStreamMessageStart:
    def test_repr(self) -> None:
        r = repr(StreamMessageStart(model="m", role="assistant"))
        assert "m" in r
        assert "assistant" in r

    def test_repr_html(self) -> None:
        h = StreamMessageStart(model="m", role="assistant")._repr_html_()
        assert "m" in h

    def test_safe_on_malformed(self) -> None:
        s = StreamMessageStart(model="m", role="r")
        s.model = object()  # type: ignore[assignment]
        result_repr = repr(s)
        assert isinstance(result_repr, str)
        result_html = s._repr_html_()
        assert isinstance(result_html, str)


class TestStreamContentDelta:
    def test_repr(self) -> None:
        assert "hi" in repr(StreamContentDelta(content="hi"))

    def test_repr_long_truncated(self) -> None:
        assert len(repr(StreamContentDelta(content="x" * 5000))) < 500

    def test_repr_html(self) -> None:
        assert "hi" in StreamContentDelta(content="hi")._repr_html_()

    def test_safe_on_malformed(self) -> None:
        d = StreamContentDelta(content="x")
        d.content = object()  # type: ignore[assignment]
        assert isinstance(repr(d), str)
        assert isinstance(d._repr_html_(), str)


class TestStreamContentChunk:
    def test_repr(self) -> None:
        c = StreamContentChunk(id="c-1", delta=StreamContentDelta(content="hi"), model="m")
        r = repr(c)
        assert "c-1" in r

    def test_repr_no_model(self) -> None:
        c = StreamContentChunk(id="c-1", delta=StreamContentDelta(content="hi"))
        assert "None" not in repr(c)

    def test_repr_long_content_truncated(self) -> None:
        c = StreamContentChunk(id="c-1", delta=StreamContentDelta(content="x" * 10_000))
        assert len(repr(c)) < 500

    def test_repr_html(self) -> None:
        c = StreamContentChunk(id="c-1", delta=StreamContentDelta(content="hi"))
        assert "hi" in c._repr_html_()

    def test_safe_on_malformed(self) -> None:
        c = StreamContentChunk(id="c-1", delta=StreamContentDelta(content="x"))
        c.delta = object()  # type: ignore[assignment]
        assert isinstance(repr(c), str)
        assert isinstance(c._repr_html_(), str)


def _cit() -> ChatCitation:
    return ChatCitation(
        position=1,
        references=[ChatReference(file=AssistantFileModel(name="d.pdf", id="f"))],
    )


class TestStreamCitationChunk:
    def test_repr(self) -> None:
        c = StreamCitationChunk(id="c-1", citation=_cit())
        assert "c-1" in repr(c)

    def test_repr_html(self) -> None:
        assert "<div" in StreamCitationChunk(id="c-1", citation=_cit())._repr_html_()

    def test_safe_on_malformed(self) -> None:
        c = StreamCitationChunk(id="c-1", citation=_cit())
        c.citation = object()  # type: ignore[assignment]
        assert isinstance(repr(c), str)
        assert isinstance(c._repr_html_(), str)


class TestChatCompletionStreamDelta:
    def test_repr_full(self) -> None:
        r = repr(ChatCompletionStreamDelta(role="assistant", content="hi"))
        assert "assistant" in r
        assert "hi" in r

    def test_repr_empty(self) -> None:
        r = repr(ChatCompletionStreamDelta())
        assert "ChatCompletionStreamDelta" in r

    def test_repr_long_content_truncated(self) -> None:
        assert len(repr(ChatCompletionStreamDelta(content="x" * 5000))) < 500

    def test_repr_html(self) -> None:
        h = ChatCompletionStreamDelta(role="assistant", content="hi")._repr_html_()
        assert "<div" in h

    def test_safe_on_malformed(self) -> None:
        d = ChatCompletionStreamDelta(role="assistant")
        d.role = object()  # type: ignore[assignment]
        assert isinstance(repr(d), str)
        assert isinstance(d._repr_html_(), str)


class TestStreamMessageEnd:
    def test_repr(self) -> None:
        e = StreamMessageEnd(
            id="e-1", usage=ChatUsage(prompt_tokens=5, completion_tokens=5, total_tokens=10)
        )
        assert "e-1" in repr(e)

    def test_repr_html(self) -> None:
        e = StreamMessageEnd(
            id="e-1", usage=ChatUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        )
        assert "<div" in e._repr_html_()

    def test_safe_on_malformed(self) -> None:
        e = StreamMessageEnd(
            id="x", usage=ChatUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        )
        e.usage = object()  # type: ignore[assignment]
        assert isinstance(repr(e), str)
        assert isinstance(e._repr_html_(), str)


class TestChatCompletionStreamChoice:
    def test_repr(self) -> None:
        c = ChatCompletionStreamChoice(
            index=0, delta=ChatCompletionStreamDelta(content="hi"), finish_reason="stop"
        )
        r = repr(c)
        assert "0" in r
        assert "stop" in r

    def test_repr_without_finish_reason(self) -> None:
        c = ChatCompletionStreamChoice(index=0, delta=ChatCompletionStreamDelta(content="hi"))
        assert "None" not in repr(c)

    def test_repr_html(self) -> None:
        c = ChatCompletionStreamChoice(index=0, delta=ChatCompletionStreamDelta(content="hi"))
        assert "<div" in c._repr_html_()

    def test_safe_on_malformed(self) -> None:
        c = ChatCompletionStreamChoice(index=0, delta=ChatCompletionStreamDelta())
        c.delta = object()  # type: ignore[assignment]
        assert isinstance(repr(c), str)
        assert isinstance(c._repr_html_(), str)


def _chunk(n_choices: int = 1) -> ChatCompletionStreamChunk:
    return ChatCompletionStreamChunk(
        id="c-1",
        choices=[
            ChatCompletionStreamChoice(index=i, delta=ChatCompletionStreamDelta(content=f"c{i}"))
            for i in range(n_choices)
        ],
        model="m",
    )


class TestChatCompletionStreamChunk:
    def test_repr(self) -> None:
        assert "c-1" in repr(_chunk())

    def test_repr_many_choices(self) -> None:
        r = repr(_chunk(n_choices=50))
        assert len(r) < 500

    def test_repr_no_choices(self) -> None:
        r = repr(_chunk(n_choices=0))
        assert "c-1" in r

    def test_repr_html(self) -> None:
        assert "<div" in _chunk()._repr_html_()

    def test_safe_on_malformed(self) -> None:
        c = _chunk()
        c.choices = object()  # type: ignore[assignment]
        assert isinstance(repr(c), str)
        assert isinstance(c._repr_html_(), str)


class TestChatStreamDisplay:
    def _dummy_iter(self) -> Iterator[ChatStreamChunk]:
        def _gen() -> Iterator[ChatStreamChunk]:
            yield from []

        return _gen()

    def test_repr_does_not_consume(self) -> None:
        it = self._dummy_iter()
        s = ChatStream(it)
        r = repr(s)
        assert "ChatStream" in r
        assert "single-pass" in r
        # The iterator should still be intact — we didn't drain it.
        assert list(s) == []

    def test_repr_html(self) -> None:
        s = ChatStream(self._dummy_iter())
        h = s._repr_html_()
        assert "ChatStream" in h

    def test_safe_on_malformed(self) -> None:
        # Simulate a ChatStream whose _stream attribute is bad
        s = ChatStream(self._dummy_iter())
        object.__setattr__(s, "_stream", object())
        assert isinstance(repr(s), str)
        assert isinstance(s._repr_html_(), str)
