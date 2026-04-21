from __future__ import annotations

from pinecone.models.assistant.chat import ChatCitation, ChatReference
from pinecone.models.assistant.file_model import AssistantFileModel
from pinecone.models.assistant.streaming import (
    StreamCitationChunk,
    StreamContentChunk,
    StreamContentDelta,
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
