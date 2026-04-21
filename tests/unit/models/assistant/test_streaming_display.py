from __future__ import annotations

from pinecone.models.assistant.streaming import StreamContentDelta, StreamMessageStart


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
