from __future__ import annotations

from pinecone.models.assistant.streaming import StreamMessageStart


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
