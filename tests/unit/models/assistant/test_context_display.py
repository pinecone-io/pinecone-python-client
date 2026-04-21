from __future__ import annotations

from pinecone.models.assistant.context import ContextImageBlock, ContextImageData


class TestContextImageData:
    def test_repr_does_not_dump_raw_data(self) -> None:
        big = "A" * 2_000_000
        r = repr(ContextImageData(type="base64", mime_type="image/jpeg", data=big))
        assert big not in r
        assert len(r) < 500

    def test_repr(self) -> None:
        r = repr(ContextImageData(type="base64", mime_type="image/png", data="AAA"))
        assert "image/png" in r

    def test_repr_html(self) -> None:
        h = ContextImageData(type="base64", mime_type="image/png", data="A" * 10_000)._repr_html_()
        assert h is not None
        assert len(h) < 5000
        assert "image/png" in h

    def test_safe_on_malformed(self) -> None:
        d = ContextImageData(type="base64", mime_type="image/png", data="x")
        d.mime_type = object()  # type: ignore[assignment]
        assert isinstance(repr(d), str)
        assert isinstance(d._repr_html_(), str)


class TestContextImageBlock:
    def test_repr_with_image(self) -> None:
        b = ContextImageBlock(caption="A cat", image_data=ContextImageData(type="base64", mime_type="image/jpeg", data="X" * 500))
        r = repr(b)
        assert "A cat" in r
        assert len(r) < 500
        assert "X" * 500 not in r

    def test_repr_without_image(self) -> None:
        assert "absent" in repr(ContextImageBlock(caption="x")).lower() or "None" not in repr(ContextImageBlock(caption="x"))

    def test_repr_long_caption_truncated(self) -> None:
        assert len(repr(ContextImageBlock(caption="x" * 5000))) < 500

    def test_repr_html(self) -> None:
        h = ContextImageBlock(caption="cat")._repr_html_()
        assert h is not None
        assert "cat" in h

    def test_safe_on_malformed(self) -> None:
        b = ContextImageBlock(caption="x")
        b.caption = object()  # type: ignore[assignment]
        assert isinstance(repr(b), str)
        assert isinstance(b._repr_html_(), str)
