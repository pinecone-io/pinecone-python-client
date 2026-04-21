from __future__ import annotations

from pinecone.models.assistant.chat import ChatUsage
from pinecone.models.assistant.context import (
    ContextImageBlock,
    ContextImageData,
    ContextResponse,
    ContextTextBlock,
    FileReference,
    MultimodalSnippet,
    TextSnippet,
)
from pinecone.models.assistant.file_model import AssistantFileModel


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

    def test_repr_pretty_normal(self) -> None:
        from IPython.lib.pretty import pretty

        r = pretty(ContextImageData(type="base64", mime_type="image/png", data="AAA"))
        assert "image/png" in r

    def test_repr_pretty_cycle(self) -> None:
        import io

        from IPython.lib.pretty import RepresentationPrinter

        buf = io.StringIO()
        printer = RepresentationPrinter(buf)
        ContextImageData(type="base64", mime_type="image/png", data="x")._repr_pretty_(
            printer, cycle=True
        )
        printer.flush()
        assert "ContextImageData(...)" in buf.getvalue()

    def test_safe_on_malformed(self) -> None:
        d = ContextImageData(type="base64", mime_type="image/png", data="x")
        d.mime_type = object()  # type: ignore[assignment]
        assert isinstance(repr(d), str)
        assert isinstance(d._repr_html_(), str)


class TestContextImageBlock:
    def test_repr_with_image(self) -> None:
        b = ContextImageBlock(
            caption="A cat",
            image_data=ContextImageData(type="base64", mime_type="image/jpeg", data="X" * 500),
        )
        r = repr(b)
        assert "A cat" in r
        assert len(r) < 500
        assert "X" * 500 not in r

    def test_repr_without_image(self) -> None:
        assert "absent" in repr(ContextImageBlock(caption="x")).lower() or "None" not in repr(
            ContextImageBlock(caption="x")
        )

    def test_repr_long_caption_truncated(self) -> None:
        assert len(repr(ContextImageBlock(caption="x" * 5000))) < 500

    def test_repr_html(self) -> None:
        h = ContextImageBlock(caption="cat")._repr_html_()
        assert h is not None
        assert "cat" in h

    def test_repr_pretty_normal(self) -> None:
        from IPython.lib.pretty import pretty

        r = pretty(ContextImageBlock(caption="A cat"))
        assert "A cat" in r

    def test_repr_pretty_cycle(self) -> None:
        import io

        from IPython.lib.pretty import RepresentationPrinter

        buf = io.StringIO()
        printer = RepresentationPrinter(buf)
        ContextImageBlock(caption="x")._repr_pretty_(printer, cycle=True)
        printer.flush()
        assert "ContextImageBlock(...)" in buf.getvalue()

    def test_safe_on_malformed(self) -> None:
        b = ContextImageBlock(caption="x")
        b.caption = object()  # type: ignore[assignment]
        assert isinstance(repr(b), str)
        assert isinstance(b._repr_html_(), str)


class TestContextTextBlock:
    def test_repr(self) -> None:
        assert "hello" in repr(ContextTextBlock(text="hello"))

    def test_repr_long_truncated(self) -> None:
        assert len(repr(ContextTextBlock(text="x" * 5000))) < 500

    def test_repr_html(self) -> None:
        assert "<div" in ContextTextBlock(text="hi")._repr_html_()

    def test_repr_html_long_truncated(self) -> None:
        h = ContextTextBlock(text="x" * 10_000)._repr_html_()
        assert len(h) < 5000

    def test_repr_pretty_normal(self) -> None:
        from IPython.lib.pretty import pretty

        r = pretty(ContextTextBlock(text="hello"))
        assert "hello" in r

    def test_repr_pretty_cycle(self) -> None:
        import io

        from IPython.lib.pretty import RepresentationPrinter

        buf = io.StringIO()
        printer = RepresentationPrinter(buf)
        ContextTextBlock(text="hi")._repr_pretty_(printer, cycle=True)
        printer.flush()
        assert "ContextTextBlock(...)" in buf.getvalue()

    def test_safe_on_malformed(self) -> None:
        b = ContextTextBlock(text="x")
        b.text = object()  # type: ignore[assignment]
        assert isinstance(repr(b), str)
        assert isinstance(b._repr_html_(), str)


def _f() -> AssistantFileModel:
    return AssistantFileModel(name="doc.pdf", id="f-1")


class TestFileReference:
    def test_repr(self) -> None:
        r = repr(FileReference(file=_f(), pages=[1, 2]))
        assert "doc.pdf" in r
        assert "1" in r

    def test_repr_no_pages(self) -> None:
        assert "doc.pdf" in repr(FileReference(file=_f()))

    def test_repr_many_pages_abbreviated(self) -> None:
        r = repr(FileReference(file=_f(), pages=list(range(1000))))
        assert len(r) < 500
        assert "more" in r

    def test_repr_html(self) -> None:
        assert "doc.pdf" in FileReference(file=_f(), pages=[1])._repr_html_()

    def test_repr_html_no_pages(self) -> None:
        h = FileReference(file=_f())._repr_html_()
        assert "<div" in h

    def test_repr_pretty_normal(self) -> None:
        from IPython.lib.pretty import pretty

        r = pretty(FileReference(file=_f(), pages=[1, 2]))
        assert "doc.pdf" in r

    def test_repr_pretty_cycle(self) -> None:
        import io

        from IPython.lib.pretty import RepresentationPrinter

        buf = io.StringIO()
        printer = RepresentationPrinter(buf)
        FileReference(file=_f())._repr_pretty_(printer, cycle=True)
        printer.flush()
        assert "FileReference(...)" in buf.getvalue()

    def test_safe_on_malformed(self) -> None:
        r = FileReference(file=_f())
        r.file = object()  # type: ignore[assignment]
        assert isinstance(repr(r), str)
        assert isinstance(r._repr_html_(), str)


def _ref() -> FileReference:
    return FileReference(file=_f(), pages=[1, 2])


class TestTextSnippet:
    def test_repr(self) -> None:
        r = repr(TextSnippet(content="hello", score=0.9, reference=_ref()))
        assert "0.9" in r
        assert "hello" in r

    def test_repr_long_content_truncated(self) -> None:
        s = TextSnippet(content="x" * 10_000, score=0.5, reference=_ref())
        assert len(repr(s)) < 500

    def test_repr_html(self) -> None:
        h = TextSnippet(content="hi", score=0.9, reference=_ref())._repr_html_()
        assert "hi" in h
        assert "doc.pdf" in h

    def test_repr_html_long_truncated(self) -> None:
        h = TextSnippet(content="x" * 100_000, score=1.0, reference=_ref())._repr_html_()
        assert len(h) < 5000

    def test_repr_pretty_normal(self) -> None:
        from IPython.lib.pretty import pretty

        r = pretty(TextSnippet(content="hello", score=0.9, reference=_ref()))
        assert "0.9" in r
        assert "hello" in r

    def test_repr_pretty_cycle(self) -> None:
        import io

        from IPython.lib.pretty import RepresentationPrinter

        buf = io.StringIO()
        printer = RepresentationPrinter(buf)
        TextSnippet(content="x", score=0.5, reference=_ref())._repr_pretty_(printer, cycle=True)
        printer.flush()
        assert "TextSnippet(...)" in buf.getvalue()

    def test_safe_on_malformed(self) -> None:
        s = TextSnippet(content="x", score=0.0, reference=_ref())
        s.reference = object()  # type: ignore[assignment]
        assert isinstance(repr(s), str)
        assert isinstance(s._repr_html_(), str)


class TestMultimodalSnippet:
    def _blocks(self) -> list:  # type: ignore[type-arg]
        return [
            ContextTextBlock(text="hi"),
            ContextImageBlock(
                caption="A cat",
                image_data=ContextImageData(type="base64", mime_type="image/png", data="AAA"),
            ),
        ]

    def test_repr(self) -> None:
        m = MultimodalSnippet(content=self._blocks(), score=0.7, reference=_ref())
        r = repr(m)
        assert "0.7" in r

    def test_repr_many_blocks_bounded(self) -> None:
        blocks = [ContextTextBlock(text="hi") for _ in range(200)]
        m = MultimodalSnippet(content=blocks, score=0.5, reference=_ref())
        assert len(repr(m)) < 500

    def test_repr_html(self) -> None:
        h = MultimodalSnippet(content=self._blocks(), score=0.5, reference=_ref())._repr_html_()
        assert "doc.pdf" in h

    def test_repr_html_empty_content(self) -> None:
        h = MultimodalSnippet(content=[], score=0.0, reference=_ref())._repr_html_()
        assert "<div" in h

    def test_repr_pretty_normal(self) -> None:
        from IPython.lib.pretty import pretty

        m = MultimodalSnippet(content=self._blocks(), score=0.7, reference=_ref())
        r = pretty(m)
        assert "0.7" in r

    def test_repr_pretty_cycle(self) -> None:
        import io

        from IPython.lib.pretty import RepresentationPrinter

        buf = io.StringIO()
        printer = RepresentationPrinter(buf)
        MultimodalSnippet(content=[], score=0.0, reference=_ref())._repr_pretty_(
            printer, cycle=True
        )
        printer.flush()
        assert "MultimodalSnippet(...)" in buf.getvalue()

    def test_safe_on_malformed(self) -> None:
        m = MultimodalSnippet(content=[], score=0.0, reference=_ref())
        m.content = object()  # type: ignore[assignment]
        assert isinstance(repr(m), str)
        assert isinstance(m._repr_html_(), str)


def _usage() -> ChatUsage:
    return ChatUsage(prompt_tokens=1, completion_tokens=0, total_tokens=1)


class TestContextResponse:
    def test_repr_populated(self) -> None:
        snippets = [TextSnippet(content="hi", score=0.9, reference=_ref())]
        r = repr(ContextResponse(snippets=snippets, usage=_usage(), id="c-1"))
        assert "c-1" in r

    def test_repr_large_snippets(self) -> None:
        s = [TextSnippet(content="x" * 1000, score=0.5, reference=_ref()) for _ in range(100)]
        assert len(repr(ContextResponse(snippets=s, usage=_usage()))) < 500

    def test_repr_html(self) -> None:
        snippets = [TextSnippet(content="hi", score=0.9, reference=_ref())]
        h = ContextResponse(snippets=snippets, usage=_usage())._repr_html_()
        assert "doc.pdf" in h

    def test_repr_html_empty_snippets(self) -> None:
        h = ContextResponse(snippets=[], usage=_usage())._repr_html_()
        assert "<div" in h

    def test_repr_pretty_normal(self) -> None:
        from IPython.lib.pretty import pretty

        snippets = [TextSnippet(content="hi", score=0.9, reference=_ref())]
        r = pretty(ContextResponse(snippets=snippets, usage=_usage(), id="c-1"))
        assert "c-1" in r
        assert "1" in r

    def test_repr_pretty_cycle(self) -> None:
        import io

        from IPython.lib.pretty import RepresentationPrinter

        buf = io.StringIO()
        printer = RepresentationPrinter(buf)
        ContextResponse(snippets=[], usage=_usage())._repr_pretty_(printer, cycle=True)
        printer.flush()
        assert "ContextResponse(...)" in buf.getvalue()

    def test_safe_on_malformed(self) -> None:
        r = ContextResponse(snippets=[], usage=_usage())
        r.snippets = object()  # type: ignore[assignment]
        assert isinstance(repr(r), str)
        assert isinstance(r._repr_html_(), str)
