"""Unit tests for __repr__ on RankedDocument and RerankResult."""

from __future__ import annotations

from pinecone.models.inference.rerank import RankedDocument, RerankResult, RerankUsage


class TestRankedDocumentRepr:
    def test_repr_with_long_text_truncates(self) -> None:
        doc = RankedDocument(index=0, score=0.95, document={"text": "x" * 500})
        r = repr(doc)
        assert len(r) < 200
        assert r.startswith("RankedDocument(")
        assert "index=0" in r
        assert "score=0.95" in r
        assert "..." in r

    def test_repr_with_short_text_no_truncation(self) -> None:
        doc = RankedDocument(index=1, score=0.72, document={"text": "short text"})
        r = repr(doc)
        assert "short text" in r
        assert "..." not in r
        assert "index=1" in r
        assert "score=0.72" in r

    def test_repr_with_none_document(self) -> None:
        doc = RankedDocument(index=2, score=0.31)
        r = repr(doc)
        assert r == "RankedDocument(index=2, score=0.31, document=None)"

    def test_repr_truncates_at_80_chars(self) -> None:
        text = "A" * 80 + "B" * 20
        doc = RankedDocument(index=0, score=0.5, document={"text": text})
        r = repr(doc)
        assert "A" * 80 + "..." in r
        assert "B" not in r

    def test_repr_exactly_80_chars_not_truncated(self) -> None:
        text = "A" * 80
        doc = RankedDocument(index=0, score=0.5, document={"text": text})
        r = repr(doc)
        assert "..." not in r
        assert text in r

    def test_repr_with_non_text_fields(self) -> None:
        doc = RankedDocument(index=0, score=0.8, document={"id": "abc", "score": 42})
        r = repr(doc)
        assert "id" in r
        assert "abc" in r

    def test_verify_command_assertion(self) -> None:
        """Replicates the task's verify command."""
        d = RankedDocument(index=0, score=0.95, document={"text": "x" * 500})
        r = repr(d)
        assert len(r) < 200, f"repr too long: {len(r)}"


class TestRerankResultRepr:
    def test_repr_shows_summary_not_full_dump(self) -> None:
        docs = [
            RankedDocument(index=i, score=float(i) / 10, document={"text": "doc " * 100})
            for i in range(20)
        ]
        usage = RerankUsage(rerank_units=1)
        result = RerankResult(model="bge-reranker-v2-m3", data=docs, usage=usage)
        r = repr(result)
        assert r.startswith("RerankResult(")
        assert "model='bge-reranker-v2-m3'" in r
        assert "count=20" in r
        assert "usage=" in r
        assert "rerank_units=1" in r
        # Should NOT dump individual documents
        assert "RankedDocument" not in r

    def test_repr_count_matches_data_length(self) -> None:
        docs = [RankedDocument(index=i, score=0.9) for i in range(5)]
        usage = RerankUsage(rerank_units=2)
        result = RerankResult(model="pinecone-rerank-v0", data=docs, usage=usage)
        r = repr(result)
        assert "count=5" in r

    def test_repr_empty_data(self) -> None:
        usage = RerankUsage(rerank_units=0)
        result = RerankResult(model="bge-reranker-v2-m3", data=[], usage=usage)
        r = repr(result)
        assert "count=0" in r
