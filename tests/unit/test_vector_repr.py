"""Unit tests for __repr__ on Vector, ScoredVector, and Hit."""

from __future__ import annotations

from pinecone.models.vectors.search import Hit
from pinecone.models.vectors.sparse import SparseValues
from pinecone.models.vectors.vector import ScoredVector, Vector


class TestVectorRepr:
    def test_vector_repr_truncates_long_values(self) -> None:
        values = [float(i) * 0.001 for i in range(1536)]
        v = Vector(id="v1", values=values)
        r = repr(v)
        assert r.startswith("Vector(")
        assert "...1533 more" in r
        assert repr(values[0]) in r
        assert repr(values[1]) in r
        assert repr(values[2]) in r
        # Should not contain the 4th value directly in the truncated section
        assert "id='v1'" in r

    def test_vector_repr_short_values(self) -> None:
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        v = Vector(id="v2", values=values)
        r = repr(v)
        assert "...more" not in r
        assert "[0.1, 0.2, 0.3, 0.4, 0.5]" in r
        assert "id='v2'" in r

    def test_vector_repr_exactly_five_values_not_truncated(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        v = Vector(id="v3", values=values)
        r = repr(v)
        assert "...more" not in r
        assert "[1.0, 2.0, 3.0, 4.0, 5.0]" in r

    def test_vector_repr_six_values_truncated(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        v = Vector(id="v4", values=values)
        r = repr(v)
        assert "...3 more" in r

    def test_vector_repr_omits_metadata_when_none(self) -> None:
        v = Vector(id="v5", values=[1.0, 2.0])
        r = repr(v)
        assert "metadata" not in r

    def test_vector_repr_includes_metadata_when_present(self) -> None:
        v = Vector(id="v6", values=[1.0], metadata={"key": "val"})
        r = repr(v)
        assert "metadata=" in r
        assert "'key'" in r

    def test_vector_repr_shows_sparse_values_none(self) -> None:
        v = Vector(id="v7", values=[1.0])
        r = repr(v)
        assert "sparse_values=None" in r

    def test_vector_repr_shows_sparse_values_when_present(self) -> None:
        sv = SparseValues(indices=[0], values=[0.5])
        v = Vector(id="v8", values=[1.0], sparse_values=sv)
        r = repr(v)
        assert "sparse_values=" in r
        assert "None" not in r.split("sparse_values=")[1].split(",")[0]


class TestScoredVectorRepr:
    def test_scored_vector_repr_omits_none_fields(self) -> None:
        sv = ScoredVector(id="v1", score=0.95)
        r = repr(sv)
        assert "sparse_values" not in r
        assert "metadata" not in r
        assert "id='v1'" in r
        assert "score=0.95" in r

    def test_scored_vector_repr_truncates_long_values(self) -> None:
        values = [0.001 * i for i in range(1536)]
        sv = ScoredVector(id="v2", score=0.8, values=values)
        r = repr(sv)
        assert "...1533 more" in r
        assert "score=0.8" in r

    def test_scored_vector_repr_short_values_shown_fully(self) -> None:
        sv = ScoredVector(id="v3", score=0.7, values=[0.1, 0.2, 0.3])
        r = repr(sv)
        assert "[0.1, 0.2, 0.3]" in r
        assert "...more" not in r

    def test_scored_vector_repr_includes_sparse_when_present(self) -> None:
        sparse = SparseValues(indices=[1], values=[0.5])
        sv = ScoredVector(id="v4", score=0.6, sparse_values=sparse)
        r = repr(sv)
        assert "sparse_values=" in r

    def test_scored_vector_repr_includes_metadata_when_present(self) -> None:
        sv = ScoredVector(id="v5", score=0.5, metadata={"tag": "test"})
        r = repr(sv)
        assert "metadata=" in r
        assert "'tag'" in r


class TestHitRepr:
    def test_hit_repr_uses_user_facing_names(self) -> None:
        hit = Hit(id_="rec-1", score_=0.99, fields={"text": "hello"})
        r = repr(hit)
        assert "id=" in r
        assert "score=" in r
        assert "id_=" not in r
        assert "score_=" not in r

    def test_hit_repr_contains_correct_values(self) -> None:
        hit = Hit(id_="rec-2", score_=0.75, fields={"text": "world"})
        r = repr(hit)
        assert "id='rec-2'" in r
        assert "score=0.75" in r
        assert "fields=" in r

    def test_hit_repr_format(self) -> None:
        hit = Hit(id_="x", score_=1.0, fields={})
        r = repr(hit)
        assert r == "Hit(id='x', score=1.0, fields={})"
