"""Unit tests for __repr__ on SparseValues, DenseEmbedding, SparseEmbedding, and EmbeddingsList."""

from __future__ import annotations

from pinecone.models.inference.embed import (
    DenseEmbedding,
    EmbeddingsList,
    EmbedUsage,
    SparseEmbedding,
)
from pinecone.models.vectors.sparse import SparseValues


class TestSparseValuesRepr:
    def test_sparse_values_repr_truncates_long_lists(self) -> None:
        indices = list(range(1000))
        values = [float(i) * 0.001 for i in range(1000)]
        sv = SparseValues(indices=indices, values=values)
        r = repr(sv)
        assert r.startswith("SparseValues(")
        assert "...997 more" in r
        assert repr(indices[0]) in r
        assert repr(indices[1]) in r
        assert repr(indices[2]) in r
        assert repr(values[0]) in r
        assert repr(values[1]) in r
        assert repr(values[2]) in r

    def test_sparse_values_repr_short_lists(self) -> None:
        sv = SparseValues(indices=[1, 42, 103], values=[0.5, 0.3, 0.1])
        r = repr(sv)
        assert "...more" not in r
        assert "indices=[1, 42, 103]" in r
        assert "values=[0.5, 0.3, 0.1]" in r

    def test_sparse_values_repr_exactly_five_not_truncated(self) -> None:
        sv = SparseValues(indices=[1, 2, 3, 4, 5], values=[0.1, 0.2, 0.3, 0.4, 0.5])
        r = repr(sv)
        assert "...more" not in r

    def test_sparse_values_repr_six_elements_truncated(self) -> None:
        sv = SparseValues(
            indices=[1, 2, 3, 4, 5, 6], values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        )
        r = repr(sv)
        assert "...3 more" in r


class TestDenseEmbeddingRepr:
    def test_dense_embedding_repr_truncates(self) -> None:
        values = [float(i) * 0.001 for i in range(1536)]
        emb = DenseEmbedding(values=values)
        r = repr(emb)
        assert r.startswith("DenseEmbedding(")
        assert "...1533 more" in r
        assert repr(values[0]) in r
        assert repr(values[1]) in r
        assert repr(values[2]) in r
        assert "vector_type='dense'" in r

    def test_dense_embedding_repr_short_values(self) -> None:
        emb = DenseEmbedding(values=[0.1, 0.2, 0.3])
        r = repr(emb)
        assert "...more" not in r
        assert "[0.1, 0.2, 0.3]" in r
        assert "vector_type='dense'" in r

    def test_dense_embedding_repr_exactly_five_not_truncated(self) -> None:
        emb = DenseEmbedding(values=[1.0, 2.0, 3.0, 4.0, 5.0])
        r = repr(emb)
        assert "...more" not in r

    def test_dense_embedding_repr_six_elements_truncated(self) -> None:
        emb = DenseEmbedding(values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        r = repr(emb)
        assert "...3 more" in r


class TestSparseEmbeddingRepr:
    def test_sparse_embedding_repr_truncates(self) -> None:
        indices = list(range(1000))
        values = [float(i) * 0.001 for i in range(1000)]
        emb = SparseEmbedding(sparse_indices=indices, sparse_values=values)
        r = repr(emb)
        assert r.startswith("SparseEmbedding(")
        assert "...997 more" in r
        assert "vector_type='sparse'" in r

    def test_sparse_embedding_repr_short_lists(self) -> None:
        emb = SparseEmbedding(
            sparse_indices=[1, 42, 103], sparse_values=[0.5, 0.3, 0.1]
        )
        r = repr(emb)
        assert "...more" not in r
        assert "sparse_indices=[1, 42, 103]" in r
        assert "sparse_values=[0.5, 0.3, 0.1]" in r

    def test_sparse_embedding_repr_omits_sparse_tokens_when_none(self) -> None:
        emb = SparseEmbedding(sparse_indices=[1], sparse_values=[0.5])
        r = repr(emb)
        assert "sparse_tokens" not in r

    def test_sparse_embedding_repr_includes_sparse_tokens_when_present(self) -> None:
        emb = SparseEmbedding(
            sparse_indices=[1, 2],
            sparse_values=[0.5, 0.3],
            sparse_tokens=["hello", "world"],
        )
        r = repr(emb)
        assert "sparse_tokens=" in r
        assert "hello" in r


class TestEmbeddingsListRepr:
    def test_embeddings_list_repr_summary(self) -> None:
        embeddings = [DenseEmbedding(values=[0.1, 0.2]) for _ in range(10)]
        usage = EmbedUsage(total_tokens=512)
        el = EmbeddingsList(
            model="multilingual-e5-large",
            vector_type="dense",
            data=embeddings,
            usage=usage,
        )
        r = repr(el)
        assert r.startswith("EmbeddingsList(")
        assert "model='multilingual-e5-large'" in r
        assert "vector_type='dense'" in r
        assert "count=10" in r
        assert "usage=" in r
        assert "512" in r
        # Should NOT dump individual embeddings
        assert "DenseEmbedding" not in r

    def test_embeddings_list_repr_count_matches_data_length(self) -> None:
        embeddings = [DenseEmbedding(values=[float(i)]) for i in range(3)]
        usage = EmbedUsage(total_tokens=100)
        el = EmbeddingsList(
            model="text-embedding-ada-002",
            vector_type="dense",
            data=embeddings,
            usage=usage,
        )
        r = repr(el)
        assert "count=3" in r
