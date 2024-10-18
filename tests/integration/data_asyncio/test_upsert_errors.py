import os
import pytest
from pinecone.grpc import Vector, SparseValues
from ..helpers import fake_api_key
from .utils import build_asyncio_idx, embedding_values
from pinecone import PineconeException, PineconeApiValueError
from pinecone.grpc import PineconeGRPC as Pinecone


class TestUpsertApiKeyMissing:
    async def test_upsert_fails_when_api_key_invalid(self, host):
        with pytest.raises(PineconeException):
            pc = Pinecone(
                api_key=fake_api_key(),
                additional_headers={"sdk-test-suite": "pinecone-python-client"},
            )
            asyncio_idx = pc.AsyncioIndex(host=host)
            await asyncio_idx.upsert(
                vectors=[
                    Vector(id="1", values=embedding_values()),
                    Vector(id="2", values=embedding_values()),
                ]
            )

    @pytest.mark.skipif(
        os.getenv("USE_GRPC") != "true", reason="Only test grpc client when grpc extras"
    )
    async def test_upsert_fails_when_api_key_invalid_grpc(self, host):
        with pytest.raises(PineconeException):
            from pinecone.grpc import PineconeGRPC

            pc = PineconeGRPC(api_key=fake_api_key())
            asyncio_idx = pc.AsyncioIndex(host=host)
            await asyncio_idx.upsert(
                vectors=[
                    Vector(id="1", values=embedding_values()),
                    Vector(id="2", values=embedding_values()),
                ]
            )


class TestUpsertFailsWhenDimensionMismatch:
    async def test_upsert_fails_when_dimension_mismatch_objects(self, host):
        with pytest.raises(PineconeApiValueError):
            asyncio_idx = build_asyncio_idx(host)
            await asyncio_idx.upsert(
                vectors=[
                    Vector(id="1", values=embedding_values(2)),
                    Vector(id="2", values=embedding_values(3)),
                ]
            )

    async def test_upsert_fails_when_dimension_mismatch_tuples(self, host):
        asyncio_idx = build_asyncio_idx(host)
        with pytest.raises(PineconeException):
            await asyncio_idx.upsert(
                vectors=[("1", embedding_values(2)), ("2", embedding_values(3))]
            )

    async def test_upsert_fails_when_dimension_mismatch_dicts(self, host):
        asyncio_idx = build_asyncio_idx(host)
        with pytest.raises(PineconeException):
            await asyncio_idx.upsert(
                vectors=[
                    {"id": "1", "values": embedding_values(2)},
                    {"id": "2", "values": embedding_values(3)},
                ]
            )


@pytest.mark.skipif(
    os.getenv("METRIC") != "dotproduct", reason="Only metric=dotprodouct indexes support hybrid"
)
class TestUpsertFailsSparseValuesDimensionMismatch:
    async def test_upsert_fails_when_sparse_values_indices_values_mismatch_objects(self, host):
        asyncio_idx = build_asyncio_idx(host)
        with pytest.raises(PineconeException):
            await asyncio_idx.upsert(
                vectors=[
                    Vector(
                        id="1",
                        values=[0.1, 0.1],
                        sparse_values=SparseValues(indices=[0], values=[0.5, 0.5]),
                    )
                ]
            )
        with pytest.raises(PineconeException):
            await asyncio_idx.upsert(
                vectors=[
                    Vector(
                        id="1",
                        values=[0.1, 0.1],
                        sparse_values=SparseValues(indices=[0, 1], values=[0.5]),
                    )
                ]
            )

    async def test_upsert_fails_when_sparse_values_in_tuples(self, host):
        asyncio_idx = build_asyncio_idx(host)
        with pytest.raises(ValueError):
            await asyncio_idx.upsert(
                vectors=[
                    ("1", SparseValues(indices=[0], values=[0.5])),
                    ("2", SparseValues(indices=[0, 1, 2], values=[0.5, 0.5, 0.5])),
                ]
            )

    async def test_upsert_fails_when_sparse_values_indices_values_mismatch_dicts(self, host):
        asyncio_idx = build_asyncio_idx(host)
        with pytest.raises(PineconeException):
            await asyncio_idx.upsert(
                vectors=[
                    {
                        "id": "1",
                        "values": [0.2, 0.2],
                        "sparse_values": SparseValues(indices=[0], values=[0.5, 0.5]),
                    }
                ]
            )
        with pytest.raises(PineconeException):
            await asyncio_idx.upsert(
                vectors=[
                    {
                        "id": "1",
                        "values": [0.1, 0.2],
                        "sparse_values": SparseValues(indices=[0, 1], values=[0.5]),
                    }
                ]
            )


class TestUpsertFailsWhenValuesMissing:
    async def test_upsert_fails_when_values_missing_objects(self, host):
        asyncio_idx = build_asyncio_idx(host)
        with pytest.raises(PineconeApiValueError):
            await asyncio_idx.upsert(vectors=[Vector(id="1"), Vector(id="2")])

    async def test_upsert_fails_when_values_missing_tuples(self, host):
        asyncio_idx = build_asyncio_idx(host)
        with pytest.raises(ValueError):
            await asyncio_idx.upsert(vectors=[("1",), ("2",)])

    async def test_upsert_fails_when_values_missing_dicts(self, host):
        asyncio_idx = build_asyncio_idx(host)
        with pytest.raises(ValueError):
            await asyncio_idx.upsert(vectors=[{"id": "1"}, {"id": "2"}])


class TestUpsertFailsWhenValuesWrongType:
    async def test_upsert_fails_when_values_wrong_type_objects(self, host):
        asyncio_idx = build_asyncio_idx(host)
        with pytest.raises(TypeError):
            await asyncio_idx.upsert(
                vectors=[Vector(id="1", values="abc"), Vector(id="2", values="def")]
            )

    async def test_upsert_fails_when_values_wrong_type_tuples(self, host):
        asyncio_idx = build_asyncio_idx(host)
        if os.environ.get("USE_GRPC", "false") == "true":
            expected_exception = TypeError
        else:
            expected_exception = PineconeException

        with pytest.raises(expected_exception):
            await asyncio_idx.upsert(vectors=[("1", "abc"), ("2", "def")])

    async def test_upsert_fails_when_values_wrong_type_dicts(self, host):
        asyncio_idx = build_asyncio_idx(host)
        with pytest.raises(TypeError):
            await asyncio_idx.upsert(
                vectors=[{"id": "1", "values": "abc"}, {"id": "2", "values": "def"}]
            )


class TestUpsertFailsWhenVectorsMissing:
    async def test_upsert_fails_when_vectors_empty(self, host):
        asyncio_idx = build_asyncio_idx(host)
        with pytest.raises(PineconeException):
            await asyncio_idx.upsert(vectors=[])

    async def test_upsert_fails_when_vectors_wrong_type(self, host):
        asyncio_idx = build_asyncio_idx(host)
        with pytest.raises(ValueError):
            await asyncio_idx.upsert(vectors="abc")

    async def test_upsert_fails_when_vectors_missing(self, host):
        asyncio_idx = build_asyncio_idx(host)
        with pytest.raises(TypeError):
            await asyncio_idx.upsert()


# class TestUpsertIdMissing:
#     async def test_upsert_fails_when_id_is_missing_objects(self, host):
#         with pytest.raises(TypeError):
#             idx.upsert(
#                 vectors=[
#                     Vector(id="1", values=embedding_values()),
#                     Vector(values=embedding_values()),
#                 ]
#             )

#     async def test_upsert_fails_when_id_is_missing_tuples(self, host):
#         with pytest.raises(ValueError):
#             idx.upsert(vectors=[("1", embedding_values()), (embedding_values())])

#     async def test_upsert_fails_when_id_is_missing_dicts(self, host):
#         with pytest.raises(ValueError):
#             idx.upsert(
#                 vectors=[{"id": "1", "values": embedding_values()}, {"values": embedding_values()}]
#             )


# class TestUpsertIdWrongType:
#     async def test_upsert_fails_when_id_wrong_type_objects(self, host):
#         with pytest.raises(Exception):
#             idx.upsert(
#                 vectors=[
#                     Vector(id="1", values=embedding_values()),
#                     Vector(id=2, values=embedding_values()),
#                 ]
#             )

#     async def test_upsert_fails_when_id_wrong_type_tuples(self, host):
#         with pytest.raises(Exception):
#             idx.upsert(vectors=[("1", embedding_values()), (2, embedding_values())])

#     async def test_upsert_fails_when_id_wrong_type_dicts(self, host):
#         with pytest.raises(Exception):
#             idx.upsert(
#                 vectors=[
#                     {"id": "1", "values": embedding_values()},
#                     {"id": 2, "values": embedding_values()},
#                 ]
#             )
