from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from pinecone import Config
from pinecone.grpc import GRPCIndex
from pinecone.core.grpc.protos.vector_service_pb2 import (
    Vector,
    UpsertRequest,
    UpsertResponse,
    SparseValues,
)
from pinecone.grpc.utils import dict_to_proto_struct


class MockUpsertDelegate:
    def __init__(self, upsert_response: UpsertResponse):
        self.response = upsert_response

    def result(self, timeout):
        return self.response


@pytest.fixture
def expected_vec1(vals1):
    return Vector(id="vec1", values=vals1, metadata={})


@pytest.fixture
def expected_vec2(vals2):
    return Vector(id="vec2", values=vals2, metadata={})


@pytest.fixture
def expected_vec_md1(vals1, md1):
    return Vector(id="vec1", values=vals1, metadata=dict_to_proto_struct(md1))


@pytest.fixture
def expected_vec_md2(vals2, md2):
    return Vector(id="vec2", values=vals2, metadata=dict_to_proto_struct(md2))


@pytest.fixture
def expected_vec_md_sparse1(vals1, md1, sparse_indices_1, sparse_values_1):
    return Vector(
        id="vec1",
        values=vals1,
        metadata=dict_to_proto_struct(md1),
        sparse_values=SparseValues(indices=sparse_indices_1, values=sparse_values_1),
    )


@pytest.fixture
def expected_vec_md_sparse2(vals2, md2, sparse_indices_2, sparse_values_2):
    return Vector(
        id="vec2",
        values=vals2,
        metadata=dict_to_proto_struct(md2),
        sparse_values=SparseValues(indices=sparse_indices_2, values=sparse_values_2),
    )


class TestGrpcIndexUpsert:
    def setup_method(self):
        self.config = Config(api_key="test-api-key", host="foo")
        self.index = GRPCIndex(config=self.config, index_name="example-name", _endpoint_override="test-endpoint")

    def _assert_called_once(self, vectors, async_call=False):
        self.index._wrap_grpc_call.assert_called_once_with(
            self.index.stub.Upsert.future if async_call else self.index.stub.Upsert,
            UpsertRequest(vectors=vectors, namespace="ns"),
            timeout=None,
        )

    def test_upsert_tuplesOfIdVec_UpserWithoutMD(self, mocker, vals1, vals2, expected_vec1, expected_vec2):
        mocker.patch.object(self.index, "_wrap_grpc_call", autospec=True)
        self.index.upsert([("vec1", vals1), ("vec2", vals2)], namespace="ns")
        self._assert_called_once([expected_vec1, expected_vec2])

    def test_upsert_tuplesOfIdVecMD_UpsertVectorsWithMD(
        self, mocker, vals1, md1, vals2, md2, expected_vec_md1, expected_vec_md2
    ):
        mocker.patch.object(self.index, "_wrap_grpc_call", autospec=True)
        self.index.upsert([("vec1", vals1, md1), ("vec2", vals2, md2)], namespace="ns")
        self._assert_called_once(
            [expected_vec_md1, expected_vec_md2],
        )

    def test_upsert_vectors_upsertInputVectors(self, mocker, expected_vec_md1, expected_vec_md2):
        mocker.patch.object(self.index, "_wrap_grpc_call", autospec=True)
        self.index.upsert([expected_vec_md1, expected_vec_md2], namespace="ns")
        self._assert_called_once(
            [expected_vec_md1, expected_vec_md2],
        )

    def test_upsert_vectors_upsertInputVectorsSparse(
        self,
        mocker,
        vals1,
        md1,
        vals2,
        md2,
        sparse_indices_1,
        sparse_values_1,
        sparse_indices_2,
        sparse_values_2,
        expected_vec_md_sparse1,
        expected_vec_md_sparse2,
    ):
        mocker.patch.object(self.index, "_wrap_grpc_call", autospec=True)
        self.index.upsert(
            [
                Vector(
                    id="vec1",
                    values=vals1,
                    metadata=dict_to_proto_struct(md1),
                    sparse_values=SparseValues(indices=sparse_indices_1, values=sparse_values_1),
                ),
                Vector(
                    id="vec2",
                    values=vals2,
                    metadata=dict_to_proto_struct(md2),
                    sparse_values=SparseValues(indices=sparse_indices_2, values=sparse_values_2),
                ),
            ],
            namespace="ns",
        )
        self._assert_called_once([expected_vec_md_sparse1, expected_vec_md_sparse2])

    def test_upsert_dict(self, mocker, vals1, vals2, expected_vec1, expected_vec2):
        mocker.patch.object(self.index, "_wrap_grpc_call", autospec=True)
        dict1 = {"id": "vec1", "values": vals1}
        dict2 = {"id": "vec2", "values": vals2}
        self.index.upsert([dict1, dict2], namespace="ns")
        self._assert_called_once([expected_vec1, expected_vec2])

    def test_upsert_dict_md(self, mocker, vals1, md1, vals2, md2, expected_vec_md1, expected_vec_md2):
        mocker.patch.object(self.index, "_wrap_grpc_call", autospec=True)
        dict1 = {"id": "vec1", "values": vals1, "metadata": md1}
        dict2 = {"id": "vec2", "values": vals2, "metadata": md2}
        self.index.upsert([dict1, dict2], namespace="ns")
        self._assert_called_once([expected_vec_md1, expected_vec_md2])

    def test_upsert_dict_sparse(
        self, mocker, vals1, vals2, sparse_indices_1, sparse_values_1, sparse_indices_2, sparse_values_2
    ):
        mocker.patch.object(self.index, "_wrap_grpc_call", autospec=True)
        dict1 = {
            "id": "vec1",
            "values": vals1,
            "sparse_values": {"indices": sparse_indices_1, "values": sparse_values_1},
        }
        dict2 = {
            "id": "vec2",
            "values": vals2,
            "sparse_values": {"indices": sparse_indices_2, "values": sparse_values_2},
        }
        self.index.upsert([dict1, dict2], namespace="ns")
        self._assert_called_once(
            [
                Vector(
                    id="vec1",
                    values=vals1,
                    metadata={},
                    sparse_values=SparseValues(indices=sparse_indices_1, values=sparse_values_1),
                ),
                Vector(
                    id="vec2",
                    values=vals2,
                    metadata={},
                    sparse_values=SparseValues(indices=sparse_indices_2, values=sparse_values_2),
                ),
            ]
        )

    def test_upsert_dict_sparse_md(
        self, mocker, vals1, md1, vals2, md2, sparse_indices_1, sparse_values_1, sparse_indices_2, sparse_values_2
    ):
        mocker.patch.object(self.index, "_wrap_grpc_call", autospec=True)
        dict1 = {
            "id": "vec1",
            "values": vals1,
            "sparse_values": {"indices": sparse_indices_1, "values": sparse_values_1},
            "metadata": md1,
        }
        dict2 = {
            "id": "vec2",
            "values": vals2,
            "sparse_values": {"indices": sparse_indices_2, "values": sparse_values_2},
            "metadata": md2,
        }
        self.index.upsert([dict1, dict2], namespace="ns")
        self._assert_called_once(
            [
                Vector(
                    id="vec1",
                    values=vals1,
                    metadata=dict_to_proto_struct(md1),
                    sparse_values=SparseValues(indices=sparse_indices_1, values=sparse_values_1),
                ),
                Vector(
                    id="vec2",
                    values=vals2,
                    metadata=dict_to_proto_struct(md2),
                    sparse_values=SparseValues(indices=sparse_indices_2, values=sparse_values_2),
                ),
            ]
        )

    def test_upsert_dict_negative(self, mocker, vals1, vals2, md2):
        mocker.patch.object(self.index, "_wrap_grpc_call", autospec=True)

        # Missing required keys
        dict1 = {"values": vals1}
        dict2 = {"id": "vec2"}
        with pytest.raises(ValueError):
            self.index.upsert([dict1, dict2])
        with pytest.raises(ValueError):
            self.index.upsert([dict1])
        with pytest.raises(ValueError):
            self.index.upsert([dict2])

        # Excess keys
        dict1 = {"id": "vec1", "values": vals1}
        dict2 = {"id": "vec2", "values": vals2, "animal": "dog"}
        with pytest.raises(ValueError) as e:
            self.index.upsert([dict1, dict2])
            assert "animal" in str(e.value)

        dict1 = {"id": "vec1", "values": vals1, "metadatta": md2}
        dict2 = {"id": "vec2", "values": vals2}
        with pytest.raises(ValueError) as e:
            self.index.upsert([dict1, dict2])
            assert "metadatta" in str(e.value)

    @pytest.mark.parametrize(
        "key,new_val",
        [
            ("values", ["the", "lazy", "fox"]),
            ("values", "the lazy fox"),
            ("values", 0.5),
            ("metadata", np.nan),
            ("metadata", ["key1", "key2"]),
            ("sparse_values", "cat"),
            ("sparse_values", []),
        ],
    )
    def test_upsert_dict_with_invalid_values(self, mocker, key, new_val, vals1, md1, sparse_indices_1, sparse_values_1):
        mocker.patch.object(self.index, "_wrap_grpc_call", autospec=True)

        full_dict1 = {
            "id": "vec1",
            "values": vals1,
            "sparse_values": {"indices": sparse_indices_1, "values": sparse_values_1},
            "metadata": md1,
        }

        dict1 = deepcopy(full_dict1)
        dict1[key] = new_val
        with pytest.raises(TypeError) as e:
            self.index.upsert([dict1])
        assert key in str(e.value)

    @pytest.mark.parametrize(
        "key,new_val",
        [
            ("id", 4.2),
            ("id", ["vec1"]),
        ],
    )
    def test_upsert_dict_with_invalid_ids(self, mocker, key, new_val, vals1, md1, sparse_indices_1, sparse_values_1):
        mocker.patch.object(self.index, "_wrap_grpc_call", autospec=True)

        full_dict1 = {
            "id": "vec1",
            "values": vals1,
            "sparse_values": {"indices": sparse_indices_1, "values": sparse_values_1},
            "metadata": md1,
        }

        dict1 = deepcopy(full_dict1)
        dict1[key] = new_val
        with pytest.raises(TypeError) as e:
            self.index.upsert([dict1])
        assert str(new_val) in str(e.value)

    @pytest.mark.parametrize(
        "key,new_val",
        [
            ("indices", 3),
            ("indices", [1.2, 0.5]),
            ("values", ["1", "4.4"]),
            ("values", 0.5),
        ],
    )
    def test_upsert_dict_with_invalid_sparse_values(
        self, mocker, key, new_val, vals1, md1, sparse_indices_1, sparse_values_1
    ):
        mocker.patch.object(self.index, "_wrap_grpc_call", autospec=True)

        full_dict1 = {
            "id": "vec1",
            "values": vals1,
            "sparse_values": {"indices": sparse_indices_1, "values": sparse_values_1},
            "metadata": md1,
        }

        dict1 = deepcopy(full_dict1)
        dict1["sparse_values"][key] = new_val
        with pytest.raises(TypeError) as e:
            self.index.upsert([dict1])
        assert "sparse" in str(e.value)
        assert key in str(e.value)

    def test_upsert_dataframe(
        self,
        mocker,
        vals1,
        sparse_indices_1,
        sparse_values_1,
        md1,
        vals2,
        sparse_indices_2,
        sparse_values_2,
        md2,
        expected_vec_md_sparse1,
        expected_vec_md_sparse2,
    ):
        mocker.patch.object(
            self.index,
            "_wrap_grpc_call",
            autospec=True,
            side_effect=lambda stub, upsert_request, timeout: MockUpsertDelegate(
                UpsertResponse(upserted_count=len(upsert_request.vectors))
            ),
        )
        df = pd.DataFrame(
            [
                {
                    "id": "vec1",
                    "values": vals1,
                    "sparse_values": {"indices": sparse_indices_1, "values": sparse_values_1},
                    "metadata": md1,
                },
                {
                    "id": "vec2",
                    "values": vals2,
                    "sparse_values": {"indices": sparse_indices_2, "values": sparse_values_2},
                    "metadata": md2,
                },
            ]
        )
        self.index.upsert_from_dataframe(df, namespace="ns")
        self._assert_called_once([expected_vec_md_sparse1, expected_vec_md_sparse2], async_call=True)

    def test_upsert_dataframe_sync(
        self,
        mocker,
        vals1,
        md1,
        vals2,
        md2,
        sparse_indices_1,
        sparse_values_1,
        sparse_indices_2,
        sparse_values_2,
        expected_vec_md_sparse1,
        expected_vec_md_sparse2,
    ):
        mocker.patch.object(
            self.index,
            "_wrap_grpc_call",
            autospec=True,
            side_effect=lambda stub, upsert_request, timeout: UpsertResponse(
                upserted_count=len(upsert_request.vectors)
            ),
        )
        df = pd.DataFrame(
            [
                {
                    "id": "vec1",
                    "values": vals1,
                    "sparse_values": {"indices": sparse_indices_1, "values": sparse_values_1},
                    "metadata": md1,
                },
                {
                    "id": "vec2",
                    "values": vals2,
                    "sparse_values": {"indices": sparse_indices_2, "values": sparse_values_2},
                    "metadata": md2,
                },
            ]
        )
        self.index.upsert_from_dataframe(df, namespace="ns", use_async_requests=False)
        self._assert_called_once([expected_vec_md_sparse1, expected_vec_md_sparse2], async_call=False)

    def test_upsert_dataframe_negative(
        self, mocker, vals1, md1, vals2, md2, sparse_indices_1, sparse_values_1, sparse_indices_2, sparse_values_2
    ):
        mocker.patch.object(self.index, "_wrap_grpc_call", autospec=True)
        full_dict1 = {
            "id": "vec1",
            "values": vals1,
            "sparse_values": {"indices": sparse_indices_1, "values": sparse_values_1},
            "metadata": md1,
        }
        full_df = pd.DataFrame([full_dict1])

        # Not a DF
        with pytest.raises(ValueError):
            self.index.upsert_from_dataframe([full_dict1])
        with pytest.raises(ValueError):
            self.index.upsert_from_dataframe(full_dict1)

        # Missing Cols
        df = full_df.copy()
        df.drop(columns=["id"], inplace=True)
        with pytest.raises(ValueError):
            self.index.upsert_from_dataframe(df)

        # Excess cols
        df = full_df.copy()
        df["animals"] = ["dog"]
        with pytest.raises(ValueError):
            self.index.upsert_from_dataframe(df)

        df = full_df.copy()
        df["metadat"] = df["metadata"]
        with pytest.raises(ValueError):
            self.index.upsert_from_dataframe(df)

    def test_upsert_async_upsertInputVectorsAsync(self, mocker, expected_vec_md1, expected_vec_md2):
        mocker.patch.object(self.index, "_wrap_grpc_call", autospec=True)
        self.index.upsert([expected_vec_md1, expected_vec_md2], namespace="ns", async_req=True)
        self._assert_called_once([expected_vec_md1, expected_vec_md2], async_call=True)

    def test_upsert_vectorListIsMultiplyOfBatchSize_vectorsUpsertedInBatches(
        self, mocker, vals1, md1, expected_vec_md1, expected_vec_md2
    ):
        mocker.patch.object(
            self.index,
            "_wrap_grpc_call",
            autospec=True,
            side_effect=lambda stub, upsert_request, timeout: UpsertResponse(
                upserted_count=len(upsert_request.vectors)
            ),
        )

        result = self.index.upsert(
            [expected_vec_md1, expected_vec_md2], namespace="ns", batch_size=1, show_progress=False
        )
        self.index._wrap_grpc_call.assert_any_call(
            self.index.stub.Upsert,
            UpsertRequest(
                vectors=[Vector(id="vec1", values=vals1, metadata=dict_to_proto_struct(md1))], namespace="ns"
            ),
            timeout=None,
        )

        self.index._wrap_grpc_call.assert_any_call(
            self.index.stub.Upsert, UpsertRequest(vectors=[expected_vec_md2], namespace="ns"), timeout=None
        )

        assert result.upserted_count == 2

    def test_upsert_vectorListNotMultiplyOfBatchSize_vectorsUpsertedInBatches(
        self, mocker, vals1, vals2, md1, md2, expected_vec_md1, expected_vec_md2
    ):
        mocker.patch.object(
            self.index,
            "_wrap_grpc_call",
            autospec=True,
            side_effect=lambda stub, upsert_request, timeout: UpsertResponse(
                upserted_count=len(upsert_request.vectors)
            ),
        )

        result = self.index.upsert(
            [
                expected_vec_md1,
                Vector(id="vec2", values=vals2, metadata=dict_to_proto_struct(md2)),
                Vector(id="vec3", values=vals1, metadata=dict_to_proto_struct(md1)),
            ],
            namespace="ns",
            batch_size=2,
        )
        self.index._wrap_grpc_call.assert_any_call(
            self.index.stub.Upsert,
            UpsertRequest(vectors=[expected_vec_md1, expected_vec_md2], namespace="ns"),
            timeout=None,
        )

        self.index._wrap_grpc_call.assert_any_call(
            self.index.stub.Upsert,
            UpsertRequest(
                vectors=[Vector(id="vec3", values=vals1, metadata=dict_to_proto_struct(md1))], namespace="ns"
            ),
            timeout=None,
        )

        assert result.upserted_count == 3

    def test_upsert_vectorListSmallerThanBatchSize_vectorsUpsertedInBatches(
        self, mocker, expected_vec_md1, expected_vec_md2
    ):
        mocker.patch.object(
            self.index,
            "_wrap_grpc_call",
            autospec=True,
            side_effect=lambda stub, upsert_request, timeout: UpsertResponse(
                upserted_count=len(upsert_request.vectors)
            ),
        )

        result = self.index.upsert([expected_vec_md1, expected_vec_md2], namespace="ns", batch_size=5)
        self._assert_called_once(
            [expected_vec_md1, expected_vec_md2],
        )

        assert result.upserted_count == 2

    def test_upsert_tuplesList_vectorsUpsertedInBatches(
        self, mocker, vals1, md1, vals2, md2, expected_vec_md1, expected_vec_md2
    ):
        mocker.patch.object(
            self.index,
            "_wrap_grpc_call",
            autospec=True,
            side_effect=lambda stub, upsert_request, timeout: UpsertResponse(
                upserted_count=len(upsert_request.vectors)
            ),
        )

        result = self.index.upsert(
            [("vec1", vals1, md1), ("vec2", vals2, md2), ("vec3", vals1, md1)],
            namespace="ns",
            batch_size=2,
        )
        self.index._wrap_grpc_call.assert_any_call(
            self.index.stub.Upsert,
            UpsertRequest(vectors=[expected_vec_md1, expected_vec_md2], namespace="ns"),
            timeout=None,
        )

        self.index._wrap_grpc_call.assert_any_call(
            self.index.stub.Upsert,
            UpsertRequest(
                vectors=[Vector(id="vec3", values=vals1, metadata=dict_to_proto_struct(md1))], namespace="ns"
            ),
            timeout=None,
        )

        assert result.upserted_count == 3

    def test_upsert_batchSizeIsNotPositive_errorIsRaised(self, vals1, md1):
        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            self.index.upsert(
                [Vector(id="vec1", values=vals1, metadata=dict_to_proto_struct(md1))],
                namespace="ns",
                batch_size=0,
            )

        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            self.index.upsert(
                [Vector(id="vec1", values=vals1, metadata=dict_to_proto_struct(md1))],
                namespace="ns",
                batch_size=-1,
            )

    def test_upsert_useBatchSizeAndAsyncReq_valueErrorRaised(self, vals1, md1):
        with pytest.raises(ValueError, match="async_req is not supported when batch_size is provided."):
            self.index.upsert(
                [Vector(id="vec1", values=vals1, metadata=dict_to_proto_struct(md1))],
                namespace="ns",
                batch_size=2,
                async_req=True,
            )
