import pandas as pd
import pytest

from pinecone.db_data import _Index, _IndexAsyncio
import pinecone.core.openapi.db_data.models as oai
from pinecone import QueryResponse, UpsertResponse, Vector


class TestRestIndex:
    def setup_method(self):
        self.vector_dim = 8
        self.id1 = "vec1"
        self.id2 = "vec2"
        self.vals1 = [0.1] * self.vector_dim
        self.vals2 = [0.2] * self.vector_dim
        self.md1 = {"genre": "action", "year": 2021}
        self.md2 = {"genre": "documentary", "year": 2020}
        self.filter1 = {"genre": {"$in": ["action"]}}
        self.filter2 = {"year": {"$eq": 2020}}
        self.svi1 = [1, 3, 5]
        self.svv1 = [0.1, 0.2, 0.3]
        self.sv1 = {"indices": self.svi1, "values": self.svv1}
        self.svi2 = [2, 4, 6]
        self.svv2 = [0.1, 0.2, 0.3]
        self.sv2 = {"indices": self.svi2, "values": self.svv2}

        self.index = _Index(api_key="asdf", host="https://test.pinecone.io")

    # region: upsert tests

    def test_upsert_tuplesOfIdVec_UpserWithoutMD(self, mocker):
        mocker.patch.object(self.index._vector_api, "upsert_vectors", autospec=True)
        self.index.upsert([("vec1", self.vals1), ("vec2", self.vals2)], namespace="ns")
        self.index._vector_api.upsert_vectors.assert_called_once_with(
            oai.UpsertRequest(
                vectors=[
                    oai.Vector(id="vec1", values=self.vals1, metadata={}),
                    oai.Vector(id="vec2", values=self.vals2, metadata={}),
                ],
                namespace="ns",
            )
        )

    def test_upsert_tuplesOfIdVecMD_UpsertVectorsWithMD(self, mocker):
        mocker.patch.object(self.index._vector_api, "upsert_vectors", autospec=True)
        self.index.upsert([("vec1", self.vals1, self.md1), ("vec2", self.vals2, self.md2)])
        self.index._vector_api.upsert_vectors.assert_called_once_with(
            oai.UpsertRequest(
                vectors=[
                    oai.Vector(id="vec1", values=self.vals1, metadata=self.md1),
                    oai.Vector(id="vec2", values=self.vals2, metadata=self.md2),
                ]
            )
        )

    def test_upsert_dictOfIdVecMD_UpsertVectorsWithMD(self, mocker):
        mocker.patch.object(self.index._vector_api, "upsert_vectors", autospec=True)
        self.index.upsert(
            [
                {"id": self.id1, "values": self.vals1, "metadata": self.md1},
                {"id": self.id2, "values": self.vals2, "metadata": self.md2},
            ]
        )
        self.index._vector_api.upsert_vectors.assert_called_once_with(
            oai.UpsertRequest(
                vectors=[
                    oai.Vector(id="vec1", values=self.vals1, metadata=self.md1),
                    oai.Vector(id="vec2", values=self.vals2, metadata=self.md2),
                ]
            )
        )

    def test_upsert_dictOfIdVecMD_UpsertVectorsWithoutMD(self, mocker):
        mocker.patch.object(self.index._vector_api, "upsert_vectors", autospec=True)
        self.index.upsert(
            [{"id": self.id1, "values": self.vals1}, {"id": self.id2, "values": self.vals2}]
        )
        self.index._vector_api.upsert_vectors.assert_called_once_with(
            oai.UpsertRequest(
                vectors=[
                    oai.Vector(id="vec1", values=self.vals1),
                    oai.Vector(id="vec2", values=self.vals2),
                ]
            )
        )

    def test_upsert_dictOfIdVecMD_UpsertVectorsWithSparseValues(self, mocker):
        mocker.patch.object(self.index._vector_api, "upsert_vectors", autospec=True)
        self.index.upsert(
            [
                {"id": self.id1, "values": self.vals1, "sparse_values": self.sv1},
                {"id": self.id2, "values": self.vals2, "sparse_values": self.sv2},
            ]
        )
        self.index._vector_api.upsert_vectors.assert_called_once_with(
            oai.UpsertRequest(
                vectors=[
                    oai.Vector(
                        id="vec1", values=self.vals1, sparse_values=oai.SparseValues(**self.sv1)
                    ),
                    oai.Vector(
                        id="vec2", values=self.vals2, sparse_values=oai.SparseValues(**self.sv2)
                    ),
                ]
            )
        )

    def test_upsert_vectors_upsertInputVectors(self, mocker):
        mocker.patch.object(self.index._vector_api, "upsert_vectors", autospec=True)
        self.index.upsert(
            vectors=[
                oai.Vector(id="vec1", values=self.vals1, metadata=self.md1),
                oai.Vector(id="vec2", values=self.vals2, metadata=self.md2),
            ],
            namespace="ns",
        )
        self.index._vector_api.upsert_vectors.assert_called_once_with(
            oai.UpsertRequest(
                vectors=[
                    oai.Vector(id="vec1", values=self.vals1, metadata=self.md1),
                    oai.Vector(id="vec2", values=self.vals2, metadata=self.md2),
                ],
                namespace="ns",
            )
        )

    def test_upsert_parallelUpsert_callUpsertParallel(self, mocker):
        chunks = [
            [Vector(id="vec1", values=self.vals1, metadata=self.md1)],
            [Vector(id="vec2", values=self.vals2, metadata=self.md2)],
        ]
        with _Index(api_key="asdf", host="https://test.pinecone.io", pool_threads=30) as index:
            mocker.patch.object(index._vector_api, "upsert_vectors", autospec=True)

            # Send requests in parallel
            async_results = [
                index.upsert(vectors=ids_vectors_chunk, namespace="ns", async_req=True)
                for ids_vectors_chunk in chunks
            ]
            # Wait for and retrieve responses (this raises in case of error)
            [async_result.get() for async_result in async_results]

            index._vector_api.upsert_vectors.assert_any_call(
                oai.UpsertRequest(
                    vectors=[oai.Vector(id="vec1", values=self.vals1, metadata=self.md1)],
                    namespace="ns",
                ),
                async_req=True,
            )

            index._vector_api.upsert_vectors.assert_any_call(
                oai.UpsertRequest(
                    vectors=[oai.Vector(id="vec2", values=self.vals2, metadata=self.md2)],
                    namespace="ns",
                ),
                async_req=True,
            )

    def test_upsert_vectorListIsMultiplyOfBatchSize_vectorsUpsertedInBatches(self, mocker):
        mocker.patch.object(
            self.index._vector_api,
            "upsert_vectors",
            autospec=True,
            side_effect=lambda upsert_request: UpsertResponse(
                upserted_count=len(upsert_request.vectors)
            ),
        )

        result = self.index.upsert(
            vectors=[
                Vector(id="vec1", values=self.vals1, metadata=self.md1),
                Vector(id="vec2", values=self.vals2, metadata=self.md2),
            ],
            namespace="ns",
            batch_size=1,
            show_progress=False,
        )

        self.index._vector_api.upsert_vectors.assert_any_call(
            oai.UpsertRequest(
                vectors=[oai.Vector(id="vec1", values=self.vals1, metadata=self.md1)],
                namespace="ns",
            )
        )

        self.index._vector_api.upsert_vectors.assert_any_call(
            oai.UpsertRequest(
                vectors=[oai.Vector(id="vec2", values=self.vals2, metadata=self.md2)],
                namespace="ns",
            )
        )

        assert result.upserted_count == 2

    def test_upsert_vectorListNotMultiplyOfBatchSize_vectorsUpsertedInBatches(self, mocker):
        mocker.patch.object(
            self.index._vector_api,
            "upsert_vectors",
            autospec=True,
            side_effect=lambda upsert_request: UpsertResponse(
                upserted_count=len(upsert_request.vectors)
            ),
        )

        result = self.index.upsert(
            vectors=[
                oai.Vector(id="vec1", values=self.vals1, metadata=self.md1),
                oai.Vector(id="vec2", values=self.vals2, metadata=self.md2),
                oai.Vector(id="vec3", values=self.vals1, metadata=self.md1),
            ],
            namespace="ns",
            batch_size=2,
        )

        self.index._vector_api.upsert_vectors.assert_any_call(
            oai.UpsertRequest(
                vectors=[
                    oai.Vector(id="vec1", values=self.vals1, metadata=self.md1),
                    oai.Vector(id="vec2", values=self.vals2, metadata=self.md2),
                ],
                namespace="ns",
            )
        )

        self.index._vector_api.upsert_vectors.assert_any_call(
            oai.UpsertRequest(
                vectors=[oai.Vector(id="vec3", values=self.vals1, metadata=self.md1)],
                namespace="ns",
            )
        )

        assert result.upserted_count == 3

    def test_upsert_vectorListSmallerThanBatchSize_vectorsUpsertedInBatches(self, mocker):
        mocker.patch.object(
            self.index._vector_api,
            "upsert_vectors",
            autospec=True,
            side_effect=lambda upsert_request: UpsertResponse(
                upserted_count=len(upsert_request.vectors)
            ),
        )

        result = self.index.upsert(
            vectors=[
                Vector(id="vec1", values=self.vals1, metadata=self.md1),
                Vector(id="vec2", values=self.vals2, metadata=self.md2),
                Vector(id="vec3", values=self.vals1, metadata=self.md1),
            ],
            namespace="ns",
            batch_size=5,
        )

        self.index._vector_api.upsert_vectors.assert_called_once_with(
            oai.UpsertRequest(
                vectors=[
                    oai.Vector(id="vec1", values=self.vals1, metadata=self.md1),
                    oai.Vector(id="vec2", values=self.vals2, metadata=self.md2),
                    oai.Vector(id="vec3", values=self.vals1, metadata=self.md1),
                ],
                namespace="ns",
            )
        )

        assert result.upserted_count == 3

    def test_upsert_tuplesList_vectorsUpsertedInBatches(self, mocker):
        mocker.patch.object(
            self.index._vector_api,
            "upsert_vectors",
            autospec=True,
            side_effect=lambda upsert_request: UpsertResponse(
                upserted_count=len(upsert_request.vectors)
            ),
        )

        result = self.index.upsert(
            vectors=[
                ("vec1", self.vals1, self.md1),
                ("vec2", self.vals2, self.md2),
                ("vec3", self.vals1, self.md1),
            ],
            namespace="ns",
            batch_size=2,
        )

        self.index._vector_api.upsert_vectors.assert_any_call(
            oai.UpsertRequest(
                vectors=[
                    oai.Vector(id="vec1", values=self.vals1, metadata=self.md1),
                    oai.Vector(id="vec2", values=self.vals2, metadata=self.md2),
                ],
                namespace="ns",
            )
        )

        self.index._vector_api.upsert_vectors.assert_any_call(
            oai.UpsertRequest(
                vectors=[oai.Vector(id="vec3", values=self.vals1, metadata=self.md1)],
                namespace="ns",
            )
        )

        assert result.upserted_count == 3

    def test_upsert_dataframe(self, mocker):
        mocker.patch.object(
            self.index._vector_api,
            "upsert_vectors",
            autospec=True,
            return_value=UpsertResponse(upserted_count=2),
        )
        df = pd.DataFrame(
            [
                {"id": self.id1, "values": self.vals1, "metadata": self.md1},
                {"id": self.id2, "values": self.vals2, "metadata": self.md2},
            ]
        )
        self.index.upsert_from_dataframe(df)

        self.index._vector_api.upsert_vectors.assert_called_once_with(
            oai.UpsertRequest(
                vectors=[
                    oai.Vector(id="vec1", values=self.vals1, metadata=self.md1),
                    oai.Vector(id="vec2", values=self.vals2, metadata=self.md2),
                ]
            )
        )

    def test_upsert_batchSizeIsNotPositive_errorIsRaised(self):
        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            self.index.upsert(
                vectors=[Vector(id="vec1", values=self.vals1, metadata=self.md1)],
                namespace="ns",
                batch_size=0,
            )

        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            self.index.upsert(
                vectors=[oai.Vector(id="vec1", values=self.vals1, metadata=self.md1)],
                namespace="ns",
                batch_size=-1,
            )

    def test_upsert_useBatchSizeAndAsyncReq_valueErrorRaised(self):
        with pytest.raises(
            ValueError, match="async_req is not supported when batch_size is provided."
        ):
            self.index.upsert(
                vectors=[Vector(id="vec1", values=self.vals1, metadata=self.md1)],
                namespace="ns",
                batch_size=1,
                async_req=True,
            )

    # endregion

    # region: query tests

    def test_query_byVectorNoFilter_queryVectorNoFilter(self, mocker):
        # Mock should return OpenAPI QueryResponse, not dataclass
        from pinecone.core.openapi.db_data.models import QueryResponse as OpenAPIQueryResponse

        response = OpenAPIQueryResponse(
            matches=[oai.ScoredVector(id="1", score=0.9, values=[0.0], metadata={"a": 2})],
            namespace="test",
        )
        mocker.patch.object(
            self.index._vector_api, "query_vectors", autospec=True, return_value=response
        )

        actual = self.index.query(top_k=10, vector=self.vals1)

        self.index._vector_api.query_vectors.assert_called_once_with(
            oai.QueryRequest(top_k=10, vector=self.vals1)
        )
        expected = QueryResponse(
            matches=[oai.ScoredVector(id="1", score=0.9, values=[0.0], metadata={"a": 2})],
            namespace="test",
        )
        # Compare dataclasses by comparing fields directly
        assert expected.matches == actual.matches
        assert expected.namespace == actual.namespace
        assert expected.usage == actual.usage
        # _response_info may not be present in test mocks, so we don't assert it

    def test_query_byVectorWithFilter_queryVectorWithFilter(self, mocker):
        mocker.patch.object(self.index._vector_api, "query_vectors", autospec=True)
        self.index.query(top_k=10, vector=self.vals1, filter=self.filter1, namespace="ns")
        self.index._vector_api.query_vectors.assert_called_once_with(
            oai.QueryRequest(top_k=10, vector=self.vals1, filter=self.filter1, namespace="ns")
        )

    def test_query_byVecId_queryByVecId(self, mocker):
        mocker.patch.object(self.index._vector_api, "query_vectors", autospec=True)
        self.index.query(top_k=10, id="vec1", include_metadata=True, include_values=False)
        self.index._vector_api.query_vectors.assert_called_once_with(
            oai.QueryRequest(top_k=10, id="vec1", include_metadata=True, include_values=False)
        )

    def test_query_rejects_both_id_and_vector(self):
        with pytest.raises(ValueError, match="Cannot specify both `id` and `vector`"):
            self.index.query(top_k=10, id="vec1", vector=[1, 2, 3])

    def test_query_with_positional_args(self, mocker):
        with pytest.raises(ValueError) as e:
            self.index.query([0.1, 0.2, 0.3], top_k=10)
        assert (
            "The argument order for `query()` has changed; please use keyword arguments instead of positional arguments"
            in str(e.value)
        )

    # endregion

    # region: delete tests

    def test_delete_byIds_deleteByIds(self, mocker):
        mocker.patch.object(self.index._vector_api, "delete_vectors", autospec=True)
        self.index.delete(ids=["vec1", "vec2"])
        self.index._vector_api.delete_vectors.assert_called_once_with(
            oai.DeleteRequest(ids=["vec1", "vec2"])
        )

    def test_delete_deleteAllByFilter_deleteAllByFilter(self, mocker):
        mocker.patch.object(self.index._vector_api, "delete_vectors", autospec=True)
        self.index.delete(delete_all=True, filter=self.filter1, namespace="ns")
        self.index._vector_api.delete_vectors.assert_called_once_with(
            oai.DeleteRequest(delete_all=True, filter=self.filter1, namespace="ns")
        )

    def test_delete_deleteAllNoFilter_deleteNoFilter(self, mocker):
        mocker.patch.object(self.index._vector_api, "delete_vectors", autospec=True)
        self.index.delete(delete_all=True)
        self.index._vector_api.delete_vectors.assert_called_once_with(
            oai.DeleteRequest(delete_all=True)
        )

    # endregion

    # region: fetch tests

    def test_fetch_byIds_fetchByIds(self, mocker):
        mocker.patch.object(self.index._vector_api, "fetch_vectors", autospec=True)
        self.index.fetch(ids=["vec1", "vec2"])
        self.index._vector_api.fetch_vectors.assert_called_once_with(ids=["vec1", "vec2"])

    def test_fetch_byIdsAndNS_fetchByIdsAndNS(self, mocker):
        mocker.patch.object(self.index._vector_api, "fetch_vectors", autospec=True)
        self.index.fetch(ids=["vec1", "vec2"], namespace="ns")
        self.index._vector_api.fetch_vectors.assert_called_once_with(
            ids=["vec1", "vec2"], namespace="ns"
        )

    def test_fetch_by_metadata_with_filter(self, mocker):
        mocker.patch.object(self.index._vector_api, "fetch_vectors_by_metadata", autospec=True)
        filter_dict = {"genre": {"$eq": "action"}}
        self.index.fetch_by_metadata(filter=filter_dict)
        call_args = self.index._vector_api.fetch_vectors_by_metadata.call_args
        assert call_args is not None
        request = call_args[0][0]
        assert isinstance(request, oai.FetchByMetadataRequest)
        assert request.filter == filter_dict

    def test_fetch_by_metadata_with_filter_and_namespace(self, mocker):
        mocker.patch.object(self.index._vector_api, "fetch_vectors_by_metadata", autospec=True)
        filter_dict = {"genre": {"$in": ["comedy", "drama"]}}
        self.index.fetch_by_metadata(filter=filter_dict, namespace="ns")
        call_args = self.index._vector_api.fetch_vectors_by_metadata.call_args
        assert call_args is not None
        request = call_args[0][0]
        assert isinstance(request, oai.FetchByMetadataRequest)
        assert request.filter == filter_dict
        assert request.namespace == "ns"

    def test_fetch_by_metadata_with_limit(self, mocker):
        mocker.patch.object(self.index._vector_api, "fetch_vectors_by_metadata", autospec=True)
        filter_dict = {"year": {"$gte": 2020}}
        self.index.fetch_by_metadata(filter=filter_dict, limit=50)
        call_args = self.index._vector_api.fetch_vectors_by_metadata.call_args
        assert call_args is not None
        request = call_args[0][0]
        assert isinstance(request, oai.FetchByMetadataRequest)
        assert request.filter == filter_dict
        assert request.limit == 50

    def test_fetch_by_metadata_with_pagination_token(self, mocker):
        mocker.patch.object(self.index._vector_api, "fetch_vectors_by_metadata", autospec=True)
        filter_dict = {"status": "active"}
        pagination_token = "token123"
        self.index.fetch_by_metadata(filter=filter_dict, pagination_token=pagination_token)
        call_args = self.index._vector_api.fetch_vectors_by_metadata.call_args
        assert call_args is not None
        request = call_args[0][0]
        assert isinstance(request, oai.FetchByMetadataRequest)
        assert request.filter == filter_dict
        assert request.pagination_token == pagination_token

    # endregion

    # region: update tests

    def test_update_byIdAnValues_updateByIdAndValues(self, mocker):
        mocker.patch.object(self.index._vector_api, "update_vector", autospec=True)
        self.index.update(id="vec1", values=self.vals1, namespace="ns")
        self.index._vector_api.update_vector.assert_called_once_with(
            oai.UpdateRequest(id="vec1", values=self.vals1, namespace="ns")
        )

    def test_update_byIdAnValuesAndMetadata_updateByIdAndValuesAndMetadata(self, mocker):
        mocker.patch.object(self.index._vector_api, "update_vector", autospec=True)
        self.index.update("vec1", values=self.vals1, metadata=self.md1)
        self.index._vector_api.update_vector.assert_called_once_with(
            oai.UpdateRequest(id="vec1", values=self.vals1, metadata=self.md1)
        )

    def test_update_withFilter_updateWithFilter(self, mocker):
        mocker.patch.object(self.index._vector_api, "update_vector", autospec=True)
        self.index.update(filter=self.filter1, namespace="ns")
        self.index._vector_api.update_vector.assert_called_once_with(
            oai.UpdateRequest(filter=self.filter1, namespace="ns")
        )

    def test_update_withFilterAndSetMetadata_updateWithFilterAndSetMetadata(self, mocker):
        mocker.patch.object(self.index._vector_api, "update_vector", autospec=True)
        self.index.update(set_metadata=self.md1, filter=self.filter1, namespace="ns")
        self.index._vector_api.update_vector.assert_called_once_with(
            oai.UpdateRequest(set_metadata=self.md1, filter=self.filter1, namespace="ns")
        )

    def test_update_withFilterAndValues_updateWithFilterAndValues(self, mocker):
        mocker.patch.object(self.index._vector_api, "update_vector", autospec=True)
        self.index.update(values=self.vals1, filter=self.filter1, namespace="ns")
        self.index._vector_api.update_vector.assert_called_once_with(
            oai.UpdateRequest(values=self.vals1, filter=self.filter1, namespace="ns")
        )

    def test_update_withFilterAndAllParams_updateWithFilterAndAllParams(self, mocker):
        mocker.patch.object(self.index._vector_api, "update_vector", autospec=True)
        self.index.update(
            values=self.vals1,
            set_metadata=self.md1,
            sparse_values=self.sv1,
            filter=self.filter1,
            namespace="ns",
        )
        self.index._vector_api.update_vector.assert_called_once_with(
            oai.UpdateRequest(
                values=self.vals1,
                set_metadata=self.md1,
                sparse_values=oai.SparseValues(indices=self.svi1, values=self.svv1),
                filter=self.filter1,
                namespace="ns",
            )
        )

    def test_update_withoutFilter_backwardCompatibility(self, mocker):
        """Test that update without filter still works (backward compatibility)."""
        mocker.patch.object(self.index._vector_api, "update_vector", autospec=True)
        self.index.update(id="vec1", values=self.vals1, namespace="ns")
        self.index._vector_api.update_vector.assert_called_once_with(
            oai.UpdateRequest(id="vec1", values=self.vals1, namespace="ns")
        )

    def test_update_withFilterOnly_noId(self, mocker):
        """Test update with filter only (no id) for bulk updates."""
        mocker.patch.object(self.index._vector_api, "update_vector", autospec=True)
        self.index.update(set_metadata=self.md1, filter=self.filter1, namespace="ns")
        self.index._vector_api.update_vector.assert_called_once_with(
            oai.UpdateRequest(set_metadata=self.md1, filter=self.filter1, namespace="ns")
        )

    def test_update_withNeitherIdNorFilter_raisesError(self, mocker):
        """Test that update raises error when neither id nor filter is provided."""
        mocker.patch.object(self.index._vector_api, "update_vector", autospec=True)
        with pytest.raises(ValueError, match="Either 'id' or 'filter' must be provided"):
            self.index.update(values=self.vals1, namespace="ns")

    def test_update_withBothIdAndFilter_raisesError(self, mocker):
        """Test that update raises error when both id and filter are provided."""
        mocker.patch.object(self.index._vector_api, "update_vector", autospec=True)
        with pytest.raises(ValueError, match="Cannot provide both 'id' and 'filter'"):
            self.index.update(id="vec1", filter=self.filter1, values=self.vals1, namespace="ns")

    def test_update_withDryRun_updateWithDryRun(self, mocker):
        """Test update with dry_run parameter."""
        mocker.patch.object(self.index._vector_api, "update_vector", autospec=True)
        self.index.update(filter=self.filter1, dry_run=True, namespace="ns")
        self.index._vector_api.update_vector.assert_called_once_with(
            oai.UpdateRequest(filter=self.filter1, dry_run=True, namespace="ns")
        )

    def test_update_withDryRunAndSetMetadata_updateWithDryRunAndSetMetadata(self, mocker):
        """Test update with dry_run and set_metadata."""
        mocker.patch.object(self.index._vector_api, "update_vector", autospec=True)
        self.index.update(set_metadata=self.md1, filter=self.filter1, dry_run=True, namespace="ns")
        self.index._vector_api.update_vector.assert_called_once_with(
            oai.UpdateRequest(
                set_metadata=self.md1, filter=self.filter1, dry_run=True, namespace="ns"
            )
        )

    def test_update_withDryRunFalse_updateWithDryRunFalse(self, mocker):
        """Test update with dry_run=False."""
        mocker.patch.object(self.index._vector_api, "update_vector", autospec=True)
        self.index.update(filter=self.filter1, dry_run=False, namespace="ns")
        self.index._vector_api.update_vector.assert_called_once_with(
            oai.UpdateRequest(filter=self.filter1, dry_run=False, namespace="ns")
        )

    def test_update_withDryRunAndAllParams_updateWithDryRunAndAllParams(self, mocker):
        """Test update with dry_run and all parameters."""
        mocker.patch.object(self.index._vector_api, "update_vector", autospec=True)
        self.index.update(
            values=self.vals1,
            set_metadata=self.md1,
            sparse_values=self.sv1,
            filter=self.filter1,
            dry_run=True,
            namespace="ns",
        )
        self.index._vector_api.update_vector.assert_called_once_with(
            oai.UpdateRequest(
                values=self.vals1,
                set_metadata=self.md1,
                sparse_values=oai.SparseValues(indices=self.svi1, values=self.svv1),
                filter=self.filter1,
                dry_run=True,
                namespace="ns",
            )
        )

    # endregion

    # region: asyncio update tests

    @pytest.mark.asyncio
    async def test_asyncio_update_withDryRun_updateWithDryRun(self, mocker):
        """Test asyncio update with dry_run parameter."""
        asyncio_index = _IndexAsyncio(api_key="asdf", host="https://test.pinecone.io")
        mocker.patch.object(asyncio_index._vector_api, "update_vector", autospec=True)
        await asyncio_index.update(filter=self.filter1, dry_run=True, namespace="ns")
        asyncio_index._vector_api.update_vector.assert_called_once_with(
            oai.UpdateRequest(filter=self.filter1, dry_run=True, namespace="ns")
        )

    @pytest.mark.asyncio
    async def test_asyncio_update_withDryRunAndSetMetadata_updateWithDryRunAndSetMetadata(
        self, mocker
    ):
        """Test asyncio update with dry_run and set_metadata."""
        asyncio_index = _IndexAsyncio(api_key="asdf", host="https://test.pinecone.io")
        mocker.patch.object(asyncio_index._vector_api, "update_vector", autospec=True)
        await asyncio_index.update(
            set_metadata=self.md1, filter=self.filter1, dry_run=True, namespace="ns"
        )
        asyncio_index._vector_api.update_vector.assert_called_once_with(
            oai.UpdateRequest(
                set_metadata=self.md1, filter=self.filter1, dry_run=True, namespace="ns"
            )
        )

    @pytest.mark.asyncio
    async def test_asyncio_update_withDryRunFalse_updateWithDryRunFalse(self, mocker):
        """Test asyncio update with dry_run=False."""
        asyncio_index = _IndexAsyncio(api_key="asdf", host="https://test.pinecone.io")
        mocker.patch.object(asyncio_index._vector_api, "update_vector", autospec=True)
        await asyncio_index.update(filter=self.filter1, dry_run=False, namespace="ns")
        asyncio_index._vector_api.update_vector.assert_called_once_with(
            oai.UpdateRequest(filter=self.filter1, dry_run=False, namespace="ns")
        )

    @pytest.mark.asyncio
    async def test_asyncio_update_withDryRunAndAllParams_updateWithDryRunAndAllParams(self, mocker):
        """Test asyncio update with dry_run and all parameters."""
        asyncio_index = _IndexAsyncio(api_key="asdf", host="https://test.pinecone.io")
        mocker.patch.object(asyncio_index._vector_api, "update_vector", autospec=True)
        await asyncio_index.update(
            values=self.vals1,
            set_metadata=self.md1,
            sparse_values=self.sv1,
            filter=self.filter1,
            dry_run=True,
            namespace="ns",
        )
        asyncio_index._vector_api.update_vector.assert_called_once_with(
            oai.UpdateRequest(
                values=self.vals1,
                set_metadata=self.md1,
                sparse_values=oai.SparseValues(indices=self.svi1, values=self.svv1),
                filter=self.filter1,
                dry_run=True,
                namespace="ns",
            )
        )

    # endregion

    # region: describe index tests

    def test_describeIndexStats_callWithoutFilter_CalledWithoutFilter(self, mocker):
        mocker.patch.object(self.index._vector_api, "describe_index_stats", autospec=True)
        self.index.describe_index_stats()
        self.index._vector_api.describe_index_stats.assert_called_once_with(
            oai.DescribeIndexStatsRequest()
        )

    def test_describeIndexStats_callWithFilter_CalledWithFilter(self, mocker):
        mocker.patch.object(self.index._vector_api, "describe_index_stats", autospec=True)
        self.index.describe_index_stats(filter=self.filter1)
        self.index._vector_api.describe_index_stats.assert_called_once_with(
            oai.DescribeIndexStatsRequest(filter=self.filter1)
        )

    # endregion
