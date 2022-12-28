import pinecone
from pinecone import DescribeIndexStatsRequest


class TestRestIndex:

    def setup_method(self):
        self.vector_dim = 8
        self.vals1 = [0.1] * self.vector_dim
        self.vals2 = [0.2] * self.vector_dim
        self.md1 = {'genre': 'action', 'year': 2021}
        self.md2 = {'genre': 'documentary', 'year': 2020}
        self.filter1 = {'genre': {'$in': ['action']}}
        self.filter2 = {'year': {'$eq': 2020}}

        pinecone.init(api_key='example-key')
        self.index = pinecone.Index('example-name')

    # region: upsert tests

    def test_upsert_tuplesOfIdVec_UpserWithoutMD(self, mocker):
        mocker.patch.object(self.index._vector_api, 'upsert', autospec=True)
        self.index.upsert([('vec1', self.vals1), ('vec2', self.vals2)], namespace='ns')
        self.index._vector_api.upsert.assert_called_once_with(
            pinecone.UpsertRequest(vectors=[
                pinecone.Vector(id='vec1', values=self.vals1, metadata={}),
                pinecone.Vector(id='vec2', values=self.vals2, metadata={})
            ], namespace='ns')
        )

    def test_upsert_tuplesOfIdVecMD_UpsertVectorsWithMD(self, mocker):
        mocker.patch.object(self.index._vector_api, 'upsert', autospec=True)
        self.index.upsert([('vec1', self.vals1, self.md1),
                           ('vec2', self.vals2, self.md2)])
        self.index._vector_api.upsert.assert_called_once_with(
            pinecone.UpsertRequest(vectors=[
                pinecone.Vector(id='vec1', values=self.vals1, metadata=self.md1),
                pinecone.Vector(id='vec2', values=self.vals2, metadata=self.md2)
            ])
        )

    def test_upsert_vectors_upsertInputVectors(self, mocker):
        mocker.patch.object(self.index._vector_api, 'upsert', autospec=True)
        self.index.upsert(vectors=[
                pinecone.Vector(id='vec1', values=self.vals1, metadata=self.md1),
                pinecone.Vector(id='vec2', values=self.vals2, metadata=self.md2)],
            namespace='ns')
        self.index._vector_api.upsert.assert_called_once_with(
            pinecone.UpsertRequest(vectors=[
                pinecone.Vector(id='vec1', values=self.vals1, metadata=self.md1),
                pinecone.Vector(id='vec2', values=self.vals2, metadata=self.md2)
            ], namespace='ns')
        )

    # endregion

    # region: query tests

    def test_query_byVectorNoFilter_queryVectorNoFilter(self, mocker):
        mocker.patch.object(self.index._vector_api, 'query', autospec=True)
        self.index.query(top_k=10, vector=self.vals1)
        self.index._vector_api.query.assert_called_once_with(
            pinecone.QueryRequest(top_k=10, vector=self.vals1)
        )

    def test_query_byVectorWithFilter_queryVectorWithFilter(self, mocker):
        mocker.patch.object(self.index._vector_api, 'query', autospec=True)
        self.index.query(top_k=10, vector=self.vals1, filter=self.filter1, namespace='ns')
        self.index._vector_api.query.assert_called_once_with(
            pinecone.QueryRequest(top_k=10, vector=self.vals1, filter=self.filter1, namespace='ns')
        )

    def test_query_byTuplesNoFilter_queryVectorsNoFilter(self, mocker):
        mocker.patch.object(self.index._vector_api, 'query', autospec=True)
        self.index.query(top_k=10, queries=[
            (self.vals1,),
            (self.vals2,)
        ])
        self.index._vector_api.query.assert_called_once_with(
            pinecone.QueryRequest(top_k=10, queries=[
                pinecone.QueryVector(values=self.vals1),
                pinecone.QueryVector(values=self.vals2)
            ])
        )

    def test_query_byTuplesWithFilter_queryVectorsWithFilter(self, mocker):
        mocker.patch.object(self.index._vector_api, 'query', autospec=True)
        self.index.query(top_k=10, queries=[
            (self.vals1, self.filter1),
            (self.vals2, self.filter2)
        ])
        self.index._vector_api.query.assert_called_once_with(
            pinecone.QueryRequest(top_k=10, queries=[
                pinecone.QueryVector(values=self.vals1, filter=self.filter1),
                pinecone.QueryVector(values=self.vals2, filter=self.filter2)
            ])
        )

    def test_query_byVecId_queryByVecId(self, mocker):
        mocker.patch.object(self.index._vector_api, 'query', autospec=True)
        self.index.query(top_k=10, id='vec1', include_metadata=True, include_values=False)
        self.index._vector_api.query.assert_called_once_with(
            pinecone.QueryRequest(top_k=10, id='vec1', include_metadata=True, include_values=False)
        )

    # endregion

    # region: delete tests

    def test_delete_byIds_deleteByIds(self, mocker):
        mocker.patch.object(self.index._vector_api, 'delete', autospec=True)
        self.index.delete(ids=['vec1', 'vec2'])
        self.index._vector_api.delete.assert_called_once_with(
            pinecone.DeleteRequest(ids=['vec1', 'vec2'])
        )

    def test_delete_deleteAllByFilter_deleteAllByFilter(self, mocker):
        mocker.patch.object(self.index._vector_api, 'delete', autospec=True)
        self.index.delete(delete_all=True, filter=self.filter1, namespace='ns')
        self.index._vector_api.delete.assert_called_once_with(
            pinecone.DeleteRequest(delete_all=True, filter=self.filter1, namespace='ns')
        )

    def test_delete_deleteAllNoFilter_deleteNoFilter(self, mocker):
        mocker.patch.object(self.index._vector_api, 'delete', autospec=True)
        self.index.delete(delete_all=True)
        self.index._vector_api.delete.assert_called_once_with(
            pinecone.DeleteRequest(delete_all=True)
        )

    # endregion

    # region: fetch tests

    def test_fetch_byIds_fetchByIds(self, mocker):
        mocker.patch.object(self.index._vector_api, 'fetch', autospec=True)
        self.index.fetch(ids=['vec1', 'vec2'])
        self.index._vector_api.fetch.assert_called_once_with(
            ids=['vec1', 'vec2']
        )

    def test_fetch_byIdsAndNS_fetchByIdsAndNS(self, mocker):
        mocker.patch.object(self.index._vector_api, 'fetch', autospec=True)
        self.index.fetch(ids=['vec1', 'vec2'], namespace='ns')
        self.index._vector_api.fetch.assert_called_once_with(
            ids=['vec1', 'vec2'], namespace='ns'
        )

    # endregion

    # region: update tests

    def test_update_byIdAnValues_updateByIdAndValues(self, mocker):
        mocker.patch.object(self.index._vector_api, 'update', autospec=True)
        self.index.update(id='vec1', values=self.vals1, namespace='ns')
        self.index._vector_api.update.assert_called_once_with(
            pinecone.UpdateRequest(id='vec1', values=self.vals1, namespace='ns')
        )

    def test_update_byIdAnValuesAndMetadata_updateByIdAndValuesAndMetadata(self, mocker):
        mocker.patch.object(self.index._vector_api, 'update', autospec=True)
        self.index.update('vec1', values=self.vals1, metadata=self.md1)
        self.index._vector_api.update.assert_called_once_with(
            pinecone.UpdateRequest(id='vec1', values=self.vals1, metadata=self.md1)
        )

    # endregion

    # region: describe index tests

    def test_describeIndexStats_callWithoutFilter_CalledWithoutFilter(self, mocker):
        mocker.patch.object(self.index._vector_api, 'describe_index_stats', autospec=True)
        self.index.describe_index_stats()
        self.index._vector_api.describe_index_stats.assert_called_once_with(
            DescribeIndexStatsRequest())

    def test_describeIndexStats_callWithFilter_CalledWithFilter(self, mocker):
        mocker.patch.object(self.index._vector_api, 'describe_index_stats', autospec=True)
        self.index.describe_index_stats(filter=self.filter1)
        self.index._vector_api.describe_index_stats.assert_called_once_with(
            DescribeIndexStatsRequest(filter=self.filter1))

    # endregion
