
class TestRestIndex:

    def setup_method(self):
        self.vector_dim = 8
        self.vals1 = [0.1] * self.vector_dim
        self.vals2 = [0.2] * self.vector_dim
        self.md1 = {'genre': 'action', 'year': 2021}
        self.md2 = {'genre': 'documentary', 'year': 2020}
        self.filter1 = {'genre': {'$in': ['action']}}
        self.filter2 = {'year': {'$eq': 2020}}

    def test_upsert_request_tuples_id_data(self, mocker):
        import pinecone
        pinecone.init(api_key='example-key')
        index = pinecone.Index('example-name')
        mocker.patch.object(index._vector_api, 'upsert', autospec=True)
        index.upsert([('vec1', self.vals1), ('vec2', self.vals2)], namespace='ns')
        index._vector_api.upsert.assert_called_once_with(
            pinecone.UpsertRequest(vectors=[
                pinecone.Vector(id='vec1', values=self.vals1, metadata={}),
                pinecone.Vector(id='vec2', values=self.vals2, metadata={})
            ], namespace='ns')
        )

    def test_upsert_request_tuples_id_data_metadata(self, mocker):
        import pinecone
        pinecone.init(api_key='example-key')
        index = pinecone.Index('example-name')
        mocker.patch.object(index._vector_api, 'upsert', autospec=True)
        index.upsert([('vec1', self.vals1, self.md1),
                      ('vec2', self.vals2, self.md2)])
        index._vector_api.upsert.assert_called_once_with(
            pinecone.UpsertRequest(vectors=[
                pinecone.Vector(id='vec1', values=self.vals1, metadata=self.md1),
                pinecone.Vector(id='vec2', values=self.vals2, metadata=self.md2)
            ])
        )

    def test_query_request_tuples_query_only(self, mocker):
        import pinecone
        pinecone.init(api_key='example-key')
        index = pinecone.Index('example-name')
        mocker.patch.object(index._vector_api, 'query', autospec=True)
        index.query(top_k=10, vector=self.vals1)
        index._vector_api.query.assert_called_once_with(
            pinecone.QueryRequest(top_k=10, vector=self.vals1)
        )

    def test_query_request_tuples_query_filter(self, mocker):
        import pinecone
        pinecone.init(api_key='example-key')
        index = pinecone.Index('example-name')
        mocker.patch.object(index._vector_api, 'query', autospec=True)
        index.query(top_k=10, queries=[
            (self.vals1, self.filter1),
            (self.vals2, self.filter2)
        ])
        index._vector_api.query.assert_called_once_with(
            pinecone.QueryRequest(top_k=10, queries=[
                pinecone.QueryVector(values=self.vals1, filter=self.filter1),
                pinecone.QueryVector(values=self.vals2, filter=self.filter2)
            ])
        )
