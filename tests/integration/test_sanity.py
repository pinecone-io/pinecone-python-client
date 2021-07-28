import logging
import os
import sys

import pinecone

# use: export PINECONE_API_KEY=foobar; export PINECONE_PROJECT_NAME=beni; export PINECONE_ENVIROMENT=alpha; python test_sanity.py
from pinecone.index import Index
from pinecone.protos.vector_service_pb2 import UpsertRequest, QueryRequest, DenseVector, AnonymousVector


def test_grpc():
    index = Index('test-1')
    logging.info('got grpc upsert response: %s', index.Upsert(
        UpsertRequest(vectors=[
            DenseVector(id='A', values=[0.1, 0.2, 0.3]),
            DenseVector(id='B', values=[0.2, 0.3, 0.4]),
            DenseVector(id='C', values=[0.3, 0.4, 0.5]),
        ])
    ))
    logging.info('got grpc query response: %s', index.Query(
        QueryRequest(
            queries=[
                QueryRequest.QueryVector(
                    vector=AnonymousVector(values=[0.1, 0.1, 0.1])
                ),
                QueryRequest.QueryVector(
                    vector=AnonymousVector(values=[0.1, 0.2, 0.3])
                )
            ],
            request_default_top_k=2,
            include_data=True
        )
    ))


def test_openapi():
    index_name = 'test-1'
    project_name = 'beni'
    env = 'dev-benjaminran'
    api_key = os.getenv('API_KEY')

    import pinecone.openapi
    from pinecone.openapi.api import vector_service_api
    from pprint import pprint
    configuration = pinecone.openapi.Configuration(
        host=f"https://{index_name}-{project_name}.{env}.svc.pinecone.io"
    )
    # Configure API key authorization: ApiKeyAuth
    configuration.api_key['Api-Key'] = api_key

    with pinecone.openapi.ApiClient(configuration) as api_client:
        api_instance = vector_service_api.VectorServiceApi(api_client)
        request_id = "requestId_example"  # str | Unique id of the request. (optional)
        ids = [
            "ids_example",
        ]
        delete_all = True
        namespace = "namespace_example"
        try:
            # The Delete operation deletes a vector by id.
            api_response = api_instance.vector_service_delete(
                request_id=request_id, ids=ids, delete_all=delete_all,
                namespace=namespace)
            pprint(api_response)
        except pinecone.openapi.ApiException as e:
            print("Exception when calling VectorServiceApi->vector_service_delete: %s\n" % e)


def test_all_legacy():
    # Create an index
    logging.info('create_index result: %s', pinecone.create_index("hello-pinecone-index", metric="euclidean"))

    # Connect to the index
    index = pinecone.Index("hello-pinecone-index")

    # Insert the data
    ids = ["A", "B", "C", "D", "E"],
    vectors = [[1]*3, [2]*3, [3]*3, [4]*3, [5]*3]
    logging.info('upsert result: %s', index.upsert(items=zip(ids, vectors)))

    # Fetch data
    logging.info('fetch result: %s', index.fetch(ids=ids[1:3]))

    # Query the index and get similar vectors
    logging.info('query result: %s', index.query(queries=[[0, 1]], top_k=3))

    # Delete data
    logging.info('delete result: %s', index.delete(ids=ids[1:3]))

    # Get index info
    logging.info('info result: %s', index.info())

    # Delete the index
    logging.info('delete_index result: %s', pinecone.delete_index("hello-pinecone-index"))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    test_grpc()
    test_openapi()
    # test_all_legacy()
