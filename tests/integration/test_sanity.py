import argparse
import logging
import os

import pinecone

# use: export PINECONE_API_KEY=foobar; export PINECONE_PROJECT_NAME=beni; export PINECONE_ENVIROMENT=alpha; python test_sanity.py
from pinecone.experimental.index_grpc import Index
from pinecone.protos.vector_service_pb2 import UpsertRequest, QueryRequest, DenseVector, AnonymousVector, DeleteRequest


def manual_test_grpc(args):
    index = Index(args.index_name)
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


def manual_test_openapi(args):
    import pinecone.experimental.openapi
    from pinecone.experimental.openapi.api import vector_service_api
    # import pinecone.experimental.openapi.exceptions
    # import pinecone.experimental.openapi.configuration
    from pprint import pprint

    configuration = pinecone.experimental.openapi.Configuration(
        host=f"https://{args.index_name}-{args.project_name}.svc.{args.pinecone_env}.pinecone.io",
        api_key={'ApiKeyAuth': args.api_key}
    )
    # configuration.verify_ssl = False
    # configuration.proxy = 'http://localhost:8111'

    with pinecone.experimental.openapi.ApiClient(configuration) as api_client:
        api_instance = vector_service_api.VectorServiceApi(api_client)
        try:
            api_response = api_instance.vector_service_delete(
                request_id='1234', ids=['A', 'B'], delete_all=False,
                namespace='ns1')
            pprint(api_response)
        except pinecone.experimental.openapi.OpenApiException as e:
            logging.exception("Exception when calling VectorServiceApi->vector_service_delete", e)


def manual_test_all_legacy():
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--api-key', default='701868b2-2f96-442e-97fd-4430dafe728d')
    parser.add_argument('--project-name', default='sharechat')
    parser.add_argument('--pinecone-env', default='sharechat-production')
    parser.add_argument('--index-name')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info('invoked with args: %s', args)
    pinecone.init(project_name=args.project_name, api_key=args.api_key, environment=args.pinecone_env)
    logging.info('config: %s', pinecone.Config._config._asdict())

    # manual_test_grpc(args)
    manual_test_openapi(args)
    # manual_test_all_legacy()
