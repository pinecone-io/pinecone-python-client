# from prometheus_client import Counter, Gauge, Histogram

from pinecone.protos import core_pb2


def operation_name_from_request(request: 'core_pb2.Request'):
    body = request.WhichOneof('body')
    if body == 'index':
        return 'upsert'
    elif body in ['delete', 'fetch', 'query', 'info', 'list']:
        return body
    else:
        return 'unknown'


# REQUEST_COUNT = \
#     Counter('pinecone_request_count',
#             'pinecone_request_count gives the number of data plane calls made by clients',
#             ['index_name', 'project_id', 'operation'])
#
# REQUEST_ERROR_COUNT = \
#     Counter('pinecone_request_error_count',
#             'pinecone_request_error_count gives the number of data plane calls made by clients that resulted in errors',
#             ['index_name', 'project_id', 'operation'])
#
# REQUEST_LATENCY = \
#     Histogram('pinecone_request_latency_seconds',
#             'pinecone_request_latency_seconds gives the distribution of server-side processing latency for pinecone data plane calls',
#             ['index_name', 'project_id', 'operation'])

ITEM_COUNT = None  #\
    # Gauge('pinecone_item_count',
    #       'pinecone_item_count gives the number of items in the pinecone index',
    #       ['index_name', 'project_id', 'namespace'])
