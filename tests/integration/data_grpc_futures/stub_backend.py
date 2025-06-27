import time
import grpc
import logging
from concurrent import futures
import pinecone.core.grpc.protos.db_data_2025_04_pb2 as pb2
import pinecone.core.grpc.protos.db_data_2025_04_pb2_grpc as pb2_grpc

logger = logging.getLogger(__name__)


class TestVectorService(pb2_grpc.VectorServiceServicer):
    def __init__(self, sleep_seconds=5):
        self.sleep_seconds = sleep_seconds

    def Upsert(self, request, context):
        # Simulate a delay that will cause a timeout
        logger.info("Received an upsert request from test client")
        logger.info(f"Request: {request}")
        logger.info(f"Sleeping for {self.sleep_seconds} seconds to simulate a slow server call")
        time.sleep(self.sleep_seconds)
        logger.info(f"Done sleeping for {self.sleep_seconds} seconds")
        logger.info("Returning an upsert response from test server")
        return pb2.UpsertResponse(upserted_count=1)

    def Query(self, request, context):
        # Simulate a delay that will cause a timeout
        logger.info("Received a query request from test client")
        logger.info(f"Request: {request}")

        logger.info(f"Sleeping for {self.sleep_seconds} seconds to simulate a slow server call")
        time.sleep(self.sleep_seconds)
        logger.info(f"Done sleeping for {self.sleep_seconds} seconds")
        logger.info("Returning a query response from test server")
        return pb2.QueryResponse(
            results=[],
            matches=[pb2.ScoredVector(id="1", score=1.0, values=[1.0, 2.0, 3.0])],
            namespace="testnamespace",
            usage=pb2.Usage(read_units=1),
        )

    def Update(self, request, context):
        # Simulate a delay that will cause a timeout
        logger.info("Received an update request from test client")
        logger.info(f"Request: {request}")
        logger.info(f"Sleeping for {self.sleep_seconds} seconds to simulate a slow server call")
        time.sleep(self.sleep_seconds)
        logger.info(f"Done sleeping for {self.sleep_seconds} seconds")
        logger.info("Returning an update response from test server")
        return pb2.UpdateResponse()

    def Delete(self, request, context):
        # Simulate a delay that will cause a timeout
        logger.info("Received a delete request from test client")
        logger.info(f"Request: {request}")
        logger.info(f"Sleeping for {self.sleep_seconds} seconds to simulate a slow server call")
        time.sleep(self.sleep_seconds)
        logger.info(f"Done sleeping for {self.sleep_seconds} seconds")
        logger.info("Returning a delete response from test server")
        return pb2.DeleteResponse()

    def Fetch(self, request, context):
        logger.info("Received a fetch request from test client")
        logger.info(f"Request: {request}")
        logger.info(f"Sleeping for {self.sleep_seconds} seconds to simulate a slow server call")
        time.sleep(self.sleep_seconds)
        logger.info(f"Done sleeping for {self.sleep_seconds} seconds")
        logger.info("Returning a fetch response from test server")
        return pb2.FetchResponse(
            vectors={
                "1": pb2.Vector(id="1", values=[1.0, 2.0, 3.0]),
                "2": pb2.Vector(id="2", values=[4.0, 5.0, 6.0]),
                "3": pb2.Vector(id="3", values=[7.0, 8.0, 9.0]),
            },
            namespace="testnamespace",
            usage=pb2.Usage(read_units=1),
        )


def create_sleepy_test_server(port=50051, sleep_seconds=5):
    """Creates and returns a configured gRPC server for testing.

    Args:
        port (int): The port number to run the server on
        sleep_seconds (int): The extra latency in seconds for simulated operations

    Returns:
        grpc.Server: A configured and started gRPC server instance
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_VectorServiceServicer_to_server(
        TestVectorService(sleep_seconds=sleep_seconds), server
    )
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    return server
