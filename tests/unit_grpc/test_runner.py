import uuid

from grpc import Compression

from pinecone.config import Config
from pinecone.grpc.config import GRPCClientConfig
from pinecone.grpc.grpc_runner import GrpcRunner
from pinecone.utils.constants import CLIENT_VERSION


class TestGrpcRunner:
    def test_run_with_default_metadata(self, mocker):
        config = Config(api_key="YOUR_API_KEY")
        runner = GrpcRunner(index_name="my-index", config=config, grpc_config=GRPCClientConfig())

        mock_func = mocker.Mock()
        runner.run(mock_func, request="request")

        passed_metadata = mock_func.call_args.kwargs["metadata"]
        # Fixed metadata fields
        assert ("api-key", "YOUR_API_KEY") in passed_metadata
        assert ("service-name", "my-index") in passed_metadata
        assert ("client-version", CLIENT_VERSION) in passed_metadata

        # Request id assigned for each request
        assert any(item[0] == "request_id" for item in passed_metadata), (
            "request_id not found in metadata"
        )
        for items in passed_metadata:
            if items[0] == "request_id":
                assert isinstance(items[1], str)
                assert uuid.UUID(items[1], version=4), "request_id is not a valid UUID"

    def test_each_run_gets_unique_request_id(self, mocker):
        config = Config(api_key="YOUR_API_KEY")
        runner = GrpcRunner(index_name="my-index", config=config, grpc_config=GRPCClientConfig())

        mock_func = mocker.Mock()
        runner.run(mock_func, request="request")

        for items in mock_func.call_args.kwargs["metadata"]:
            if items[0] == "request_id":
                first_request_id = items[1]

        mock_func.reset_mock()
        runner.run(mock_func, request="request")
        for items in mock_func.call_args.kwargs["metadata"]:
            if items[0] == "request_id":
                second_request_id = items[1]
                assert second_request_id != first_request_id, (
                    "request_id is not unique for each request"
                )

    def test_run_with_additional_metadata_from_grpc_config(self, mocker):
        config = Config(api_key="YOUR_API_KEY")
        grpc_config = GRPCClientConfig(
            additional_metadata={"debug-header": "value123", "debug-header2": "value456"}
        )
        runner = GrpcRunner(index_name="my-index", config=config, grpc_config=grpc_config)

        mock_func = mocker.Mock()
        runner.run(mock_func, request="request")

        passed_metadata = mock_func.call_args.kwargs["metadata"]
        assert ("api-key", "YOUR_API_KEY") in passed_metadata
        assert ("service-name", "my-index") in passed_metadata
        assert ("client-version", CLIENT_VERSION) in passed_metadata
        assert ("debug-header", "value123") in passed_metadata
        assert ("debug-header2", "value456") in passed_metadata

    def test_with_additional_metadata_from_run(self, mocker):
        config = Config(api_key="YOUR_API_KEY")
        grpc_config = GRPCClientConfig(
            additional_metadata={"debug-header": "value123", "debug-header2": "value456"}
        )
        runner = GrpcRunner(index_name="my-index", config=config, grpc_config=grpc_config)

        mock_func = mocker.Mock()
        runner.run(
            mock_func,
            request="request",
            metadata={"user-extra": "extra-value", "user-extra2": "extra-value2"},
        )

        passed_metadata = mock_func.call_args.kwargs["metadata"]

        # Fixed metadata fields
        assert ("api-key", "YOUR_API_KEY") in passed_metadata
        assert ("service-name", "my-index") in passed_metadata
        assert ("client-version", CLIENT_VERSION) in passed_metadata
        # Request id
        assert any(item[0] == "request_id" for item in passed_metadata), (
            "request_id not found in metadata"
        )
        # Extras from configuration
        assert ("debug-header", "value123") in passed_metadata
        assert ("debug-header2", "value456") in passed_metadata
        # Extras from call to run()
        assert ("user-extra", "extra-value") in passed_metadata
        assert ("user-extra2", "extra-value2") in passed_metadata

    def test_run_with_other_args(self, mocker):
        config = Config(api_key="YOUR_API_KEY")
        grpc_config = GRPCClientConfig(
            additional_metadata={"debug-header": "value123", "debug-header2": "value456"}
        )
        runner = GrpcRunner(index_name="my-index", config=config, grpc_config=grpc_config)

        mock_func = mocker.Mock()
        runner.run(
            mock_func,
            request="request",
            timeout=10,
            wait_for_ready=True,
            compression=Compression.Gzip,
        )

        assert mock_func.call_args.kwargs["timeout"] == 10
        assert mock_func.call_args.kwargs["wait_for_ready"] == True
        assert mock_func.call_args.kwargs["compression"] == Compression.Gzip
