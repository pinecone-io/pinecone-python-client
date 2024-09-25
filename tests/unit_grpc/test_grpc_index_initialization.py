import re
from pinecone.grpc import PineconeGRPC, GRPCClientConfig


class TestGRPCIndexInitialization:
    def test_init_with_default_config(self):
        pc = PineconeGRPC(api_key="YOUR_API_KEY")
        index = pc.Index(name="my-index", host="host")

        assert index.grpc_client_config.secure == True
        assert index.grpc_client_config.timeout == 20
        assert index.grpc_client_config.conn_timeout == 1
        assert index.grpc_client_config.reuse_channel == True
        assert index.grpc_client_config.retry_config is None
        assert index.grpc_client_config.grpc_channel_options is None
        assert index.grpc_client_config.additional_metadata is None

        # Default metadata, grpc equivalent to http request headers
        assert len(index.fixed_metadata) == 3
        assert index.fixed_metadata["api-key"] == "YOUR_API_KEY"
        assert index.fixed_metadata["service-name"] == "my-index"
        assert index.fixed_metadata["client-version"] is not None

    def test_init_with_additional_metadata(self):
        pc = PineconeGRPC(api_key="YOUR_API_KEY")
        config = GRPCClientConfig(
            additional_metadata={"debug-header": "value123", "debug-header2": "value456"}
        )
        index = pc.Index(name="my-index", host="host", grpc_config=config)
        assert len(index.fixed_metadata) == 5
        assert index.fixed_metadata["api-key"] == "YOUR_API_KEY"
        assert index.fixed_metadata["service-name"] == "my-index"
        assert index.fixed_metadata["client-version"] is not None
        assert index.fixed_metadata["debug-header"] == "value123"
        assert index.fixed_metadata["debug-header2"] == "value456"

    def test_init_with_grpc_config_from_dict(self):
        pc = PineconeGRPC(api_key="YOUR_API_KEY")
        config = GRPCClientConfig._from_dict({"timeout": 10})
        index = pc.Index(name="my-index", host="host", grpc_config=config)

        assert index.grpc_client_config.timeout == 10

        # Unset fields still get default values
        assert index.grpc_client_config.reuse_channel == True
        assert index.grpc_client_config.secure == True

    def test_init_with_grpc_config_non_dict(self):
        pc = PineconeGRPC(api_key="YOUR_API_KEY")
        config = GRPCClientConfig(timeout=10, secure=False)
        index = pc.Index(name="my-index", host="host", grpc_config=config)

        assert index.grpc_client_config.timeout == 10
        assert index.grpc_client_config.secure == False

        # Unset fields still get default values
        assert index.grpc_client_config.reuse_channel == True
        assert index.grpc_client_config.conn_timeout == 1

    def test_config_passed_when_target_by_name(self):
        pc = PineconeGRPC(api_key="YOUR_API_KEY")

        # Set this state in the host store to skip network call
        # to find host for name
        pc.index_host_store.set_host(pc.config, "my-index", "myhost")

        config = GRPCClientConfig(timeout=10, secure=False)
        index = pc.Index(name="my-index", grpc_config=config)

        assert index.grpc_client_config.timeout == 10
        assert index.grpc_client_config.secure == False

        # Unset fields still get default values
        assert index.grpc_client_config.reuse_channel == True
        assert index.grpc_client_config.conn_timeout == 1

    def test_config_passed_when_target_by_host(self):
        pc = PineconeGRPC(api_key="YOUR_API_KEY")
        config = GRPCClientConfig(timeout=5, secure=True)
        index = pc.Index(host="myhost", grpc_config=config)

        assert index.grpc_client_config.timeout == 5
        assert index.grpc_client_config.secure == True

        # Unset fields still get default values
        assert index.grpc_client_config.reuse_channel == True
        assert index.grpc_client_config.conn_timeout == 1

        # Endpoint port defaults to 443
        assert index._endpoint() == "myhost:443"

    def test_config_passed_when_target_by_host_and_port(self):
        pc = PineconeGRPC(api_key="YOUR_API_KEY")
        config = GRPCClientConfig(timeout=5, secure=False)
        index = pc.Index(host="myhost:4343", grpc_config=config)

        assert index.grpc_client_config.timeout == 5
        assert index.grpc_client_config.secure == False

        # Unset fields still get default values
        assert index.grpc_client_config.reuse_channel == True
        assert index.grpc_client_config.conn_timeout == 1

        # Endpoint calculation does not override port
        assert index._endpoint() == "myhost:4343"

    def test_config_passes_source_tag_when_set(self):
        pc = PineconeGRPC(api_key="YOUR_API_KEY", source_tag="my_source_tag")
        assert (
            re.search(r"source_tag=my_source_tag", pc.index_api.api_client.user_agent) is not None
        )
