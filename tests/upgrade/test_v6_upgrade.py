import pinecone
import logging

logger = logging.getLogger(__name__)


class TestExpectedImports_UpgradeFromV6:
    def test_mapped_data_imports(self):
        data_imports = [
            "Vector",
            "QueryRequest",
            "FetchResponse",
            "DeleteRequest",
            "DescribeIndexStatsRequest",
            "DescribeIndexStatsResponse",
            "RpcStatus",
            "ScoredVector",
            "ServiceException",
            "SingleQueryResults",
            "QueryResponse",
            "RerankModel",
            "SearchQuery",
            "SearchQueryVector",
            "SearchRerank",
            "UpsertResponse",
            "UpdateRequest",
        ]

        control_imports = [
            "CollectionDescription",
            "CollectionList",
            "ServerlessSpec",
            "ServerlessSpecDefinition",
            "PodSpec",
            "PodSpecDefinition",
            # 'ForbiddenException',
            # 'ImportErrorMode',
            # 'Index',
            "IndexList",
            "IndexModel",
            # 'ListConversionException',
            # 'MetadataDictionaryExpectedError',
            # 'NotFoundException',
        ]

        config_imports = [
            "Config",
            "ConfigBuilder",
            "PineconeConfig",
            "PineconeConfigurationError",
            "PineconeException",
            "PineconeProtocolError",
            "PineconeApiAttributeError",
            "PineconeApiException",
        ]

        exception_imports = [
            "PineconeConfigurationError",
            "PineconeProtocolError",
            "PineconeException",
            "PineconeApiAttributeError",
            "PineconeApiTypeError",
            "PineconeApiValueError",
            "PineconeApiKeyError",
            "PineconeApiException",
            "NotFoundException",
            "UnauthorizedException",
            "ForbiddenException",
            "ServiceException",
            "ListConversionException",
        ]
        mapped_imports = data_imports + control_imports + config_imports + exception_imports

        for import_name in mapped_imports:
            assert hasattr(pinecone, import_name), f"Import {import_name} not found in pinecone"

    def test_v6_upgrade_root_imports(self):
        v6_dir_items = [
            "CollectionDescription",
            "CollectionList",
            "Config",
            "ConfigBuilder",
            "DeleteRequest",
            "DescribeIndexStatsRequest",
            "DescribeIndexStatsResponse",
            "FetchResponse",
            "ForbiddenException",
            "ImportErrorMode",
            "Index",
            "IndexList",
            "IndexModel",
            "ListConversionException",
            "MetadataDictionaryExpectedError",
            "NotFoundException",
            "Pinecone",
            "PineconeApiAttributeError",
            "PineconeApiException",
            "PineconeApiKeyError",
            "PineconeApiTypeError",
            "PineconeApiValueError",
            "PineconeConfig",
            "PineconeConfigurationError",
            "PineconeException",
            "PineconeProtocolError",
            "PodSpec",
            "PodSpecDefinition",
            "QueryRequest",
            "QueryResponse",
            "RpcStatus",
            "ScoredVector",
            "ServerlessSpec",
            "ServerlessSpecDefinition",
            "ServiceException",
            "SingleQueryResults",
            "SparseValues",
            "SparseValuesDictionaryExpectedError",
            "SparseValuesMissingKeysError",
            "SparseValuesTypeError",
            "TqdmExperimentalWarning",
            "UnauthorizedException",
            "UpdateRequest",
            "UpsertRequest",
            "UpsertResponse",
            "Vector",
            "VectorDictionaryExcessKeysError",
            "VectorDictionaryMissingKeysError",
            "VectorTupleLengthError",
            "__builtins__",
            "__cached__",
            "__doc__",
            "__file__",
            "__loader__",
            "__name__",
            "__package__",
            "__path__",
            "__spec__",
            "__version__",
            "config",
            "configure_index",
            "control",
            "core",
            "core_ea",
            "create_collection",
            "create_index",
            "data",
            "delete_collection",
            "delete_index",
            "deprecation_warnings",
            "describe_collection",
            "describe_index",
            "errors",
            "exceptions",
            "features",
            "index",
            "index_host_store",
            "init",
            "install_repr_overrides",
            "langchain_import_warnings",
            "list_collections",
            "list_indexes",
            "logging",
            "models",
            "openapi",
            "os",
            "pinecone",
            "pinecone_config",
            "repr_overrides",
            "scale_index",
            "sparse_vector_factory",
            "utils",
            "vector_factory",
            "warnings",
        ]

        intentionally_removed_items = ["os"]

        expected_items = [item for item in v6_dir_items if item not in intentionally_removed_items]

        missing_items = []
        for item in expected_items:
            if not hasattr(pinecone, item):
                missing_items.append(item)
                logger.debug(f"Exported: ❌ {item}")
            else:
                logger.debug(f"Exported: ✅ {item}")

        extra_items = []
        for item in intentionally_removed_items:
            if hasattr(pinecone, item):
                extra_items.append(item)
                logger.debug(f"Removed: ❌ {item}")
            else:
                logger.debug(f"Removed: ✅ {item}")

        assert len(missing_items) == 0, f"Missing items: {missing_items}"
        assert len(extra_items) == 0, f"Extra items: {extra_items}"

    # def test_v6_upgrade_data_imports(self):
    #     v6_data_dir_items = [
    #         "DescribeIndexStatsResponse",
    #         "EmbedModel",
    #         "FetchResponse",
    #         "ImportErrorMode",
    #         "Index",
    #         "IndexClientInstantiationError",
    #         "Inference",
    #         "InferenceInstantiationError",
    #         "MetadataDictionaryExpectedError",
    #         "QueryResponse",
    #         "RerankModel",
    #         "SearchQuery",
    #         "SearchQueryVector",
    #         "SearchRerank",
    #         "SparseValues",
    #         "SparseValuesDictionaryExpectedError",
    #         "SparseValuesMissingKeysError",
    #         "SparseValuesTypeError",
    #         "UpsertResponse",
    #         "Vector",
    #         "VectorDictionaryExcessKeysError",
    #         "VectorDictionaryMissingKeysError",
    #         "VectorTupleLengthError",
    #         "_AsyncioInference",
    #         "_Index",
    #         "_IndexAsyncio",
    #         "_Inference",
    #         "__builtins__",
    #         "__cached__",
    #         "__doc__",
    #         "__file__",
    #         "__loader__",
    #         "__name__",
    #         "__package__",
    #         "__path__",
    #         "__spec__",
    #         "dataclasses",
    #         "errors",
    #         "features",
    #         "fetch_response",
    #         "import_error",
    #         "index",
    #         "index_asyncio",
    #         "index_asyncio_interface",
    #         "interfaces",
    #         "query_results_aggregator",
    #         "request_factory",
    #         "search_query",
    #         "search_query_vector",
    #         "search_rerank",
    #         "sparse_values",
    #         "sparse_values_factory",
    #         "types",
    #         "utils",
    #         "vector",
    #         "vector_factory",
    #     ]

    #     missing_items = []
    #     for item in v6_data_dir_items:
    #         if item not in dir(pinecone.db_data):
    #             missing_items.append(item)

    #     assert len(missing_items) == 0, f"Missing items: {missing_items}"
