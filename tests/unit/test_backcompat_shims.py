"""Import round-trip tests for pre-rewrite backcompat shim modules."""

from __future__ import annotations

import importlib

import pytest

# (legacy_path, symbol_name, canonical_path)
_REEXPORT_TRIPLES = [
    (
        "pinecone.core.openapi.db_data.models",
        "DescribeIndexStatsResponse",
        "pinecone.models.vectors.responses",
    ),
    (
        "pinecone.db_control.models.byoc_spec",
        "ByocSpec",
        "pinecone.models.indexes.specs",
    ),
    (
        "pinecone.db_data.dataclasses.fetch_by_metadata_response",
        "FetchByMetadataResponse",
        "pinecone.models.vectors.responses",
    ),
    (
        "pinecone.db_data.dataclasses.fetch_by_metadata_response",
        "Pagination",
        "pinecone.models.vectors.responses",
    ),
    (
        "pinecone.db_data.dataclasses.fetch_response",
        "FetchResponse",
        "pinecone.models.vectors.responses",
    ),
    (
        "pinecone.db_data.dataclasses.query_response",
        "QueryResponse",
        "pinecone.models.vectors.responses",
    ),
    (
        "pinecone.db_data.dataclasses.sparse_values",
        "SparseValues",
        "pinecone.models.vectors.sparse",
    ),
    (
        "pinecone.db_data.dataclasses.update_response",
        "UpdateResponse",
        "pinecone.models.vectors.responses",
    ),
    (
        "pinecone.db_data.dataclasses.upsert_response",
        "UpsertResponse",
        "pinecone.models.vectors.responses",
    ),
    (
        "pinecone.db_data.dataclasses.vector",
        "Vector",
        "pinecone.models.vectors.vector",
    ),
    (
        "pinecone.db_data.query_results_aggregator",
        "QueryResultsAggregator",
        "pinecone.models.vectors.query_aggregator",
    ),
    (
        "pinecone.db_data.query_results_aggregator",
        "QueryResultsAggregatorInvalidTopKError",
        "pinecone.models.vectors.query_aggregator",
    ),
]


@pytest.mark.parametrize("legacy_path,symbol_name,canonical_path", _REEXPORT_TRIPLES)
def test_shim_reexports_canonical(legacy_path: str, symbol_name: str, canonical_path: str) -> None:
    legacy_module = importlib.import_module(legacy_path)
    canonical_module = importlib.import_module(canonical_path)
    assert getattr(legacy_module, symbol_name) is getattr(canonical_module, symbol_name)


@pytest.mark.parametrize("legacy_path,symbol_name,canonical_path", _REEXPORT_TRIPLES)
def test_shim_module_has_all_matching_reexport(
    legacy_path: str, symbol_name: str, canonical_path: str
) -> None:
    legacy_module = importlib.import_module(legacy_path)
    assert symbol_name in legacy_module.__all__
    assert all(hasattr(legacy_module, name) for name in legacy_module.__all__)


class TestLegacyListResponse:
    def test_importable(self) -> None:
        from pinecone.db_control.models.list_response import ListResponse, Pagination

        assert ListResponse.__name__ == "ListResponse"
        assert Pagination.__name__ == "Pagination"

    def test_pagination_instantiation_and_access(self) -> None:
        from pinecone.db_control.models.list_response import Pagination

        p = Pagination(next="token-abc")
        assert p.next == "token-abc"

    def test_list_response_instantiation_and_access(self) -> None:
        from pinecone.db_control.models.list_response import ListResponse

        lr = ListResponse(namespace="ns", vectors=[], pagination=None)
        assert lr.namespace == "ns"
        assert lr.vectors == []
        assert lr.pagination is None

    def test_list_response_with_pagination(self) -> None:
        from pinecone.db_control.models.list_response import ListResponse, Pagination

        p = Pagination(next="t")
        lr = ListResponse(namespace="ns", vectors=[], pagination=p)
        assert lr.pagination is not None
        assert lr.pagination.next == "t"


class TestScoredVectorTopLevelExport:
    def test_importable(self) -> None:
        from pinecone import ScoredVector

        assert ScoredVector.__name__ == "ScoredVector"

    def test_is_canonical_class(self) -> None:
        import pinecone
        import pinecone.models.vectors.vector as canonical

        assert pinecone.ScoredVector is canonical.ScoredVector

    def test_in_all(self) -> None:
        import pinecone

        assert "ScoredVector" in pinecone.__all__

    def test_in_dir(self) -> None:
        import pinecone

        assert "ScoredVector" in dir(pinecone)


class TestDbDataDataclassesPackage:
    def test_package_exports_all_symbols(self) -> None:
        import pinecone.db_data.dataclasses as pkg

        for name in pkg.__all__:
            assert hasattr(pkg, name), f"__all__ lists {name!r} but it is not an attribute"


# (legacy_path, symbol_name, canonical_path)
_CONTROL_SHIM_TRIPLES = [
    ("pinecone.control", "AwsRegion", "pinecone.models.enums"),
    ("pinecone.control", "AzureRegion", "pinecone.models.enums"),
    ("pinecone.control", "BackupList", "pinecone.models.backups.list"),
    ("pinecone.control", "BackupModel", "pinecone.models.backups.model"),
    ("pinecone.control", "ByocSpec", "pinecone.models.indexes.specs"),
    ("pinecone.control", "CloudProvider", "pinecone.models.enums"),
    ("pinecone.control", "CollectionDescription", "pinecone.models.collections.description"),
    ("pinecone.control", "CollectionList", "pinecone.models.collections.list"),
    ("pinecone.control", "DeletionProtection", "pinecone.models.enums"),
    ("pinecone.control", "GcpRegion", "pinecone.models.enums"),
    ("pinecone.control", "IndexEmbed", "pinecone.inference.models.index_embed"),
    ("pinecone.control", "IndexList", "pinecone.models.indexes.list"),
    ("pinecone.control", "IndexModel", "pinecone.models.indexes.index"),
    ("pinecone.control", "Metric", "pinecone.models.enums"),
    ("pinecone.control", "PodIndexEnvironment", "pinecone.models.enums"),
    ("pinecone.control", "PodSpec", "pinecone.models.indexes.specs"),
    ("pinecone.control", "PodType", "pinecone.models.enums"),
    ("pinecone.control", "RestoreJobList", "pinecone.models.backups.list"),
    ("pinecone.control", "RestoreJobModel", "pinecone.models.backups.model"),
    ("pinecone.control", "ServerlessSpec", "pinecone.models.indexes.specs"),
    ("pinecone.control", "VectorType", "pinecone.models.enums"),
]


@pytest.mark.parametrize("legacy_path,symbol_name,canonical_path", _CONTROL_SHIM_TRIPLES)
def test_control_shim_reexports_canonical(
    legacy_path: str, symbol_name: str, canonical_path: str
) -> None:
    legacy_module = importlib.import_module(legacy_path)
    canonical_module = importlib.import_module(canonical_path)
    assert getattr(legacy_module, symbol_name) is getattr(canonical_module, symbol_name)


@pytest.mark.parametrize("legacy_path,symbol_name,canonical_path", _CONTROL_SHIM_TRIPLES)
def test_control_shim_module_has_all_matching_reexport(
    legacy_path: str, symbol_name: str, canonical_path: str
) -> None:
    legacy_module = importlib.import_module(legacy_path)
    assert symbol_name in legacy_module.__all__
    assert all(hasattr(legacy_module, name) for name in legacy_module.__all__)


def test_control_shim_all_matches_module_attrs() -> None:
    import pinecone.control

    expected = {
        "AwsRegion",
        "AzureRegion",
        "BackupList",
        "BackupModel",
        "ByocSpec",
        "CloudProvider",
        "CollectionDescription",
        "CollectionList",
        "DeletionProtection",
        "GcpRegion",
        "IndexEmbed",
        "IndexList",
        "IndexModel",
        "Metric",
        "PodIndexEnvironment",
        "PodSpec",
        "PodType",
        "RestoreJobList",
        "RestoreJobModel",
        "ServerlessSpec",
        "VectorType",
    }
    assert set(pinecone.control.__all__) == expected
    for name in expected:
        assert hasattr(pinecone.control, name), f"pinecone.control missing attribute {name!r}"


def test_control_shim_omits_renamed_symbols() -> None:
    import pinecone.control as ctrl

    omitted = [
        "PodSpecDefinition",
        "ServerlessSpecDefinition",
        "ConfigureIndexEmbed",
        "CreateIndexForModelEmbedTypedDict",
        "DBControl",
        "DBControlAsyncio",
    ]
    for name in omitted:
        assert not hasattr(ctrl, name), (
            f"`from pinecone.control import {name}` should raise ImportError"
        )


# (legacy_path, symbol_name, canonical_path)
_DATA_SHIM_TRIPLES = [
    ("pinecone.data", "DescribeIndexStatsResponse", "pinecone.models.vectors.responses"),
    ("pinecone.data", "FetchResponse", "pinecone.models.vectors.responses"),
    ("pinecone.data", "ImportErrorMode", "pinecone.models.imports.error_mode"),
    ("pinecone.data", "Index", "pinecone.index"),
    ("pinecone.data", "QueryResponse", "pinecone.models.vectors.responses"),
    ("pinecone.data", "SearchQuery", "pinecone.models.vectors.search"),
    ("pinecone.data", "SearchRerank", "pinecone.models.vectors.search"),
    ("pinecone.data", "SparseValues", "pinecone.models.vectors.sparse"),
    ("pinecone.data", "UpsertResponse", "pinecone.models.vectors.responses"),
    ("pinecone.data", "Vector", "pinecone.models.vectors.vector"),
]


@pytest.mark.parametrize("legacy_path,symbol_name,canonical_path", _DATA_SHIM_TRIPLES)
def test_data_shim_reexports_canonical(
    legacy_path: str, symbol_name: str, canonical_path: str
) -> None:
    legacy_module = importlib.import_module(legacy_path)
    canonical_module = importlib.import_module(canonical_path)
    assert getattr(legacy_module, symbol_name) is getattr(canonical_module, symbol_name)


@pytest.mark.parametrize("legacy_path,symbol_name,canonical_path", _DATA_SHIM_TRIPLES)
def test_data_shim_module_has_all_matching_reexport(
    legacy_path: str, symbol_name: str, canonical_path: str
) -> None:
    legacy_module = importlib.import_module(legacy_path)
    assert symbol_name in legacy_module.__all__
    assert all(hasattr(legacy_module, name) for name in legacy_module.__all__)


def test_data_shim_index_asyncio_alias() -> None:
    import pinecone.async_client.async_index as canonical
    import pinecone.data as data_shim

    assert data_shim.IndexAsyncio is canonical.AsyncIndex
    assert "IndexAsyncio" in data_shim.__all__


def test_data_shim_all_matches_module_attrs() -> None:
    import pinecone.data

    expected = {
        "DescribeIndexStatsResponse",
        "FetchResponse",
        "ImportErrorMode",
        "Index",
        "IndexAsyncio",
        "QueryResponse",
        "SearchQuery",
        "SearchRerank",
        "SparseValues",
        "UpsertResponse",
        "Vector",
    }
    assert set(pinecone.data.__all__) == expected
    for name in expected:
        assert hasattr(pinecone.data, name), f"pinecone.data missing attribute {name!r}"


class TestPineconeModuleShim:
    def test_import_works(self) -> None:
        from pinecone.pinecone import Pinecone

        assert Pinecone.__name__ == "Pinecone"

    def test_class_identity(self) -> None:
        import pinecone._client as canonical
        import pinecone.pinecone as shim

        assert shim.Pinecone is canonical.Pinecone

    def test_top_level_identity(self) -> None:
        import pinecone
        import pinecone.pinecone as shim

        assert shim.Pinecone is pinecone.Pinecone

    def test_all_matches(self) -> None:
        import pinecone.pinecone as shim

        assert shim.__all__ == ["Pinecone"]


class TestPineconeAsyncioModuleShim:
    def test_import_works(self) -> None:
        from pinecone.pinecone_asyncio import AsyncPinecone, PineconeAsyncio

        assert PineconeAsyncio.__name__ == "AsyncPinecone"
        assert AsyncPinecone.__name__ == "AsyncPinecone"

    def test_alias_identity(self) -> None:
        import pinecone.pinecone_asyncio as shim

        assert shim.PineconeAsyncio is shim.AsyncPinecone

    def test_canonical_identity(self) -> None:
        import pinecone.async_client.pinecone as canonical
        import pinecone.pinecone_asyncio as shim

        assert shim.AsyncPinecone is canonical.AsyncPinecone

    def test_top_level_identity(self) -> None:
        import pinecone
        import pinecone.pinecone_asyncio as shim

        assert shim.AsyncPinecone is pinecone.AsyncPinecone

    def test_all_matches(self) -> None:
        import pinecone.pinecone_asyncio as shim

        assert set(shim.__all__) == {"AsyncPinecone", "PineconeAsyncio"}


_SHIM_PACKAGE_EXPORTS: dict[str, set[str]] = {
    "pinecone.config": {"Config", "PineconeConfig"},
    "pinecone.db_control": {
        "AwsRegion",
        "AzureRegion",
        "BackupList",
        "BackupModel",
        "ByocSpec",
        "CloudProvider",
        "CollectionDescription",
        "CollectionList",
        "DeletionProtection",
        "GcpRegion",
        "IndexEmbed",
        "IndexList",
        "IndexModel",
        "Metric",
        "PodIndexEnvironment",
        "PodSpec",
        "PodType",
        "RestoreJobList",
        "RestoreJobModel",
        "ServerlessSpec",
        "VectorType",
    },
    "pinecone.db_control.enums": {
        "AwsRegion",
        "AzureRegion",
        "CloudProvider",
        "DeletionProtection",
        "GcpRegion",
        "Metric",
        "PodIndexEnvironment",
        "PodType",
        "VectorType",
    },
    "pinecone.db_control.models": {
        "BackupList",
        "BackupModel",
        "ByocSpec",
        "CollectionDescription",
        "CollectionList",
        "IndexEmbed",
        "IndexList",
        "IndexModel",
        "PodSpec",
        "RestoreJobList",
        "RestoreJobModel",
        "ServerlessSpec",
    },
    "pinecone.db_data": {
        "DescribeIndexStatsResponse",
        "FetchResponse",
        "ImportErrorMode",
        "Index",
        "IndexAsyncio",
        "QueryResponse",
        "SearchQuery",
        "SearchQueryVector",
        "SearchRerank",
        "SparseValues",
        "UpsertResponse",
        "Vector",
        "_Index",
        "_IndexAsyncio",
    },
    "pinecone.exceptions": {
        "ForbiddenException",
        "ListConversionException",
        "NotFoundException",
        "PineconeApiAttributeError",
        "PineconeApiException",
        "PineconeApiKeyError",
        "PineconeApiTypeError",
        "PineconeApiValueError",
        "PineconeConfigurationError",
        "PineconeException",
        "PineconeProtocolError",
        "RateLimitException",
        "ServiceException",
        "UnauthorizedException",
        "ValidationError",
    },
    "pinecone.inference.models": {
        "EmbeddingsList",
        "IndexEmbed",
        "ModelInfo",
        "ModelInfoList",
        "RerankResult",
    },
}


@pytest.mark.parametrize("module_path", sorted(_SHIM_PACKAGE_EXPORTS))
def test_shim_package_all_is_exhaustive(module_path: str) -> None:
    module = importlib.import_module(module_path)
    assert set(module.__all__) == _SHIM_PACKAGE_EXPORTS[module_path]


@pytest.mark.parametrize(
    "module_path,symbol_name",
    sorted(
        (module_path, symbol_name)
        for module_path, names in _SHIM_PACKAGE_EXPORTS.items()
        for symbol_name in names
    ),
)
def test_shim_package_all_names_import(module_path: str, symbol_name: str) -> None:
    module = importlib.import_module(module_path)
    assert getattr(module, symbol_name) is not None


def test_db_control_matches_control_shim() -> None:
    import pinecone.control as control
    import pinecone.db_control as db_control

    assert set(db_control.__all__) == set(control.__all__)
    for name in control.__all__:
        assert getattr(db_control, name) is getattr(control, name)


def test_db_data_matches_data_shim() -> None:
    import pinecone.data as data
    import pinecone.db_data as db_data

    for name in data.__all__:
        assert getattr(db_data, name) is getattr(data, name)


def test_db_data_underscore_aliases_are_the_index_classes() -> None:
    import pinecone.db_data as db_data

    assert db_data._Index is db_data.Index
    assert db_data._IndexAsyncio is db_data.IndexAsyncio


def test_config_shim_package_reexports_own_submodules() -> None:
    import pinecone.config as config
    import pinecone.config.config as config_module
    import pinecone.config.pinecone_config as pinecone_config_module

    assert config.Config is config_module.Config
    assert config.PineconeConfig is pinecone_config_module.PineconeConfig


def test_exceptions_shim_names_are_the_canonical_classes() -> None:
    import pinecone.errors.exceptions as canonical
    import pinecone.exceptions as shim

    for name in shim.__all__:
        assert getattr(shim, name) is getattr(canonical, name)


def test_inference_models_shim_reexports_canonical() -> None:
    import pinecone.inference.models as shim
    import pinecone.inference.models.index_embed as index_embed
    import pinecone.models.inference.embed as embed
    import pinecone.models.inference.model_list as model_list
    import pinecone.models.inference.models as models
    import pinecone.models.inference.rerank as rerank

    assert shim.EmbeddingsList is embed.EmbeddingsList
    assert shim.ModelInfo is models.ModelInfo
    assert shim.ModelInfoList is model_list.ModelInfoList
    assert shim.RerankResult is rerank.RerankResult
    assert shim.IndexEmbed is index_embed.IndexEmbed


def test_index_embed_is_not_marked_private() -> None:
    import pinecone.inference.models.index_embed as module
    from pinecone.inference.models.index_embed import IndexEmbed

    assert IndexEmbed.__doc__ is not None
    assert ":meta private:" not in IndexEmbed.__doc__
    assert module.__doc__ is not None
    assert ":meta private:" not in module.__doc__


def test_data_shim_omits_removed_vector_errors() -> None:
    import pinecone.data as data_shim

    removed = [
        "VectorDictionaryMissingKeysError",
        "VectorDictionaryExcessKeysError",
        "VectorTupleLengthError",
        "SparseValuesTypeError",
        "SparseValuesMissingKeysError",
        "SparseValuesDictionaryExpectedError",
        "MetadataDictionaryExpectedError",
    ]
    for name in removed:
        assert not hasattr(data_shim, name), (
            f"`from pinecone.data import {name}` should raise ImportError"
        )
