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


class TestAlignmentMetricsProxyCompat:
    def _make_fake_result(self) -> object:

        from pinecone.models.assistant.chat import ChatUsage
        from pinecone.models.assistant.evaluation import (
            AlignmentResult,
            AlignmentScores,
            EntailmentResult,
        )

        scores = AlignmentScores(correctness=0.9, completeness=0.8, alignment=0.85)
        facts = [
            EntailmentResult(fact="The sky is blue.", entailment="entailed"),
            EntailmentResult(fact="Water is wet.", entailment="neutral"),
        ]
        usage = ChatUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        return AlignmentResult(scores=scores, facts=facts, usage=usage)

    def _make_proxy(self) -> object:
        from unittest.mock import MagicMock

        from pinecone.client._assistants_legacy import _AlignmentMetricsProxy

        fake_result = self._make_fake_result()
        mock_assistants = MagicMock()
        mock_assistants.evaluate_alignment.return_value = fake_result
        return _AlignmentMetricsProxy(mock_assistants)

    def test_evaluation_metrics_alignment_legacy_shape(self) -> None:
        proxy = self._make_proxy()
        result = proxy.alignment(  # type: ignore[union-attr]
            question="q", answer="a", ground_truth_answer="g"
        )
        assert result.metrics.alignment == pytest.approx(0.85)
        assert result.metrics.correctness == pytest.approx(0.9)
        assert result.metrics.completeness == pytest.approx(0.8)
        assert result.reasoning.evaluated_facts[0].fact.content == "The sky is blue."
        assert result.reasoning.evaluated_facts[0].entailment == "entailed"
        assert result.reasoning.evaluated_facts[1].fact.content == "Water is wet."

    def test_evaluation_metrics_alignment_new_shape(self) -> None:
        proxy = self._make_proxy()
        result = proxy.alignment(  # type: ignore[union-attr]
            question="q", answer="a", ground_truth_answer="g"
        )
        assert result.scores.alignment == pytest.approx(0.85)
        assert isinstance(result.facts[0].fact, str)
        assert result.facts[0].fact == "The sky is blue."

    def test_evaluation_metrics_alignment_usage_unchanged(self) -> None:
        proxy = self._make_proxy()
        result = proxy.alignment(  # type: ignore[union-attr]
            question="q", answer="a", ground_truth_answer="g"
        )
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 20
        assert result.usage.total_tokens == 30


class TestInferenceModuleExports:
    def test_inference_module_exports_embed_model(self) -> None:
        from pinecone.inference import EmbedModel

        assert EmbedModel is not None

    def test_inference_module_exports_rerank_model(self) -> None:
        from pinecone.inference import RerankModel

        assert RerankModel is not None

    def test_inference_module_exports_model_info(self) -> None:
        from pinecone.inference import ModelInfo

        assert ModelInfo is not None

    def test_inference_module_exports_model_info_list(self) -> None:
        from pinecone.inference import ModelInfoList

        assert ModelInfoList is not None

    def test_inference_module_exports_embeddings_list(self) -> None:
        from pinecone.inference import EmbeddingsList

        assert EmbeddingsList is not None

    def test_inference_module_exports_rerank_result(self) -> None:
        from pinecone.inference import RerankResult

        assert RerankResult is not None

    def test_inference_module_all_matches(self) -> None:
        import pinecone.inference as inf

        expected = {
            "AsyncioInference",
            "EmbedModel",
            "EmbeddingsList",
            "Inference",
            "ModelInfo",
            "ModelInfoList",
            "RerankModel",
            "RerankResult",
        }
        assert set(inf.__all__) == expected
        for name in expected:
            assert hasattr(inf, name), f"pinecone.inference missing attribute {name!r}"


class TestDbControlModuleExports:
    def test_db_control_top_exports_cloud_provider(self) -> None:
        from pinecone.db_control import CloudProvider

        assert CloudProvider is not None

    def test_db_control_top_exports_serverless_spec(self) -> None:
        from pinecone.db_control import ServerlessSpec

        assert ServerlessSpec is not None

    def test_db_control_top_exports_index_model(self) -> None:
        from pinecone.db_control import IndexModel

        assert IndexModel is not None

    def test_db_control_enums_exports_cloud_provider(self) -> None:
        from pinecone.db_control.enums import CloudProvider

        assert CloudProvider is not None

    def test_db_control_enums_exports_metric(self) -> None:
        from pinecone.db_control.enums import Metric

        assert Metric is not None

    def test_db_control_enums_exports_pod_type(self) -> None:
        from pinecone.db_control.enums import PodType

        assert PodType is not None

    def test_db_control_models_exports_serverless_spec(self) -> None:
        from pinecone.db_control.models import ServerlessSpec

        assert ServerlessSpec is not None

    def test_db_control_models_exports_pod_spec(self) -> None:
        from pinecone.db_control.models import PodSpec

        assert PodSpec is not None

    def test_db_control_models_exports_index_model(self) -> None:
        from pinecone.db_control.models import IndexModel

        assert IndexModel is not None
