"""Tests for Pinecone.__repr__ masking and deprecated create-index delegates."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from pinecone import Pinecone
from pinecone.inference.models.index_embed import IndexEmbed
from pinecone.models.enums import CloudProvider
from pinecone.models.indexes.specs import EmbedConfig, IntegratedSpec, ServerlessSpec


class TestPineconeRepr:
    def test_pinecone_repr_masks_full_api_key(self) -> None:
        pc = Pinecone(api_key="pcsk_secret_12345")
        result = repr(pc)
        assert "pcsk_secret_12345" not in result
        assert "...2345" in result
        assert "host=" in result

    def test_pinecone_repr_masks_short_api_key(self) -> None:
        pc = Pinecone(api_key="ab")
        result = repr(pc)
        assert "api_key='***'" in result
        assert "ab" not in result

    def test_pinecone_repr_exactly_four_char_key_shows_last_four(self) -> None:
        pc = Pinecone(api_key="wxyz")
        result = repr(pc)
        assert "...wxyz" in result


def _make_pc_with_mock_indexes() -> tuple[Pinecone, MagicMock]:
    pc = Pinecone(api_key="test-key")
    mock_indexes = MagicMock()
    mock_indexes.create.return_value = MagicMock()
    pc._indexes = mock_indexes
    return pc, mock_indexes


def test_pinecone_create_index_delegate_emits_deprecation_warning_and_forwards() -> None:
    pc, mock_indexes = _make_pc_with_mock_indexes()
    with pytest.warns(DeprecationWarning, match=r"create_index\(\) is deprecated"):
        pc.create_index(
            name="x",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            dimension=4,
        )
    mock_indexes.create.assert_called_once()
    _, kwargs = mock_indexes.create.call_args
    assert kwargs["metric"] == "cosine"
    assert kwargs["vector_type"] == "dense"
    assert kwargs["deletion_protection"] == "disabled"


def test_pinecone_create_index_delegate_with_explicit_metric_and_vector_type_forwards_verbatim() -> (
    None
):
    pc, mock_indexes = _make_pc_with_mock_indexes()
    with pytest.warns(DeprecationWarning, match="create_index"):
        pc.create_index(
            name="x",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            dimension=4,
            metric="euclidean",
            vector_type="sparse",
        )
    _, kwargs = mock_indexes.create.call_args
    assert kwargs["metric"] == "euclidean"
    assert kwargs["vector_type"] == "sparse"


def test_pinecone_create_index_delegate_with_none_deletion_protection_defaults_to_disabled() -> (
    None
):
    pc, mock_indexes = _make_pc_with_mock_indexes()
    with pytest.warns(DeprecationWarning, match="create_index"):
        pc.create_index(
            name="x",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            deletion_protection=None,
        )
    _, kwargs = mock_indexes.create.call_args
    assert kwargs["deletion_protection"] == "disabled"


def test_pinecone_create_index_for_model_delegate_with_index_embed_converts_to_embed_config() -> (
    None
):
    pc, mock_indexes = _make_pc_with_mock_indexes()
    index_embed = IndexEmbed(
        model="multilingual-e5-large",
        field_map={"text": "my_field"},
    )
    with pytest.warns(DeprecationWarning, match="create_index_for_model"):
        pc.create_index_for_model(
            name="my-index",
            cloud=CloudProvider.AWS,
            region="us-east-1",
            embed=index_embed,
        )
    _, kwargs = mock_indexes.create.call_args
    spec = kwargs["spec"]
    assert isinstance(spec, IntegratedSpec)
    assert isinstance(spec.embed, EmbedConfig)
    assert spec.embed.model == "multilingual-e5-large"
    assert spec.embed.field_map == {"text": "my_field"}
    assert spec.cloud == "aws"


def test_pinecone_create_index_for_model_delegate_with_embed_config_passes_through() -> None:
    pc, mock_indexes = _make_pc_with_mock_indexes()
    embed_config = EmbedConfig(
        model="multilingual-e5-large",
        field_map={"text": "my_field"},
    )
    with pytest.warns(DeprecationWarning, match="create_index_for_model"):
        pc.create_index_for_model(
            name="my-index",
            cloud=CloudProvider.AWS,
            region="us-east-1",
            embed=embed_config,
        )
    _, kwargs = mock_indexes.create.call_args
    spec = kwargs["spec"]
    assert isinstance(spec, IntegratedSpec)
    assert spec.embed is embed_config
    assert spec.cloud == "aws"


def test_pinecone_create_index_for_model_delegate_with_dict_constructs_embed_config() -> None:
    pc, mock_indexes = _make_pc_with_mock_indexes()
    with pytest.warns(DeprecationWarning, match="create_index_for_model"):
        pc.create_index_for_model(
            name="my-index",
            cloud=CloudProvider.AWS,
            region="us-east-1",
            embed={"model": "m", "field_map": {"text": "a"}},
        )
    _, kwargs = mock_indexes.create.call_args
    spec = kwargs["spec"]
    assert isinstance(spec, IntegratedSpec)
    assert isinstance(spec.embed, EmbedConfig)
    assert spec.embed.model == "m"
    assert spec.embed.field_map == {"text": "a"}
    assert spec.cloud == "aws"
