"""Unit tests for integrated (model-backed) index creation via Indexes.create()."""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import CONTROL_PLANE_API_VERSION
from pinecone._internal.http_client import HTTPClient
from pinecone.client.indexes import Indexes
from pinecone.errors.exceptions import PineconeError, ValidationError
from pinecone.models.enums import EmbedModel, Metric
from pinecone.models.indexes.index import IndexModel
from pinecone.models.indexes.specs import EmbedConfig, IntegratedSpec
from tests.factories import make_index_response

BASE_URL = "https://api.test.pinecone.io"


@pytest.fixture()
def http_client() -> HTTPClient:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    return HTTPClient(config, CONTROL_PLANE_API_VERSION)


@pytest.fixture()
def indexes(http_client: HTTPClient) -> Indexes:
    return Indexes(http=http_client)


def _integrated_response(**overrides: object) -> dict[str, object]:
    """Return a realistic response for an integrated index."""
    return make_index_response(
        name="my-integrated-index",
        embed={
            "model": "multilingual-e5-large",
            "metric": "cosine",
            "dimension": 1024,
            "field_map": {"text": "my_text_field"},
        },
        **overrides,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Success paths
# ---------------------------------------------------------------------------


@respx.mock
def test_create_integrated_index(indexes: Indexes) -> None:
    """Create with IntegratedSpec — verify correct wire format."""
    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(201, json=_integrated_response()),
    )

    result = indexes.create(
        name="my-integrated-index",
        spec=IntegratedSpec(
            cloud="aws",
            region="us-east-1",
            embed=EmbedConfig(
                model="multilingual-e5-large",
                field_map={"text": "my_text_field"},
            ),
        ),
        timeout=-1,
    )

    assert isinstance(result, IndexModel)
    assert result.name == "my-integrated-index"

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body["name"] == "my-integrated-index"
    assert body["cloud"] == "aws"
    assert body["region"] == "us-east-1"
    assert body["embed"]["model"] == "multilingual-e5-large"
    assert body["embed"]["field_map"] == {"text": "my_text_field"}
    # dimension and metric should NOT be in body (inferred by server)
    assert "dimension" not in body
    assert "metric" not in body
    # spec should NOT be in body (integrated uses flat structure)
    assert "spec" not in body


@respx.mock
def test_create_integrated_with_metric(indexes: Indexes) -> None:
    """Metric override in embed config is included in request."""
    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(201, json=_integrated_response()),
    )

    indexes.create(
        name="my-integrated-index",
        spec=IntegratedSpec(
            cloud="aws",
            region="us-east-1",
            embed=EmbedConfig(
                model="multilingual-e5-large",
                field_map={"text": "my_text_field"},
                metric="dotproduct",
            ),
        ),
        timeout=-1,
    )

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body["embed"]["metric"] == "dotproduct"


@respx.mock
def test_create_integrated_with_parameters(indexes: Indexes) -> None:
    """Read and write parameters are passed through."""
    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(201, json=_integrated_response()),
    )

    indexes.create(
        name="my-integrated-index",
        spec=IntegratedSpec(
            cloud="aws",
            region="us-east-1",
            embed=EmbedConfig(
                model="multilingual-e5-large",
                field_map={"text": "my_text_field"},
                read_parameters={"input_type": "query", "truncate": "NONE"},
                write_parameters={"input_type": "passage"},
            ),
        ),
        timeout=-1,
    )

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body["embed"]["read_parameters"] == {"input_type": "query", "truncate": "NONE"}
    assert body["embed"]["write_parameters"] == {"input_type": "passage"}


@respx.mock
def test_create_integrated_with_tags(indexes: Indexes) -> None:
    """Tags are included in the request body."""
    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(201, json=_integrated_response()),
    )

    indexes.create(
        name="my-integrated-index",
        spec=IntegratedSpec(
            cloud="aws",
            region="us-east-1",
            embed=EmbedConfig(
                model="multilingual-e5-large",
                field_map={"text": "my_text_field"},
            ),
        ),
        tags={"env": "test"},
        timeout=-1,
    )

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body["tags"] == {"env": "test"}


@respx.mock
def test_create_integrated_with_embed_model_enum(indexes: Indexes) -> None:
    """EmbedModel enum values are accepted for model parameter."""
    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(201, json=_integrated_response()),
    )

    indexes.create(
        name="my-integrated-index",
        spec=IntegratedSpec(
            cloud="aws",
            region="us-east-1",
            embed=EmbedConfig(
                model=EmbedModel.MULTILINGUAL_E5_LARGE,
                field_map={"text": "my_text_field"},
            ),
        ),
        timeout=-1,
    )

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body["embed"]["model"] == "multilingual-e5-large"


@respx.mock
def test_create_integrated_string_model_accepted(indexes: Indexes) -> None:
    """Plain strings also accepted for model parameter."""
    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(201, json=_integrated_response()),
    )

    indexes.create(
        name="my-integrated-index",
        spec=IntegratedSpec(
            cloud="aws",
            region="us-east-1",
            embed=EmbedConfig(
                model="pinecone-sparse-english-v0",
                field_map={"text": "my_text_field"},
            ),
        ),
        timeout=-1,
    )

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body["embed"]["model"] == "pinecone-sparse-english-v0"


# ---------------------------------------------------------------------------
# Validation error paths
# ---------------------------------------------------------------------------


def test_create_integrated_missing_cloud_raises(indexes: Indexes) -> None:
    """Empty cloud raises ValidationError (unified-index-0038)."""
    with pytest.raises(ValidationError, match="cloud"):
        indexes.create(
            name="my-integrated-index",
            spec=IntegratedSpec(
                cloud="",
                region="us-east-1",
                embed=EmbedConfig(
                    model="multilingual-e5-large",
                    field_map={"text": "my_text_field"},
                ),
            ),
        )


def test_create_integrated_missing_model_raises(indexes: Indexes) -> None:
    """Empty model raises ValidationError (unified-index-0039)."""
    with pytest.raises(ValidationError, match="model"):
        indexes.create(
            name="my-integrated-index",
            spec=IntegratedSpec(
                cloud="aws",
                region="us-east-1",
                embed=EmbedConfig(
                    model="",
                    field_map={"text": "my_text_field"},
                ),
            ),
        )


def test_create_integrated_missing_field_map_raises(indexes: Indexes) -> None:
    """Empty field_map raises ValidationError (unified-index-0040)."""
    with pytest.raises(ValidationError, match="field_map"):
        indexes.create(
            name="my-integrated-index",
            spec=IntegratedSpec(
                cloud="aws",
                region="us-east-1",
                embed=EmbedConfig(
                    model="multilingual-e5-large",
                    field_map={},
                ),
            ),
        )


def test_create_integrated_rejects_long_name(indexes: Indexes) -> None:
    """Name exceeding 45 characters raises ValidationError."""
    with pytest.raises(ValidationError, match="must not exceed 45 characters"):
        indexes.create(
            name="a" * 46,
            spec=IntegratedSpec(
                cloud="aws",
                region="us-east-1",
                embed=EmbedConfig(
                    model="multilingual-e5-large",
                    field_map={"text": "my_text_field"},
                ),
            ),
        )


def test_create_integrated_rejects_invalid_chars(indexes: Indexes) -> None:
    """Name with uppercase or special chars raises ValidationError."""
    with pytest.raises(ValidationError, match="lowercase letters, digits, and hyphens"):
        indexes.create(
            name="My_Invalid_Index!",
            spec=IntegratedSpec(
                cloud="aws",
                region="us-east-1",
                embed=EmbedConfig(
                    model="multilingual-e5-large",
                    field_map={"text": "my_text_field"},
                ),
            ),
        )


def test_create_integrated_missing_name_raises(indexes: Indexes) -> None:
    """Empty name raises ValidationError."""
    with pytest.raises(ValidationError, match="name"):
        indexes.create(
            name="",
            spec=IntegratedSpec(
                cloud="aws",
                region="us-east-1",
                embed=EmbedConfig(
                    model="multilingual-e5-large",
                    field_map={"text": "my_text_field"},
                ),
            ),
        )


# ---------------------------------------------------------------------------
# Polling
# ---------------------------------------------------------------------------


@respx.mock
def test_create_integrated_polls_until_ready(indexes: Indexes) -> None:
    """Integrated indexes use the same readiness polling (unified-index-0031)."""
    respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(
            201,
            json=_integrated_response(status={"ready": False, "state": "Initializing"}),
        ),
    )
    respx.get(f"{BASE_URL}/indexes/my-integrated-index").mock(
        side_effect=[
            httpx.Response(
                200,
                json=_integrated_response(status={"ready": False, "state": "Initializing"}),
            ),
            httpx.Response(
                200,
                json=_integrated_response(status={"ready": True, "state": "Ready"}),
            ),
        ]
    )

    with patch("pinecone.client.indexes.time.sleep"):
        result = indexes.create(
            name="my-integrated-index",
            spec=IntegratedSpec(
                cloud="aws",
                region="us-east-1",
                embed=EmbedConfig(
                    model="multilingual-e5-large",
                    field_map={"text": "my_text_field"},
                ),
            ),
            timeout=300,
        )

    assert result.status.ready is True


@respx.mock
def test_create_integrated_init_failed_raises(indexes: Indexes) -> None:
    """InitializationFailed raises immediately during polling."""
    respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(
            201,
            json=_integrated_response(status={"ready": False, "state": "Initializing"}),
        ),
    )
    respx.get(f"{BASE_URL}/indexes/my-integrated-index").mock(
        return_value=httpx.Response(
            200,
            json=_integrated_response(status={"ready": False, "state": "InitializationFailed"}),
        ),
    )

    with patch("pinecone.client.indexes.time.sleep"):
        with pytest.raises(PineconeError, match="InitializationFailed"):
            indexes.create(
                name="my-integrated-index",
                spec=IntegratedSpec(
                    cloud="aws",
                    region="us-east-1",
                    embed=EmbedConfig(
                        model="multilingual-e5-large",
                        field_map={"text": "my_text_field"},
                    ),
                ),
                timeout=300,
            )


@respx.mock
def test_create_integrated_no_polling_with_timeout_neg1(indexes: Indexes) -> None:
    """With timeout=-1, return immediately without polling."""
    respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(
            201,
            json=_integrated_response(status={"ready": False, "state": "Initializing"}),
        ),
    )

    result = indexes.create(
        name="my-integrated-index",
        spec=IntegratedSpec(
            cloud="aws",
            region="us-east-1",
            embed=EmbedConfig(
                model="multilingual-e5-large",
                field_map={"text": "my_text_field"},
            ),
        ),
        timeout=-1,
    )

    assert result.status.ready is False
    assert len(respx.calls) == 1  # only the POST


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


def test_embed_config_is_immutable() -> None:
    """EmbedConfig is frozen — attribute assignment raises AttributeError."""
    config = EmbedConfig(model="multilingual-e5-large", field_map={"text": "content"})
    with pytest.raises(AttributeError):
        config.model = "other"  # type: ignore[misc]


def test_integrated_spec_is_immutable() -> None:
    """IntegratedSpec is frozen — attribute assignment raises AttributeError."""
    spec = IntegratedSpec(
        cloud="aws",
        region="us-east-1",
        embed=EmbedConfig(
            model="multilingual-e5-large",
            field_map={"text": "content"},
        ),
    )
    with pytest.raises(AttributeError):
        spec.cloud = "gcp"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# EmbedConfig.to_dict() serialization
# ---------------------------------------------------------------------------


def test_embed_config_to_dict_basic() -> None:
    """to_dict serializes model, field_map, and defaults for read/write params."""
    config = EmbedConfig(model="multilingual-e5-large", field_map={"text": "content"})
    result = config.to_dict()
    assert result == {
        "model": "multilingual-e5-large",
        "field_map": {"text": "content"},
        "read_parameters": {},
        "write_parameters": {},
    }


def test_embed_config_to_dict_with_enum_metric() -> None:
    """Metric enum values are resolved to their string value in to_dict."""
    config = EmbedConfig(
        model="multilingual-e5-large",
        field_map={"text": "content"},
        metric=Metric.COSINE,
    )
    result = config.to_dict()
    assert result["metric"] == "cosine"
    assert not isinstance(result["metric"], Metric)


def test_embed_config_to_dict_with_read_write_params() -> None:
    """Explicit read/write parameters are included as-is."""
    config = EmbedConfig(
        model="multilingual-e5-large",
        field_map={"text": "content"},
        read_parameters={"k": 10},
        write_parameters={"batch": 32},
    )
    result = config.to_dict()
    assert result["read_parameters"] == {"k": 10}
    assert result["write_parameters"] == {"batch": 32}


def test_embed_config_to_dict_defaults_empty_params() -> None:
    """Omitted read/write parameters default to empty dicts."""
    config = EmbedConfig(model="multilingual-e5-large", field_map={"text": "content"})
    result = config.to_dict()
    assert result["read_parameters"] == {}
    assert result["write_parameters"] == {}
