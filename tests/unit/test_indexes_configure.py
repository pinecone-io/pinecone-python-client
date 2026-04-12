"""Unit tests for Indexes.configure() — update an existing index."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import CONTROL_PLANE_API_VERSION
from pinecone._internal.http_client import HTTPClient
from pinecone.client.indexes import Indexes
from pinecone.errors.exceptions import ValidationError
from pinecone.models.enums import DeletionProtection
from tests.factories import make_index_response

BASE_URL = "https://api.test.pinecone.io"


@pytest.fixture
def http_client() -> HTTPClient:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    return HTTPClient(config, CONTROL_PLANE_API_VERSION)


@pytest.fixture
def indexes(http_client: HTTPClient) -> Indexes:
    return Indexes(http=http_client)


# ---------------------------------------------------------------------------
# Success paths
# ---------------------------------------------------------------------------


@respx.mock
def test_configure_replicas_only(indexes: Indexes) -> None:
    """PATCH body has spec.pod.replicas only; returns None."""
    route = respx.patch(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(202, json=make_index_response()),
    )

    indexes.configure("test-index", replicas=4)

    payload = _request_json(route)
    assert payload == {"spec": {"pod": {"replicas": 4}}}


@respx.mock
def test_configure_pod_type_only(indexes: Indexes) -> None:
    """PATCH body has spec.pod.pod_type only."""
    route = respx.patch(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(202, json=make_index_response()),
    )

    indexes.configure("test-index", pod_type="p1.x2")

    payload = _request_json(route)
    assert payload == {"spec": {"pod": {"pod_type": "p1.x2"}}}


@respx.mock
def test_configure_deletion_protection(indexes: Indexes) -> None:
    """PATCH body includes deletion_protection."""
    route = respx.patch(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(202, json=make_index_response()),
    )

    indexes.configure("test-index", deletion_protection="enabled")

    payload = _request_json(route)
    assert payload == {"deletion_protection": "enabled"}


@respx.mock
def test_configure_tags_with_merging(indexes: Indexes) -> None:
    """Tags are merged with existing tags from describe."""
    # describe returns existing tags
    respx.get(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(
            200,
            json=make_index_response(tags={"existing": "val", "keep": "me"}),
        ),
    )
    patch_route = respx.patch(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(202, json=make_index_response()),
    )

    indexes.configure("test-index", tags={"new_tag": "hello", "existing": "overwritten"})

    payload = _request_json(patch_route)
    assert payload == {
        "tags": {"existing": "overwritten", "keep": "me", "new_tag": "hello"},
    }


@respx.mock
def test_configure_tag_removal_via_empty_string(indexes: Indexes) -> None:
    """Setting a tag value to empty string passes through for removal."""
    respx.get(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(
            200,
            json=make_index_response(tags={"remove_me": "old_val", "keep": "val"}),
        ),
    )
    patch_route = respx.patch(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(202, json=make_index_response()),
    )

    indexes.configure("test-index", tags={"remove_me": ""})

    payload = _request_json(patch_route)
    assert payload["tags"]["remove_me"] == ""
    assert payload["tags"]["keep"] == "val"


@respx.mock
def test_configure_no_deletion_protection_when_not_specified(indexes: Indexes) -> None:
    """When deletion_protection is not specified, it's absent from the body."""
    route = respx.patch(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(202, json=make_index_response()),
    )

    indexes.configure("test-index", replicas=2)

    payload = _request_json(route)
    assert "deletion_protection" not in payload


@respx.mock
def test_configure_returns_none(indexes: Indexes) -> None:
    """configure() always returns None regardless of API response."""
    respx.patch(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(202, json=make_index_response()),
    )

    assert indexes.configure("test-index", replicas=1) is None  # type: ignore[func-returns-value]


@respx.mock
def test_configure_multiple_fields(indexes: Indexes) -> None:
    """Can set replicas, pod_type, and deletion_protection together."""
    respx.get(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(200, json=make_index_response(tags={})),
    )
    route = respx.patch(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(202, json=make_index_response()),
    )

    indexes.configure(
        "test-index",
        replicas=4,
        pod_type="p1.x2",
        deletion_protection="disabled",
        tags={"env": "prod"},
    )

    payload = _request_json(route)
    assert payload["spec"]["pod"]["replicas"] == 4
    assert payload["spec"]["pod"]["pod_type"] == "p1.x2"
    assert payload["deletion_protection"] == "disabled"
    assert payload["tags"] == {"env": "prod"}


# ---------------------------------------------------------------------------
# Enum handling
# ---------------------------------------------------------------------------


@respx.mock
def test_configure_deletion_protection_enum(indexes: Indexes) -> None:
    """Accept DeletionProtection enum for deletion_protection param."""
    route = respx.patch(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(202, json=make_index_response()),
    )

    indexes.configure("test-index", deletion_protection=DeletionProtection.ENABLED)

    payload = _request_json(route)
    assert payload == {"deletion_protection": "enabled"}


@respx.mock
def test_configure_deletion_protection_disabled_enum(indexes: Indexes) -> None:
    """Accept DeletionProtection.DISABLED enum value."""
    route = respx.patch(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(202, json=make_index_response()),
    )

    indexes.configure("test-index", deletion_protection=DeletionProtection.DISABLED)

    payload = _request_json(route)
    assert payload == {"deletion_protection": "disabled"}


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_configure_empty_name_raises(indexes: Indexes) -> None:
    """Empty name raises ValidationError before any HTTP call."""
    with pytest.raises(ValidationError):
        indexes.configure("")


def test_configure_invalid_deletion_protection_raises(indexes: Indexes) -> None:
    with pytest.raises(ValidationError, match="deletion_protection"):
        indexes.configure("test-index", deletion_protection="maybe")


# ---------------------------------------------------------------------------
# BYOC read_capacity
# ---------------------------------------------------------------------------


@respx.mock
def test_configure_byoc_read_capacity_on_demand(indexes: Indexes) -> None:
    """PATCH body has spec.byoc.read_capacity with OnDemand mode."""
    route = respx.patch(f"{BASE_URL}/indexes/my-idx").mock(
        return_value=httpx.Response(202, json=make_index_response()),
    )

    indexes.configure("my-idx", read_capacity={"mode": "OnDemand"})

    payload = _request_json(route)
    assert payload == {"spec": {"byoc": {"read_capacity": {"mode": "OnDemand"}}}}


@respx.mock
def test_configure_byoc_read_capacity_dedicated(indexes: Indexes) -> None:
    """PATCH body has full dedicated read_capacity structure."""
    route = respx.patch(f"{BASE_URL}/indexes/my-idx").mock(
        return_value=httpx.Response(202, json=make_index_response()),
    )

    indexes.configure(
        "my-idx",
        read_capacity={
            "mode": "Dedicated",
            "dedicated": {
                "node_type": "t1",
                "scaling": "Manual",
                "manual": {"replicas": 2, "shards": 1},
            },
        },
    )

    payload = _request_json(route)
    assert payload == {
        "spec": {
            "byoc": {
                "read_capacity": {
                    "mode": "Dedicated",
                    "dedicated": {
                        "node_type": "t1",
                        "scaling": "Manual",
                        "manual": {"replicas": 2, "shards": 1},
                    },
                }
            }
        }
    }


def test_configure_byoc_read_capacity_dedicated_missing_node_type(
    indexes: Indexes,
) -> None:
    """Missing node_type in dedicated config raises ValidationError."""
    with pytest.raises(ValidationError, match="node_type"):
        indexes.configure(
            "my-idx",
            read_capacity={"mode": "Dedicated", "dedicated": {"scaling": "Manual"}},
        )


def test_configure_byoc_read_capacity_dedicated_missing_scaling(
    indexes: Indexes,
) -> None:
    """Missing scaling in dedicated config raises ValidationError."""
    with pytest.raises(ValidationError, match="scaling"):
        indexes.configure(
            "my-idx",
            read_capacity={"mode": "Dedicated", "dedicated": {"node_type": "t1"}},
        )


def test_configure_byoc_read_capacity_missing_mode(indexes: Indexes) -> None:
    """Missing mode key raises ValidationError."""
    with pytest.raises(ValidationError, match="mode"):
        indexes.configure(
            "my-idx",
            read_capacity={"dedicated": {"node_type": "t1"}},
        )


def test_configure_rejects_pod_fields_with_read_capacity(indexes: Indexes) -> None:
    """Passing both pod fields and read_capacity raises ValidationError."""
    with pytest.raises(ValidationError, match=r"pod.*read_capacity"):
        indexes.configure(
            "my-idx",
            replicas=2,
            read_capacity={"mode": "OnDemand"},
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _request_json(route: respx.Route) -> dict[str, Any]:
    """Extract the JSON body from the last request on a route."""
    import orjson

    request = route.calls.last.request
    return orjson.loads(request.content)  # type: ignore[no-any-return]
