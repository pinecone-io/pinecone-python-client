"""Unit tests for PreviewIndexModel."""

from __future__ import annotations

from unittest.mock import MagicMock

import msgspec
import pytest

from pinecone.preview.models.deployment import (
    PreviewByocDeployment,
    PreviewManagedDeployment,
    PreviewPodDeployment,
)
from pinecone.preview.models.indexes import PreviewIndexModel
from pinecone.preview.models.read_capacity import (
    PreviewReadCapacityOnDemandResponse,
    PreviewReadCapacityStatus,
)
from pinecone.preview.models.schema import (
    PreviewDenseVectorField,
    PreviewFullTextSearchConfig,
    PreviewSchema,
    PreviewSchemaField,
    PreviewStringField,
)
from pinecone.preview.models.status import PreviewIndexStatus

_FULL_PAYLOAD = b"""
{
    "name": "my-index",
    "host": "my-index-abc123.svc.pinecone.io",
    "status": {"ready": true, "state": "Ready"},
    "schema": {
        "fields": {
            "title": {"type": "string", "full_text_search": {"language": "en"}},
            "embedding": {"type": "dense_vector", "dimension": 768, "metric": "cosine"}
        }
    },
    "deployment": {
        "deployment_type": "managed",
        "environment": "aped-4627-b74a",
        "cloud": "aws",
        "region": "us-east-1"
    },
    "deletion_protection": "disabled",
    "read_capacity": {"mode": "OnDemand", "status": {"state": "Ready"}},
    "tags": {"env": "test", "team": "ml"}
}
"""

_MINIMAL_PAYLOAD = b"""
{
    "name": "bare-index",
    "host": "bare-index-xyz.svc.pinecone.io",
    "status": {"ready": false, "state": "Initializing"},
    "schema": {"fields": {}},
    "deployment": {
        "deployment_type": "managed",
        "environment": "aped-0001",
        "cloud": "gcp",
        "region": "us-central1"
    },
    "deletion_protection": "enabled"
}
"""


def _make_model(
    *,
    name: str = "test-index",
    host: str = "test-index.svc.pinecone.io",
    state: str = "Ready",
    ready: bool = True,
    n_fields: int = 0,
    deployment_type: str = "managed",
    deletion_protection: str = "disabled",
    read_capacity: PreviewReadCapacityOnDemandResponse | None = None,
    tags: dict[str, str] | None = None,
) -> PreviewIndexModel:
    fields: dict[str, PreviewSchemaField] = {}
    for i in range(n_fields):
        fields[f"field_{i}"] = PreviewStringField(full_text_search=PreviewFullTextSearchConfig())

    if deployment_type == "pod":
        deployment: PreviewManagedDeployment | PreviewPodDeployment = PreviewPodDeployment(
            environment="us-east1-gcp", pod_type="p1.x1"
        )
    else:
        deployment = PreviewManagedDeployment(
            environment="aped-4627-b74a", cloud="aws", region="us-east-1"
        )

    return PreviewIndexModel(
        name=name,
        host=host,
        status=PreviewIndexStatus(ready=ready, state=state),
        schema=PreviewSchema(fields=fields),
        deployment=deployment,
        deletion_protection=deletion_protection,
        read_capacity=read_capacity,
        tags=tags,
    )


def test_preview_index_model_decode_full() -> None:
    m = msgspec.json.decode(_FULL_PAYLOAD, type=PreviewIndexModel)
    assert isinstance(m, PreviewIndexModel)
    assert m.name == "my-index"
    assert m.host == "my-index-abc123.svc.pinecone.io"
    assert isinstance(m.status, PreviewIndexStatus)
    assert m.status.ready is True
    assert m.status.state == "Ready"
    assert isinstance(m.schema, PreviewSchema)
    assert len(m.schema.fields) == 2
    assert isinstance(m.schema.fields["title"], PreviewStringField)
    assert isinstance(m.schema.fields["embedding"], PreviewDenseVectorField)
    assert isinstance(m.deployment, PreviewManagedDeployment)
    assert m.deployment.cloud == "aws"
    assert m.deployment.region == "us-east-1"
    assert m.deletion_protection == "disabled"
    assert isinstance(m.read_capacity, PreviewReadCapacityOnDemandResponse)
    assert m.tags == {"env": "test", "team": "ml"}


def test_preview_index_model_decode_minimal() -> None:
    m = msgspec.json.decode(_MINIMAL_PAYLOAD, type=PreviewIndexModel)
    assert m.name == "bare-index"
    assert m.read_capacity is None
    assert m.tags is None


def test_preview_index_model_decode_ignores_private_host() -> None:
    payload = b"""
    {
        "name": "x",
        "host": "x.svc.pinecone.io",
        "status": {"ready": true, "state": "Ready"},
        "schema": {"fields": {}},
        "deployment": {
            "deployment_type": "managed",
            "environment": "e1",
            "cloud": "aws",
            "region": "us-east-1"
        },
        "deletion_protection": "disabled",
        "private_host": "p.svc.pinecone.io",
        "source_collection": "c1"
    }
    """
    m = msgspec.json.decode(payload, type=PreviewIndexModel)
    assert m.name == "x"


def test_preview_index_model_repr_single_line() -> None:
    m = _make_model()
    r = repr(m)
    assert r.startswith("PreviewIndexModel(")
    assert r.endswith(")")
    assert "\n" not in r
    assert "name=" in r
    assert "status=" in r
    assert "host=" in r


def test_preview_index_model_repr_includes_schema_fields() -> None:
    m = _make_model(n_fields=3)
    assert "schema_fields=3" in repr(m)


def test_preview_index_model_repr_omits_empty_tags() -> None:
    m = _make_model(tags=None)
    assert "tags=" not in repr(m)


def test_preview_index_model_repr_pretty_cycle() -> None:
    m = _make_model()
    p = MagicMock()
    m._repr_pretty_(p, cycle=True)
    p.text.assert_called_with("PreviewIndexModel(...)")


def test_preview_index_model_repr_html_contains_table() -> None:
    m = _make_model(n_fields=2)
    html = m._repr_html_()
    assert "PreviewIndexModel" in html
    assert "Name:" in html
    assert "Status:" in html
    assert "Deployment:" in html
    assert "Host:" in html
    assert "Schema fields:" in html


def test_preview_index_model_repr_html_includes_tags_when_present() -> None:
    m = _make_model(tags={"env": "prod"})
    html = m._repr_html_()
    assert "env=prod" in html


def test_preview_index_model_repr_html_pod_deployment_detail() -> None:
    m = _make_model(deployment_type="pod")
    html = m._repr_html_()
    assert "Pod" in html
    assert "(us-east1-gcp)" in html
    assert "Deployment:" in html


def test_preview_index_model_repr_html_byoc_deployment_detail() -> None:
    m = PreviewIndexModel(
        name="byoc-index",
        host="byoc-index.svc.pinecone.io",
        status=PreviewIndexStatus(ready=True, state="Ready"),
        schema=PreviewSchema(fields={}),
        deployment=PreviewByocDeployment(environment="e1", cloud="gcp", region="us-east1"),
        deletion_protection="disabled",
    )
    html = m._repr_html_()
    assert "Byoc" in html
    assert "(gcp/us-east1)" in html


@pytest.mark.xfail(
    reason="_repr_html_ uses hasattr which is True for None-valued struct fields, emitting '(None/None)'; see PreviewIndexModel._repr_html_"
)
def test_preview_index_model_repr_html_byoc_deployment_minimal() -> None:
    m = PreviewIndexModel(
        name="byoc-minimal",
        host="byoc-minimal.svc.pinecone.io",
        status=PreviewIndexStatus(ready=True, state="Ready"),
        schema=PreviewSchema(fields={}),
        deployment=PreviewByocDeployment(environment="e1"),
        deletion_protection="disabled",
    )
    html = m._repr_html_()
    assert "Byoc" in html
    assert "(None/None)" not in html


def test_preview_index_model_repr_html_includes_read_capacity_row() -> None:
    rc = PreviewReadCapacityOnDemandResponse(
        status=PreviewReadCapacityStatus(state="Ready"),
    )
    m = _make_model(read_capacity=rc)
    html = m._repr_html_()
    assert "Read capacity:" in html
    assert "OnDemand" in html


def test_preview_index_model_repr_html_omits_read_capacity_when_none() -> None:
    m = _make_model()
    html = m._repr_html_()
    assert "Read capacity:" not in html


def test_preview_index_model_repr_pretty_non_cycle_emits_core_fields() -> None:
    m = _make_model(n_fields=2)
    p = MagicMock()
    m._repr_pretty_(p, cycle=False)
    emitted = "".join(c.args[0] for c in p.text.call_args_list)
    assert "PreviewIndexModel(" in emitted
    assert "name='test-index'" in emitted
    assert "status='Ready'" in emitted
    assert "host='test-index.svc.pinecone.io'" in emitted
    assert "deletion_protection='disabled'" in emitted
    assert "schema=Schema(fields=2 fields)" in emitted
    assert p.breakable.call_count >= 1


def test_preview_index_model_repr_pretty_non_cycle_includes_read_capacity_when_present() -> None:
    rc = PreviewReadCapacityOnDemandResponse(status=PreviewReadCapacityStatus(state="Ready"))
    m = _make_model(read_capacity=rc)
    p = MagicMock()
    m._repr_pretty_(p, cycle=False)
    emitted = "".join(c.args[0] for c in p.text.call_args_list)
    assert "read_capacity=" in emitted


def test_preview_index_model_repr_pretty_non_cycle_includes_tags_when_present() -> None:
    m = _make_model(tags={"env": "prod"})
    p = MagicMock()
    m._repr_pretty_(p, cycle=False)
    emitted = "".join(c.args[0] for c in p.text.call_args_list)
    assert "tags=" in emitted
    assert "'env'" in emitted


def test_preview_index_model_repr_pretty_non_cycle_omits_optional_fields_when_none() -> None:
    m = _make_model()
    p = MagicMock()
    m._repr_pretty_(p, cycle=False)
    emitted = "".join(c.args[0] for c in p.text.call_args_list)
    assert "read_capacity=" not in emitted
    assert "tags=" not in emitted
