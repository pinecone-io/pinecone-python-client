"""Unit tests for preview read-capacity response models."""

from __future__ import annotations

import msgspec

from pinecone.preview.models.read_capacity import (
    PreviewReadCapacity,
    PreviewReadCapacityDedicatedResponse,
    PreviewReadCapacityOnDemandResponse,
    PreviewReadCapacityStatus,
)


def test_on_demand_response_decode() -> None:
    raw = b'{"mode": "OnDemand", "status": {"state": "Ready", "current_shards": null, "current_replicas": null}}'
    result = msgspec.json.decode(raw, type=PreviewReadCapacity)
    assert isinstance(result, PreviewReadCapacityOnDemandResponse)
    assert result.status.state == "Ready"
    assert result.status.current_shards is None
    assert result.status.current_replicas is None


def test_dedicated_response_decode_manual() -> None:
    raw = b'{"mode": "Dedicated", "dedicated": {"node_type": "b1", "scaling": "Manual", "manual": {"shards": 2, "replicas": 1}, "auto": null}, "status": {"state": "Ready", "current_shards": null, "current_replicas": null}}'
    result = msgspec.json.decode(raw, type=PreviewReadCapacity)
    assert isinstance(result, PreviewReadCapacityDedicatedResponse)
    assert result.dedicated.node_type == "b1"
    assert result.dedicated.manual is not None
    assert result.dedicated.manual.shards == 2
    assert result.dedicated.manual.replicas == 1
    assert result.dedicated.auto is None


def test_dedicated_response_decode_auto() -> None:
    raw = b'{"mode": "Dedicated", "dedicated": {"node_type": "b1", "scaling": "Auto", "manual": null, "auto": {"target_utilization": 0.7}}, "status": {"state": "Ready", "current_shards": null, "current_replicas": null}}'
    result = msgspec.json.decode(raw, type=PreviewReadCapacity)
    assert isinstance(result, PreviewReadCapacityDedicatedResponse)
    assert result.dedicated.scaling == "Auto"
    assert result.dedicated.manual is None
    assert result.dedicated.auto == {"target_utilization": 0.7}


def test_read_capacity_union_decode_dispatches_on_mode() -> None:
    on_demand_raw = b'{"mode": "OnDemand", "status": {"state": "Ready", "current_shards": null, "current_replicas": null}}'
    dedicated_raw = b'{"mode": "Dedicated", "dedicated": {"node_type": "b1", "scaling": "Manual", "manual": {"shards": 1, "replicas": 1}, "auto": null}, "status": {"state": "Ready", "current_shards": null, "current_replicas": null}}'

    on_demand = msgspec.json.decode(on_demand_raw, type=PreviewReadCapacity)
    dedicated = msgspec.json.decode(dedicated_raw, type=PreviewReadCapacity)

    assert isinstance(on_demand, PreviewReadCapacityOnDemandResponse)
    assert isinstance(dedicated, PreviewReadCapacityDedicatedResponse)


def test_status_decode_with_scaling_states() -> None:
    initializing_raw = (
        b'{"state": "Initializing", "current_shards": null, "current_replicas": null}'
    )
    status = msgspec.json.decode(initializing_raw, type=PreviewReadCapacityStatus)
    assert status.state == "Initializing"
    assert status.current_shards is None
    assert status.current_replicas is None

    migrating_raw = b'{"state": "Migrating", "current_shards": 2, "current_replicas": 1}'
    status2 = msgspec.json.decode(migrating_raw, type=PreviewReadCapacityStatus)
    assert status2.state == "Migrating"
    assert status2.current_shards == 2
    assert status2.current_replicas == 1
