"""Preview read-capacity response models (2026-01.alpha API)."""

from __future__ import annotations

from typing import Any

from msgspec import Struct

__all__ = [
    "PreviewReadCapacity",
    "PreviewReadCapacityDedicatedInner",
    "PreviewReadCapacityDedicatedResponse",
    "PreviewReadCapacityManualScaling",
    "PreviewReadCapacityOnDemandResponse",
    "PreviewReadCapacityStatus",
]


class PreviewReadCapacityManualScaling(Struct, kw_only=True):
    """Manual scaling configuration for dedicated read capacity.

    .. admonition:: Preview
       :class: warning

       Uses Pinecone API version ``2026-01.alpha``.
       Preview surface is not covered by SemVer — signatures and behavior
       may change in any minor SDK release. Pin your SDK version when
       relying on preview features.

    Attributes:
        shards: Number of shards for horizontal scaling.
        replicas: Number of replicas for high availability.
    """

    shards: int
    replicas: int


class PreviewReadCapacityStatus(Struct, kw_only=True):
    """Read capacity provisioning status.

    .. admonition:: Preview
       :class: warning

       Uses Pinecone API version ``2026-01.alpha``.
       Preview surface is not covered by SemVer — signatures and behavior
       may change in any minor SDK release. Pin your SDK version when
       relying on preview features.

    Attributes:
        state: Current provisioning state. Observed values: ``"Ready"``,
            ``"Initializing"``, ``"Migrating"``.
        current_shards: Current number of active shards (may be ``None``
            during provisioning).
        current_replicas: Current number of active replicas (may be ``None``
            during provisioning).
    """

    state: str
    current_shards: int | None = None
    current_replicas: int | None = None


class PreviewReadCapacityDedicatedInner(Struct, kw_only=True):
    """Inner dedicated capacity configuration in API responses.

    .. admonition:: Preview
       :class: warning

       Uses Pinecone API version ``2026-01.alpha``.
       Preview surface is not covered by SemVer — signatures and behavior
       may change in any minor SDK release. Pin your SDK version when
       relying on preview features.

    Attributes:
        node_type: Node type (e.g., ``"b1"``).
        scaling: Scaling strategy — ``"Manual"`` or ``"Auto"``.
        manual: Manual scaling configuration, present when ``scaling="Manual"``.
        auto: Auto-scaling configuration dict, present when ``scaling="Auto"``.
    """

    node_type: str
    scaling: str
    manual: PreviewReadCapacityManualScaling | None = None
    auto: dict[str, Any] | None = None


class PreviewReadCapacityOnDemandResponse(
    Struct, tag="OnDemand", tag_field="mode", kw_only=True
):
    """On-demand read capacity in API responses.

    .. admonition:: Preview
       :class: warning

       Uses Pinecone API version ``2026-01.alpha``.
       Preview surface is not covered by SemVer — signatures and behavior
       may change in any minor SDK release. Pin your SDK version when
       relying on preview features.

    Attributes:
        status: Current provisioning status.

    Note:
        The ``mode`` field is automatically set to ``"OnDemand"`` by
        msgspec's tagged-union system.
    """

    status: PreviewReadCapacityStatus


class PreviewReadCapacityDedicatedResponse(
    Struct, tag="Dedicated", tag_field="mode", kw_only=True
):
    """Dedicated read capacity in API responses.

    .. admonition:: Preview
       :class: warning

       Uses Pinecone API version ``2026-01.alpha``.
       Preview surface is not covered by SemVer — signatures and behavior
       may change in any minor SDK release. Pin your SDK version when
       relying on preview features.

    Attributes:
        dedicated: Dedicated capacity configuration details.
        status: Current provisioning status.

    Note:
        The ``mode`` field is automatically set to ``"Dedicated"`` by
        msgspec's tagged-union system.
    """

    dedicated: PreviewReadCapacityDedicatedInner
    status: PreviewReadCapacityStatus


# Union of all read-capacity response variants, dispatched on the ``mode`` field.
PreviewReadCapacity = PreviewReadCapacityOnDemandResponse | PreviewReadCapacityDedicatedResponse
