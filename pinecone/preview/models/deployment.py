"""Preview deployment configuration response models (2026-01.alpha API)."""

from __future__ import annotations

from msgspec import Struct

__all__ = [
    "PreviewByocDeployment",
    "PreviewDeployment",
    "PreviewManagedDeployment",
    "PreviewPodDeployment",
]


class PreviewManagedDeployment(Struct, tag="managed", tag_field="deployment_type", kw_only=True):
    """Managed deployment configuration.

    .. admonition:: Preview
       :class: warning

       Uses Pinecone API version ``2026-01.alpha``.
       Preview surface is not covered by SemVer — signatures and behavior
       may change in any minor SDK release. Pin your SDK version when
       relying on preview features.

    Attributes:
        environment: Environment identifier.
        cloud: Cloud provider (e.g., ``"aws"``, ``"gcp"``, ``"azure"``).
        region: Cloud region (e.g., ``"us-east-1"``).

    Note:
        The ``deployment_type`` field is automatically set to ``"managed"``
        by msgspec's tagged union system.
    """

    environment: str
    cloud: str
    region: str


class PreviewPodDeployment(Struct, tag="pod", tag_field="deployment_type", kw_only=True):
    """Pod-based deployment configuration.

    .. admonition:: Preview
       :class: warning

       Uses Pinecone API version ``2026-01.alpha``.
       Preview surface is not covered by SemVer — signatures and behavior
       may change in any minor SDK release. Pin your SDK version when
       relying on preview features.

    Attributes:
        environment: Environment identifier.
        pod_type: Pod type (e.g., ``"p1.x1"``, ``"p2.x1"``).
        pods: Number of pods (optional).
        replicas: Number of replicas (optional).
        shards: Number of shards (optional).

    Note:
        The ``deployment_type`` field is automatically set to ``"pod"``
        by msgspec's tagged union system.
    """

    environment: str
    pod_type: str
    pods: int | None = None
    replicas: int | None = None
    shards: int | None = None


class PreviewByocDeployment(Struct, tag="byoc", tag_field="deployment_type", kw_only=True):
    """Bring Your Own Cloud (BYOC) deployment configuration.

    .. admonition:: Preview
       :class: warning

       Uses Pinecone API version ``2026-01.alpha``.
       Preview surface is not covered by SemVer — signatures and behavior
       may change in any minor SDK release. Pin your SDK version when
       relying on preview features.

    Attributes:
        environment: BYOC environment identifier.
        cloud: Cloud provider (optional).
        region: Cloud region (optional).

    Note:
        The ``deployment_type`` field is automatically set to ``"byoc"``
        by msgspec's tagged union system.
    """

    environment: str
    cloud: str | None = None
    region: str | None = None


PreviewDeployment = PreviewManagedDeployment | PreviewPodDeployment | PreviewByocDeployment
