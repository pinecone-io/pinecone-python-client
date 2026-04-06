"""Enumeration models for Pinecone SDK configuration values."""

from __future__ import annotations

from enum import Enum


class CloudProvider(str, Enum):
    """Supported cloud providers for Pinecone indexes."""

    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"


class Metric(str, Enum):
    """Supported similarity metrics for vector search."""

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOTPRODUCT = "dotproduct"


class DeletionProtection(str, Enum):
    """Deletion protection setting for indexes."""

    ENABLED = "enabled"
    DISABLED = "disabled"


class VectorType(str, Enum):
    """Supported vector types."""

    DENSE = "dense"
    SPARSE = "sparse"


class EmbedModel(str, Enum):
    """Known embedding models for integrated indexes."""

    MULTILINGUAL_E5_LARGE = "multilingual-e5-large"
    PINECONE_SPARSE_ENGLISH_V0 = "pinecone-sparse-english-v0"


class PodType(str, Enum):
    """Supported pod type and size combinations."""

    P1_X1 = "p1.x1"
    P1_X2 = "p1.x2"
    P1_X4 = "p1.x4"
    P1_X8 = "p1.x8"
    S1_X1 = "s1.x1"
    S1_X2 = "s1.x2"
    S1_X4 = "s1.x4"
    S1_X8 = "s1.x8"
    P2_X1 = "p2.x1"
    P2_X2 = "p2.x2"
    P2_X4 = "p2.x4"
    P2_X8 = "p2.x8"
