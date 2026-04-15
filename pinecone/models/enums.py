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


class RerankModel(str, Enum):
    """Known reranking models."""

    BGE_RERANKER_V2_M3 = "bge-reranker-v2-m3"
    COHERE_RERANK_3_5 = "cohere-rerank-3.5"
    PINECONE_RERANK_V0 = "pinecone-rerank-v0"


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


class AwsRegion(str, Enum):
    """AWS regions supported for serverless indexes."""

    US_EAST_1 = "us-east-1"
    US_WEST_2 = "us-west-2"
    EU_WEST_1 = "eu-west-1"


class GcpRegion(str, Enum):
    """GCP regions supported for serverless indexes."""

    US_CENTRAL1 = "us-central1"
    EUROPE_WEST4 = "europe-west4"


class AzureRegion(str, Enum):
    """Azure regions supported for serverless indexes."""

    EASTUS2 = "eastus2"


class PodIndexEnvironment(str, Enum):
    """Deployment environments for pod-based indexes."""

    US_WEST1_GCP = "us-west1-gcp"
    US_CENTRAL1_GCP = "us-central1-gcp"
    US_WEST4_GCP = "us-west4-gcp"
    US_EAST4_GCP = "us-east4-gcp"
    NORTHAMERICA_NORTHEAST1_GCP = "northamerica-northeast1-gcp"
    ASIA_NORTHEAST1_GCP = "asia-northeast1-gcp"
    ASIA_SOUTHEAST1_GCP = "asia-southeast1-gcp"
    US_EAST1_GCP = "us-east1-gcp"
    EU_WEST1_GCP = "eu-west1-gcp"
    EU_WEST4_GCP = "eu-west4-gcp"
    US_EAST1_AWS = "us-east-1-aws"
    EASTUS_AZURE = "eastus-azure"
