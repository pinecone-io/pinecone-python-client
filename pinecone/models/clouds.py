from enum import Enum


class CloudProvider(Enum):
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"


class AwsRegion(Enum):
    US_EAST_1 = "us-east-1"
    US_WEST_2 = "us-west-2"
    EU_WEST_1 = "eu-west-1"


class GcpRegion(Enum):
    US_CENTRAL1 = "us-central1"
    EUROPE_WEST4 = "europe-west4"


class AzureRegion(Enum):
    EAST_US = "eastus2"


class PodIndexEnvironment(Enum):
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
