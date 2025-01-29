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
