from enum import Enum


class CloudProvider(Enum):
    """Cloud providers available for use with Pinecone serverless indexes"""

    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"


class AwsRegion(Enum):
    """AWS (Amazon Web Services) regions available for use with Pinecone serverless indexes"""

    US_EAST_1 = "us-east-1"
    US_WEST_2 = "us-west-2"
    EU_WEST_1 = "eu-west-1"


class GcpRegion(Enum):
    """GCP (Google Cloud Platform) regions available for use with Pinecone serverless indexes"""

    US_CENTRAL1 = "us-central1"
    EUROPE_WEST4 = "europe-west4"


class AzureRegion(Enum):
    """Azure regions available for use with Pinecone serverless indexes"""

    EAST_US = "eastus2"
