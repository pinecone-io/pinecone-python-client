import pytest

from pinecone import CloudProvider, AwsRegion, ServerlessSpec


@pytest.fixture()
def spec1(serverless_cloud, serverless_region):
    return {"serverless": {"cloud": serverless_cloud, "region": serverless_region}}


@pytest.fixture()
def spec2():
    return ServerlessSpec(cloud=CloudProvider.AWS, region=AwsRegion.US_EAST_1)


@pytest.fixture()
def spec3():
    return {"serverless": {"cloud": CloudProvider.AWS, "region": AwsRegion.US_EAST_1}}
