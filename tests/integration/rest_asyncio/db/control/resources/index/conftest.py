import pytest


@pytest.fixture()
def spec1(serverless_cloud, serverless_region):
    return {"serverless": {"cloud": serverless_cloud, "region": serverless_region}}
