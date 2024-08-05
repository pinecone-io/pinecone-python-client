import os
import pytest
import random
import time
import json
from pinecone import Pinecone, NotFoundException, PineconeApiException
from ...helpers import generate_index_name, get_environment_var


@pytest.fixture()
def use_grpc():
    return get_environment_var("USE_GRPC", "false") == "true"


@pytest.fixture()
def client(use_grpc):
    api_key = get_environment_var("PINECONE_API_KEY")
    if use_grpc:
        from pinecone.grpc import PineconeGRPC

        return PineconeGRPC(api_key=api_key, additional_headers={"sdk-test-suite": "pinecone-python-client"})
    else:
        return Pinecone(api_key=api_key, additional_headers={"sdk-test-suite": "pinecone-python-client"})


@pytest.fixture()
def spec():
    return json.loads(get_environment_var("SPEC"))


@pytest.fixture()
def create_sl_index_params(index_name, spec):
    return dict(name=index_name, dimension=10, metric="cosine", spec=spec)


@pytest.fixture()
def random_vector():
    return [random.uniform(0, 1) for _ in range(10)]


@pytest.fixture()
def index_name(request):
    test_name = request.node.name
    return generate_index_name(test_name)


@pytest.fixture()
def ready_sl_index(client, index_name, create_sl_index_params):
    create_sl_index_params["timeout"] = None
    client.create_index(**create_sl_index_params)
    yield index_name
    client.delete_index(index_name, -1)


@pytest.fixture()
def notready_sl_index(client, index_name, create_sl_index_params):
    client.create_index(**create_sl_index_params, timeout=-1)
    yield index_name


def delete_with_retry(client, index_name, retries=0, sleep_interval=5):
    print("Deleting index " + index_name + ", retry " + str(retries) + ", next sleep interval " + str(sleep_interval))
    try:
        client.delete_index(index_name, -1)
    except NotFoundException:
        pass
    except PineconeApiException as e:
        if e.error.code == "PRECONDITON_FAILED":
            if retries > 5:
                raise "Unable to delete index " + index_name
            time.sleep(sleep_interval)
            delete_with_retry(client, index_name, retries + 1, sleep_interval * 2)
        else:
            print(e.__class__)
            print(e)
            raise "Unable to delete index " + index_name
    except Exception as e:
        print(e.__class__)
        print(e)
        raise "Unable to delete index " + index_name


@pytest.fixture(autouse=True)
def cleanup(client, index_name):
    yield

    try:
        client.delete_index(index_name, -1)
    except:
        pass


@pytest.fixture(autouse=True, scope="session")
def cleanup_all():
    yield

    client = Pinecone(additional_headers={"sdk-test-suite": "pinecone-python-client"})
    for index in client.list_indexes():
        buildNumber = os.getenv("GITHUB_BUILD_NUMBER")
        if index.name.startswith(buildNumber):
            try:
                delete_with_retry(client, index.name)
            except:
                pass
