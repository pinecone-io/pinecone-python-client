import pytest
import uuid
import time
import logging
import dotenv
import os
from datetime import datetime
from pinecone import Pinecone, NotFoundException, PineconeApiException
from ...helpers import get_environment_var

dotenv.load_dotenv()

logger = logging.getLogger(__name__)
""" @private """

# Generate a unique ID for the entire test run
RUN_ID = str(uuid.uuid4())


@pytest.fixture()
def index_tags(request):
    test_name = request.node.name
    if test_name is None:
        test_name = ""
    else:
        test_name = test_name.replace(":", "_").replace("[", "_").replace("]", "_")

    tags = {
        "test-suite": "pinecone-python-client",
        "test-run": RUN_ID,
        "test": test_name,
        "created-at": datetime.now().strftime("%Y-%m-%d"),
    }

    if os.getenv("USER"):
        tags["user"] = os.getenv("USER")
    return tags


@pytest.fixture()
def pc():
    api_key = get_environment_var("PINECONE_API_KEY")
    return Pinecone(
        api_key=api_key, additional_headers={"sdk-test-suite": "pinecone-python-client"}
    )


@pytest.fixture()
def pod_environment():
    return get_environment_var("PINECONE_ENVIRONMENT", "us-east1-gcp")


def delete_with_retry(pc, index_name, retries=0, sleep_interval=5):
    logger.debug(
        "Deleting index "
        + index_name
        + ", retry "
        + str(retries)
        + ", next sleep interval "
        + str(sleep_interval)
    )
    try:
        pc.db.index.delete(name=index_name, timeout=-1)
    except NotFoundException:
        pass
    except PineconeApiException as e:
        if e.error.code == "PRECONDITON_FAILED":
            if retries > 5:
                raise Exception("Unable to delete index " + index_name)
            time.sleep(sleep_interval)
            delete_with_retry(pc, index_name, retries + 1, sleep_interval * 2)
        else:
            logger.error(e.__class__)
            logger.error(e)
            raise Exception("Unable to delete index " + index_name)
    except Exception as e:
        logger.error(e.__class__)
        logger.error(e)
        raise Exception("Unable to delete index " + index_name)


def pytest_sessionfinish(session, exitstatus):
    """
    Hook that runs after all tests have completed.
    This is a good place to clean up any resources that were created during the test session.
    """
    logger.info("Running final cleanup after all collection tests...")

    try:
        pc = Pinecone()
        indexes = pc.db.index.list()
        test_indexes = [
            idx for idx in indexes if idx.tags is not None and idx.tags.get("test-run") == RUN_ID
        ]

        logger.info(f"Indexes to delete: {[idx.name for idx in test_indexes]}")

        for idx in test_indexes:
            if idx.deletion_protection == "enabled":
                logger.info(f"Disabling deletion protection for index: {idx.name}")
                pc.db.index.configure(name=idx.name, deletion_protection="disabled")
                # Wait for index to be updated with status ready
                logger.info(f"Waiting for index {idx.name} to be ready...")
                timeout = 60
                while True and timeout > 0:
                    is_ready = pc.db.index.describe(name=idx.name).ready
                    if is_ready:
                        break
                    time.sleep(1)
                    timeout -= 1
                if timeout <= 0:
                    logger.warning(f"Index {idx.name} did not become ready in time")
            else:
                logger.info(f"Deletion protection is already disabled for index: {idx.name}")

        for idx in test_indexes:
            try:
                logger.info(f"Deleting index: {idx.name}")
                pc.db.index.delete(name=idx.name, timeout=-1)
            except Exception as e:
                logger.warning(f"Failed to delete index {idx.name}: {str(e)}")

        collections = pc.db.collection.list()
        logger.info(f"Collections to delete: {[col.name for col in collections]}")

        for col in collections:
            try:
                logger.info(f"Deleting collection: {col.name}")
                pc.db.collection.delete(name=col.name)
            except Exception as e:
                logger.warning(f"Failed to delete collection {col.name}: {str(e)}")

    except Exception as e:
        logger.error(f"Error during final cleanup: {str(e)}")

    logger.info("Final cleanup of collections tests completed")
