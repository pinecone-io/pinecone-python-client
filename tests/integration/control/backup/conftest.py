import pytest
import uuid
import time
import logging
import dotenv
from pinecone import Pinecone, NotFoundException, PineconeApiException
from ...helpers import generate_index_name, get_environment_var, index_tags as index_tags_helper

dotenv.load_dotenv()

logger = logging.getLogger(__name__)
""" @private """

# Generate a unique ID for the entire test run
RUN_ID = str(uuid.uuid4())


@pytest.fixture()
def index_tags(request):
    return index_tags_helper(request, RUN_ID)


@pytest.fixture()
def pc():
    api_key = get_environment_var("PINECONE_API_KEY")
    return Pinecone(
        api_key=api_key, additional_headers={"sdk-test-suite": "pinecone-python-client"}
    )


@pytest.fixture()
def serverless_cloud():
    return get_environment_var("SERVERLESS_CLOUD", "aws")


@pytest.fixture()
def serverless_region():
    return get_environment_var("SERVERLESS_REGION", "us-west-2")


@pytest.fixture()
def create_sl_index_params(index_name, serverless_cloud, serverless_region, index_tags):
    spec = {"serverless": {"cloud": serverless_cloud, "region": serverless_region}}
    return dict(name=index_name, dimension=10, metric="cosine", spec=spec, tags=index_tags)


@pytest.fixture()
def index_name(request):
    test_name = request.node.name
    return generate_index_name(test_name)


@pytest.fixture()
def ready_sl_index(pc, index_name, create_sl_index_params):
    create_sl_index_params["timeout"] = None
    pc.create_index(**create_sl_index_params)
    yield index_name
    pc.db.index.delete(name=index_name, timeout=-1)


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


@pytest.fixture(autouse=True)
def cleanup(pc, index_name):
    yield

    try:
        desc = pc.db.index.describe(name=index_name)
        if desc.deletion_protection == "enabled":
            logger.info(f"Disabling deletion protection for index: {index_name}")
            pc.db.index.configure(name=index_name, deletion_protection="disabled")
        logger.debug("Attempting to delete index with name: " + index_name)
        pc.db.index.delete(name=index_name, timeout=-1)
    except Exception:
        logger.warning(f"Failed to delete index {index_name}")
        pass


def pytest_sessionfinish(session, exitstatus):
    """
    Hook that runs after all tests have completed.
    This is a good place to clean up any resources that were created during the test session.
    """
    logger.info("Running final cleanup after all tests...")

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

        for backup in pc.db.backup.list():
            if backup.tags is not None and backup.tags.get("test-run") == RUN_ID:
                logger.debug(f"Deleting backup: {backup.name}")
                try:
                    pc.db.backup.delete(backup_id=backup.backup_id)
                except Exception as e:
                    logger.warning(f"Failed to delete backup: {backup.name}: {str(e)}")
            else:
                logger.info(
                    f"Backup {backup.name} is not a test backup from run {RUN_ID}. Skipping."
                )

    except Exception as e:
        logger.error(f"Error during final cleanup: {str(e)}")

    logger.info("Final cleanup completed")
