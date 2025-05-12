import time
import random
import logging

logger = logging.getLogger(__name__)


def random_vector(dimension):
    return [random.uniform(0, 1) for _ in range(dimension)]


def attempt_cleanup_collection(pc, collection_name):
    max_wait = 120
    time_waited = 0
    deleted = False

    while time_waited < max_wait:
        try:
            pc.db.collection.delete(name=collection_name)
            deleted = True
            break
        except Exception as e:
            # Failures here usually happen because the backend thinks there is still some
            # operation pending on the resource.
            # These orphaned resources will get cleaned up by the cleanup job later.
            logger.debug(f"Error while cleaning up collection: {e}")
            logger.debug(
                f"Waiting for collection {collection_name} to be deleted. Waited {time_waited} seconds..."
            )
            time.sleep(10)
            time_waited += 10
    if not deleted:
        logger.warning(f"Collection {collection_name} was not deleted after {max_wait} seconds")


def attempt_cleanup_index(pc, index_name):
    max_wait = 120
    time_waited = 0
    deleted = False

    while time_waited < max_wait:
        try:
            pc.db.index.delete(name=index_name)
            deleted = True
            break
        except Exception as e:
            # Failures here usually happen because the backend thinks there is still some
            # operation pending on the resource.
            # These orphaned resources will get cleaned up by the cleanup job later.
            logger.debug(f"Error while cleaning up index: {e}")
            logger.debug(
                f"Waiting for index {index_name} to be deleted. Waited {time_waited} seconds..."
            )
            time.sleep(10)
            time_waited += 10
    if not deleted:
        logger.warning(f"Index {index_name} was not deleted after {max_wait} seconds")
