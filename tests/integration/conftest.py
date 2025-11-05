import logging
from pinecone import Pinecone
from datetime import datetime, timedelta
import dotenv

dotenv.load_dotenv()


logger = logging.getLogger(__name__)


def pytest_sessionfinish(session, exitstatus):
    """
    Hook that runs after all tests have completed.
    This is a good place to clean up any resources that were created during the test session.
    """
    logger.info("Running final cleanup after all tests...")

    try:
        # Initialize Pinecone client
        pc = Pinecone()

        # Get all indexes
        indexes = pc.list_indexes()

        # Find test indexes (those created during this test run)
        test_indexes = [idx for idx in indexes.names() if idx.startswith("test-")]

        # Delete test indexes that are older than 1 hour (in case of failed cleanup)
        for index_name in test_indexes:
            try:
                description = pc.describe_index(name=index_name)
                created_at = datetime.fromisoformat(description.created_at.replace("Z", "+00:00"))

                if datetime.now(created_at.tzinfo) - created_at > timedelta(hours=1):
                    logger.info(f"Cleaning up old test index: {index_name}")
                    pc.delete_index(name=index_name, timeout=-1)
            except Exception as e:
                logger.warning(f"Failed to clean up index {index_name}: {str(e)}")

    except Exception as e:
        logger.error(f"Error during final cleanup: {str(e)}")

    logger.info("Final cleanup completed")
