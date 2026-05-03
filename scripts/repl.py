from __future__ import annotations

import code
import logging
import os
import readline
import time

import dotenv

from pinecone import Pinecone


def main() -> None:
    dotenv.load_dotenv()

    logging.basicConfig(
        level=logging.DEBUG, format="%(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
    )
    logger = logging.getLogger(__name__)

    histfile = os.path.join(os.path.expanduser("~"), ".python_repl_history")
    try:
        readline.read_history_file(histfile)
        readline.set_history_length(1000)
    except FileNotFoundError:
        pass

    banner = """
    Welcome to the custom Python REPL!
    Your initialization steps have been completed.

    Two Pinecone objects are available:
    - pc: Built using the PINECONE_API_KEY env var, if set
    - pcci: Built using the PINECONE_API_KEY_CI_TESTING env var, if set

    You can use the following functions to clean up the environment:
    - delete_all_indexes(pc)
    - delete_all_pod_indexes(pc)
    - delete_all_collections(pc)
    - delete_all_backups(pc)
    - cleanup_all(pc)
    """

    # In situations where there are a lot of resources, slow down the rate of requests.
    sleep_interval = 30

    def delete_all_pod_indexes(pc: Pinecone) -> None:
        for index in pc.indexes.list():
            if index.spec.pod is not None:
                logger.info(f"Deleting index {index.name}")
                try:
                    if index.deletion_protection == "enabled":
                        logger.info(f"Disabling deletion protection for index {index.name}")
                        pc.indexes.configure(name=index.name, deletion_protection="disabled")
                        time.sleep(sleep_interval)
                    pc.indexes.delete(name=index.name)
                    time.sleep(sleep_interval)
                except Exception as e:
                    logger.error(f"Error deleting index {index.name}: {e}")

    def delete_all_indexes(pc: Pinecone) -> None:
        for index in pc.indexes.list():
            logger.info(f"Deleting index {index.name}")
            try:
                if index.deletion_protection == "enabled":
                    logger.info(f"Disabling deletion protection for index {index.name}")
                    pc.indexes.configure(name=index.name, deletion_protection="disabled")
                    time.sleep(sleep_interval)
                pc.indexes.delete(name=index.name)
                time.sleep(sleep_interval)
            except Exception as e:
                logger.error(f"Error deleting index {index.name}: {e}")

    def delete_all_collections(pc: Pinecone) -> None:
        for collection in pc.collections.list():
            logger.info(f"Deleting collection {collection.name}")
            try:
                pc.collections.delete(name=collection.name)
                time.sleep(sleep_interval)
            except Exception as e:
                logger.error(f"Error deleting collection {collection.name}: {e}")

    def delete_all_backups(pc: Pinecone) -> None:
        for backup in pc.backups.list():
            logger.info(f"Deleting backup {backup.name}")
            try:
                pc.backups.delete(backup_id=backup.backup_id)
                time.sleep(sleep_interval)
            except Exception as e:
                logger.error(f"Error deleting backup {backup.name}: {e}")

    def cleanup_all(pc: Pinecone) -> None:
        delete_all_indexes(pc)
        delete_all_collections(pc)
        delete_all_backups(pc)

    # We want to route through preprod by default
    if os.environ.get("PINECONE_ADDITIONAL_HEADERS") is None:
        logger.warning(
            'You have not set a value for PINECONE_ADDITIONAL_HEADERS in your .env file '
            'so the default value of {"x-environment": "preprod-aws-0"} will be used.'
        )
        os.environ["PINECONE_ADDITIONAL_HEADERS"] = '{"x-environment": "preprod-aws-0"}'

    namespace: dict[str, object] = {
        "__name__": "__main__",
        "__doc__": None,
        "delete_all_indexes": delete_all_indexes,
        "delete_all_collections": delete_all_collections,
        "delete_all_backups": delete_all_backups,
        "delete_all_pod_indexes": delete_all_pod_indexes,
        "cleanup_all": cleanup_all,
        "pcl": Pinecone(api_key="foo", host="http://localhost:8000"),
    }

    if os.environ.get("PINECONE_API_KEY") is not None:
        namespace["pc"] = Pinecone()
    else:
        logger.warning(
            "You have not set a value for PINECONE_API_KEY in your .env file so the pc "
            "object was not pre-created for you. See .env.example for more information."
        )

    if os.environ.get("PINECONE_API_KEY_CI_TESTING") is not None:
        namespace["pcci"] = Pinecone(api_key=os.environ.get("PINECONE_API_KEY_CI_TESTING"))
    else:
        logger.warning(
            "You have not set a value for PINECONE_API_KEY_CI_TESTING in your .env file "
            "so the pcci object was not pre-created for you. See .env.example for more information."
        )

    try:
        code.interact(banner=banner, local=namespace)
    finally:
        readline.write_history_file(histfile)


if __name__ == "__main__":
    main()
