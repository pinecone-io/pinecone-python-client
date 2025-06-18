import logging
from pinecone import Admin
from pinecone.admin.eraser.project_eraser import _ProjectEraser
import time
import os

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    from pinecone import __version__

    logger.info(f"Pinecone version: {__version__}")

    if (
        os.environ.get("PINECONE_SERVICE_ACCOUNT_CLIENT_ID") is not None
        and os.environ.get("PINECONE_SERVICE_ACCOUNT_CLIENT_SECRET") is not None
    ):
        admin_api = Admin(
            client_id=os.environ.get("PINECONE_SERVICE_ACCOUNT_CLIENT_ID"),
            client_secret=os.environ.get("PINECONE_SERVICE_ACCOUNT_CLIENT_SECRET"),
        )
    else:
        raise Exception(
            "PINECONE_SERVICE_ACCOUNT_CLIENT_ID and PINECONE_SERVICE_ACCOUNT_CLIENT_SECRET must be set"
        )

    donotdelete = ["python-plugin-embeddings", "pinecone-python-client"]
    projects = admin_api.projects.list()
    for project in projects.data:
        project_name = project.name

        if project_name.startswith("python") or project_name in donotdelete:
            logger.info(f"=== Cleaning up project {project_name} ===")
            api_key_response = admin_api.api_keys.create(project_id=project.id, name="ci-cleanup")

            # force_delete=True overrides deletion protection. This seems too
            # risky to include in the Admin project delete function for
            # end users, so we do this extra delete step separately.
            eraser = _ProjectEraser(api_key=api_key_response.value)
            eraser.delete_all_indexes(force_delete=True)
            eraser.delete_all_collections()
            eraser.delete_all_backups()

            admin_api.api_keys.delete(api_key_id=api_key_response.key.id)

        if project_name.startswith("python") and project_name not in donotdelete:
            logger.info(f"=== Deleting project {project_name} ===")
            admin_api.projects.delete(project_id=project.id)
            time.sleep(10)
