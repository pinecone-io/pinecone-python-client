import os
import logging
import time
from pinecone import Admin
from pinecone.admin.eraser.project_eraser import _ProjectEraser

logging.basicConfig(
    level=logging.DEBUG, format="%(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
)
logger = logging.getLogger(__name__)


def mask(value):
    """Mask the value in Github Actions logs"""
    print(f"::add-mask::{value}")


client_id = os.getenv("PINECONE_SERVICE_ACCOUNT_CLIENT_ID")
client_secret = os.getenv("PINECONE_SERVICE_ACCOUNT_CLIENT_SECRET")

if client_id is None:
    raise Exception("PINECONE_SERVICE_ACCOUNT_CLIENT_ID must be set")
if client_secret is None:
    raise Exception("PINECONE_SERVICE_ACCOUNT_CLIENT_SECRET must be set")

admin = Admin(client_id=client_id, client_secret=client_secret)

project_id = os.getenv("PROJECT_ID")
if project_id is None:
    raise Exception("PROJECT_ID must be set")

key_response = admin.api_keys.create(project_id=project_id, name="ci-cleanup")

eraser = _ProjectEraser(api_key=key_response.value)

done = False
retries = 5
while not done and retries > 0:
    try:
        eraser.delete_all_indexes(force_delete=True)
        eraser.delete_all_collections()
        eraser.delete_all_backups()
        done = True
    except Exception as e:
        logger.error(f"Error deleting project resources: {e}")
        time.sleep(10)
        retries -= 1

admin.api_keys.delete(api_key_id=key_response.key.id)
admin.projects.delete(project_id=project_id)
