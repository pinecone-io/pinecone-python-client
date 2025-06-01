import logging
from pinecone import Pinecone, NotFoundException
import time
from collections import deque
import json
import urllib3
import os
import dotenv

dotenv.load_dotenv()

logging.basicConfig(
    level=logging.DEBUG, format="%(levelname)-8s | %(name)s:%(lineno)4d | %(message)s"
)
logger = logging.getLogger(__name__)


class AdminAPI:
    def __init__(self, client_id, client_secret):
        self.http = urllib3.PoolManager()
        self.token = None
        self.token = self._get_token(client_id, client_secret)

    def _request(self, method, url, headers={}, body_dict=None):
        logger.info(f"Requesting {method} {url} with body {body_dict}")
        api_version = os.environ.get("API_VERSION", "2025-04")
        default_headers = {
            "X-Pinecone-Api-Version": api_version,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.token is not None:
            default_headers["Authorization"] = f"Bearer {self.token}"
        headers = {**default_headers, **headers}

        args = {"method": method, "url": url, "headers": headers}
        if body_dict is not None:
            args["body"] = json.dumps(body_dict)
        response = self.http.request(**args)

        logger.info(f"Response Status: {response.status}")
        if response.status >= 400:
            raise Exception(
                f"Request failed with status {response.status}: {response.data.decode('utf-8')}"
            )

        if response is None or response.data is None or response.data == b"":
            return None
        return json.loads(response.data.decode("utf-8"))

    def _get_token(self, client_id, client_secret):
        response = self._request(
            "POST",
            "https://login.pinecone.io/oauth/token",
            body_dict={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
                "audience": "https://api.pinecone.io/",
            },
        )

        token = response["access_token"]
        return token

    def create_project(self, project_name, max_pods):
        response = self._request(
            "POST",
            "https://api.pinecone.io/admin/projects",
            body_dict={"name": project_name, "max_pods": max_pods},
        )
        return response

    def create_api_key(self, project_id):
        response = self._request(
            "POST",
            f"https://api.pinecone.io/admin/projects/{project_id}/api-keys",
            body_dict={"name": "ci-key"},
        )
        return response

    def list_projects(self):
        response = self._request("GET", "https://api.pinecone.io/admin/projects")
        return response

    def describe_project(self, project_id):
        response = self._request("GET", f"https://api.pinecone.io/admin/projects/{project_id}")
        return response

    def delete_project(self, project_id):
        response = self._request("DELETE", f"https://api.pinecone.io/admin/projects/{project_id}")
        return response

    def get_project_id(self, project_name):
        project_list = self.list_projects()["data"]
        for project in project_list:
            if project["name"] == project_name:
                return project["id"]
        return None


class RetryCounter:
    def __init__(self, max_retries):
        self.max_retries = max_retries
        self.counts = {}

    def increment(self, key):
        if key not in self.counts:
            self.counts[key] = 0
        self.counts[key] += 1

    def get_count(self, key):
        return self.counts.get(key, 0)

    def is_maxed_out(self, key):
        return self.get_count(key) >= self.max_retries


class ProjectEraser:
    def __init__(self, api_key):
        self.pc = Pinecone(api_key=api_key)

        # In situations where there are a lot of resources, we want to
        # slow down the rate of requests just to avoid any concerns about
        # rate limits
        self.sleep_interval = 5
        self.undeleteable_resources = []

    def pluralize(self, resource_name):
        if resource_name.lower() == "index":
            return resource_name + "es"
        else:
            return resource_name + "s"

    def _delete_all_of_resource(self, resource_name, list_func, delete_func, get_state_func):
        resources_to_delete = deque(list_func())
        if len(resources_to_delete) == 0:
            logger.info(f"No {self.pluralize(resource_name)} to delete")
            return

        state_check_retries = RetryCounter(3)
        failed_delete_retries = RetryCounter(3)
        is_deletable_retries = RetryCounter(3)
        is_terminating_retries = RetryCounter(10)

        undeletable_resources = []

        while len(resources_to_delete) > 0:
            logger.info(
                f"There are {len(resources_to_delete)} {self.pluralize(resource_name)} left to delete"
            )
            time.sleep(self.sleep_interval)

            resource = resources_to_delete.popleft()
            logger.info(f"Processing {resource_name} {resource.name}")

            # Get the latest description of the resource
            try:
                state_check_retries.increment(resource.name)
                state = get_state_func(name=resource.name)
                logger.info(f"{resource_name} {resource.name} has state {state}")
            except NotFoundException:
                logger.info(f"{resource_name} {resource.name} has already been deleted, continuing")
                continue
            except Exception as e:
                if state_check_retries.is_maxed_out(resource.name):
                    logger.error(f"Error describing {resource_name} {resource.name}: {e}")
                    undeletable_resources.append(
                        {
                            "resource": resource,
                            "type": resource_name,
                            "reason": f"Error describing {resource_name} {resource.name}: {e}",
                        }
                    )
                    continue
                else:
                    logger.info(
                        f"{resource_name} {resource.name} has been returned to the back of the delete queue"
                    )
                    resources_to_delete.append(resource)
                    continue

            if state == "Terminating" or state == "Terminated":
                is_terminating_retries.increment(resource.name)
                if is_terminating_retries.is_maxed_out(resource.name):
                    logger.error(
                        f"{resource_name} {resource.name} has been in the terminating state for too long, skipping"
                    )
                    undeletable_resources.append(
                        {
                            "resource": resource,
                            "type": resource_name,
                            "reason": f"{resource_name} has been in the terminating state for too long",
                        }
                    )
                    continue
                else:
                    logger.info(
                        f"{resource_name} {resource.name} is in the process of being deleted, adding to the back of the delete queue to recheck later"
                    )
                    resources_to_delete.append(resource)
                    continue

            # If the index is not in a deleteable state, add it to the back of the delete queue
            deleteable_states = ["Ready", "InitializationFailed"]
            if state not in deleteable_states:
                is_deletable_retries.increment(resource.name)
                if is_deletable_retries.is_maxed_out(resource.name):
                    attempts = is_deletable_retries.get_count(resource.name)
                    logger.error(
                        f"{resource_name} {resource.name} did not enter a deleteable state after {attempts} attempts, skipping"
                    )
                    undeletable_resources.append(
                        {
                            "resource": resource,
                            "type": resource_name,
                            "reason": f"Not in a deleteable state after {attempts} attempts",
                        }
                    )
                    continue
                else:
                    logger.info(
                        f"{resource_name} {resource.name} state {state} is not deleteable, adding to the back of the delete queue"
                    )
                    resources_to_delete.append(resource)
                    continue

            try:
                logger.info(f"Attempting deleting of {resource_name} {resource.name}")
                delete_func(name=resource.name)
                logger.info(f"Successfully deleted {resource_name} {resource.name}")
            except Exception as e:
                logger.error(f"Error deleting {resource_name} {resource.name}: {e}")
                failed_delete_retries.increment(resource.name)
                if failed_delete_retries.is_maxed_out(resource.name):
                    attempts = failed_delete_retries.get_count(resource.name)
                    logger.error(
                        f"Failed to delete {resource_name} {resource.name} after {attempts} attempts, skipping"
                    )
                    undeletable_resources.append(
                        {
                            "resource": resource,
                            "type": resource_name,
                            "reason": f"Failed to delete after {attempts} attempts",
                        }
                    )
                    continue
                else:
                    logger.info(
                        f"{resource_name} {resource.name} has been returned to the back of the delete queue"
                    )
                    resources_to_delete.append(resource)
                    continue

        if len(undeletable_resources) > 0:
            logger.error(
                f"There were {len(undeletable_resources)} {self.pluralize(resource_name)} that were not deleted"
            )
            for item in undeletable_resources:
                logger.error(
                    f"{resource_name} {item['resource'].name} was not deleted because {item['reason']}"
                )
                self.undeleteable_resources.append(item)
        else:
            logger.info(f"All {self.pluralize(resource_name)} were deleted successfully")

    def delete_all_indexes(self):
        index_list = self.pc.db.index.list()
        if len(index_list) == 0:
            logger.info("No indexes to delete")
            return

        index_with_deletion_protection = [
            index for index in index_list if index.deletion_protection == "enabled"
        ]
        for index in index_with_deletion_protection:
            logger.info(f"Disabling deletion protection for Index {index.name}")
            time.sleep(self.sleep_interval)
            try:
                self.pc.db.index.configure(name=index.name, deletion_protection="disabled")
            except Exception as e:
                logger.error(f"Error disabling deletion protection for Index {index.name}: {e}")
                self.undeleteable_resources.append(
                    {
                        "resource": index,
                        "type": "index",
                        "reason": f"Failed to disable deletion protection: {e}",
                    }
                )

        def get_state_func(name):
            desc = self.pc.db.index.describe(name=name)
            return desc.status.state

        return self._delete_all_of_resource(
            resource_name="index",
            list_func=self.pc.db.index.list,
            delete_func=self.pc.db.index.delete,
            get_state_func=get_state_func,
        )

    def delete_all_collections(self):
        def get_state_func(name):
            desc = self.pc.db.collection.describe(name=name)
            return desc["status"]

        return self._delete_all_of_resource(
            resource_name="collection",
            list_func=self.pc.db.collection.list,
            delete_func=self.pc.db.collection.delete,
            get_state_func=get_state_func,
        )

    def delete_all_backups(self):
        def _get_backup_by_name(name):
            for backup in self.pc.db.backup.list():
                if backup.name == name:
                    return backup
            raise Exception(f"Backup {name} not found")

        def delete_func(name):
            backup = _get_backup_by_name(name)
            return self.pc.db.backup.delete(backup_id=backup.backup_id)

        def get_state_func(name):
            backup = _get_backup_by_name(name)
            return backup.status

        return self._delete_all_of_resource(
            resource_name="backup",
            list_func=self.pc.db.backup.list,
            delete_func=delete_func,
            get_state_func=get_state_func,
        )

    def _cleanup_all(self):
        self.undeleteable_resources = []
        self.delete_all_backups()
        self.delete_all_collections()
        self.delete_all_indexes()

    def cleanup_all(self):
        self._cleanup_all()

        if len(self.undeleteable_resources) > 0:
            logger.info(
                f"There were {len(self.undeleteable_resources)} undeleteable resources, retrying in 60 seconds"
            )
            time.sleep(60)
            self._cleanup_all()

        if len(self.undeleteable_resources) > 0:
            logger.info(
                f"There were {len(self.undeleteable_resources)} undeleteable resources, retrying in 120 seconds"
            )
            time.sleep(120)
            self._cleanup_all()

        if len(self.undeleteable_resources) > 0:
            logger.info(
                f"There were {len(self.undeleteable_resources)} undeleteable resources, retrying in 240 seconds"
            )
            time.sleep(240)
            self._cleanup_all()

        if len(self.undeleteable_resources) > 0:
            logger.info(
                f"There were {len(self.undeleteable_resources)} undeleteable resources, retrying in 240 seconds"
            )
            time.sleep(240)
            self._cleanup_all()

        if len(self.undeleteable_resources) > 0:
            logger.error(
                f"There were {len(self.undeleteable_resources)} undeleteable resources, giving up"
            )
            raise Exception(
                f"There were {len(self.undeleteable_resources)} undeleteable resources, giving up"
            )


if __name__ == "__main__":
    from pinecone import __version__

    logger.info(f"Pinecone version: {__version__}")

    if (
        os.environ.get("PINECONE_SERVICE_ACCOUNT_CLIENT_ID") is not None
        and os.environ.get("PINECONE_SERVICE_ACCOUNT_CLIENT_SECRET") is not None
    ):
        admin_api = AdminAPI(
            os.environ.get("PINECONE_SERVICE_ACCOUNT_CLIENT_ID"),
            os.environ.get("PINECONE_SERVICE_ACCOUNT_CLIENT_SECRET"),
        )
    else:
        raise Exception(
            "PINECONE_SERVICE_ACCOUNT_CLIENT_ID and PINECONE_SERVICE_ACCOUNT_CLIENT_SECRET must be set"
        )

    projects = admin_api.list_projects()
    for project in projects["data"]:
        project_name = project["name"]
        donotdelete = ["python-plugin-embeddings", "pinecone-python-client"]
        if project_name.startswith("python") or project_name in donotdelete:
            logger.info(f"=== Cleaning up project {project_name} ===")
            api_key = admin_api.create_api_key(project["id"])["value"]
            ProjectEraser(api_key).cleanup_all()

        if project_name.startswith("python") and project_name not in donotdelete:
            logger.info(f"=== Deleting project {project_name} ===")
            admin_api.delete_project(project["id"])
            time.sleep(10)
