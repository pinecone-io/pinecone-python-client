import logging
from pinecone import Pinecone, NotFoundException
import time
from collections import deque

logger = logging.getLogger(__name__)


class _RetryCounter:
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


class _ProjectEraser:
    """
    This class is used to delete all resources within a project
    """

    def __init__(self, api_key, max_retries=5, sleep_interval=0.5):
        self.pc = Pinecone(api_key=api_key)

        # In situations where there are a lot of resources, we want to
        # slow down the rate of requests just to avoid any concerns about
        # rate limits
        self.sleep_interval = sleep_interval
        self.undeleteable_resources = []
        self.max_retries = max_retries

    def _pluralize(self, resource_name):
        if resource_name.lower() == "index":
            return resource_name + "es"
        else:
            return resource_name + "s"

    def _delete_all_of_resource(self, resource_name, list_func, delete_func, get_state_func):
        resources_to_delete = deque(list_func())
        if len(resources_to_delete) == 0:
            logger.info(f"No {self._pluralize(resource_name)} to delete")
            return

        state_check_retries = _RetryCounter(self.max_retries)
        failed_delete_retries = _RetryCounter(self.max_retries)
        is_deletable_retries = _RetryCounter(self.max_retries)
        is_terminating_retries = _RetryCounter(self.max_retries * 3)

        undeletable_resources = []

        while len(resources_to_delete) > 0:
            logger.info(
                f"There are {len(resources_to_delete)} {self._pluralize(resource_name)} left to delete"
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
                f"There were {len(undeletable_resources)} {self._pluralize(resource_name)} that were not deleted"
            )
            for item in undeletable_resources:
                logger.error(
                    f"{resource_name} {item['resource'].name} was not deleted because {item['reason']}"
                )
                self.undeleteable_resources.append(item)
        else:
            logger.info(f"All {self._pluralize(resource_name)} were deleted successfully")

    def delete_all_indexes(self, force_delete=False):
        index_list = self.pc.db.index.list()
        if len(index_list) == 0:
            logger.info("No indexes to delete")
            return

        index_with_deletion_protection = [
            index for index in index_list if index.deletion_protection == "enabled"
        ]
        if not force_delete and len(index_with_deletion_protection) > 0:
            logger.info(
                "There are indexes with deletion protection enabled. You must disable deletion protection before the index can be deleted."
            )
            raise Exception(
                f"Indexes with deletion protection enabled cannot be deleted: {[i.name for i in index_with_deletion_protection]}"
            )

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

    def retry_needed(self):
        if len(self.undeleteable_resources) > 0:
            logger.info(
                f"Retry needed for {len(self.undeleteable_resources)} undeleteable resources"
            )
            return True
        else:
            return False
