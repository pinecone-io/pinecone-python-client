import logging
from pinecone import Pinecone, NotFoundException
import time
from collections import deque
from .resources import (
    _DeleteableResource,
    _DeleteableIndex,
    _DeleteableCollection,
    _DeleteableBackup,
)
from typing import NamedTuple
from .retry_counter import _RetryCounter

logger = logging.getLogger(__name__)


class DeletionFailure(NamedTuple):
    resource_type: str
    resource_name: str
    reason: str


class _ProjectEraser:
    """
    This class is used to delete all resources within a project

    It should not be used directly, but rather through :func:`pinecone.admin.resources.ProjectResource.delete`
    """

    def __init__(self, api_key, max_retries=5, sleep_interval=0.5):
        self.api_key = api_key
        self.pc = Pinecone(api_key=api_key)

        # In situations where there are a lot of resources, we want to
        # slow down the rate of requests just to avoid any concerns about
        # rate limits
        self.sleep_interval = sleep_interval
        self.undeleteable_resources = []
        self.max_retries = max_retries

        self.state_check_retries = _RetryCounter(self.max_retries)
        self.failed_delete_retries = _RetryCounter(self.max_retries)
        self.is_deletable_retries = _RetryCounter(self.max_retries)
        self.is_terminating_retries = _RetryCounter(self.max_retries * 3)

        self.undeletable_resources = []

    def _get_state(self, dr, resource, delete_queue):
        should_continue = False
        label = f"{dr.name()} {resource.name}"
        try:
            self.state_check_retries.increment(resource.name)
            state = dr.get_state(name=resource.name)
            logger.debug(f"{label} has state {state}")
            should_continue = False
        except NotFoundException:
            logger.debug(f"{label} has already been deleted, continuing")
            should_continue = True
        except Exception as e:
            if self.state_check_retries.is_maxed_out(resource.name):
                logger.error(f"Error describing {label}: {e}")
                self.undeletable_resources.append(
                    DeletionFailure(
                        resource_type=dr.name(),
                        resource_name=resource.name,
                        reason=f"Error describing {label}: {e}",
                    )
                )
                should_continue = True
            else:
                logger.debug(f"{label} has been returned to the back of the delete queue")
                delete_queue.append(resource)
                should_continue = True

        return (state, should_continue)

    def _check_if_terminating(self, state, dr, resource, delete_queue):
        terminating_states = ["Terminating", "Terminated"]
        label = f"{dr.name()} {resource.name}"

        if state not in terminating_states:
            return False

        self.is_terminating_retries.increment(resource.name)
        if self.is_terminating_retries.is_maxed_out(resource.name):
            logger.error(f"{label} has been in the terminating state for too long, skipping")
            self.undeletable_resources.append(
                DeletionFailure(
                    resource_type=dr.name(),
                    resource_name=resource.name,
                    reason=f"{label} has been in the terminating state for too long",
                )
            )
        else:
            logger.debug(
                f"{label} is in the process of being deleted, adding to the back of the delete queue to recheck later"
            )
        delete_queue.append(resource)

        return True

    def _check_if_deletable(self, state, dr, resource, delete_queue):
        label = f"{dr.name()} {resource.name}"
        deleteable_states = ["Ready", "InitializationFailed"]

        if state in deleteable_states:
            return False

        self.is_deletable_retries.increment(resource.name)
        if self.is_deletable_retries.is_maxed_out(resource.name):
            attempts = self.is_deletable_retries.get_count(resource.name)
            logger.error(
                f"{label} did not enter a deleteable state after {attempts} attempts, skipping"
            )
            self.undeletable_resources.append(
                DeletionFailure(
                    resource_type=dr.name(),
                    resource_name=resource.name,
                    reason=f"Not in a deleteable state after {attempts} attempts",
                )
            )
        else:
            logger.debug(
                f"{label} state {state} is not deleteable, adding to the back of the delete queue"
            )
            delete_queue.append(resource)

        return True

    def _attempt_delete(self, dr, resource, delete_queue):
        label = f"{dr.name()} {resource.name}"
        try:
            logger.debug(f"Attempting deleting of {label}")
            dr.delete(name=resource.name)
            logger.debug(f"Successfully deleted {label}")
        except NotFoundException:
            logger.debug(f"{label} has already been deleted, continuing")
        except Exception as e:
            logger.error(f"Error deleting {label}: {e}")
            self.failed_delete_retries.increment(resource.name)

            if self.failed_delete_retries.is_maxed_out(resource.name):
                attempts = self.failed_delete_retries.get_count(resource.name)
                logger.error(f"Failed to delete {label} after {attempts} attempts, skipping")
                self.undeletable_resources.append(
                    DeletionFailure(
                        resource_type=dr.name(),
                        resource_name=resource.name,
                        reason=f"Failed to delete after {attempts} attempts",
                    )
                )
            else:
                logger.debug(f"{label} has been returned to the back of the delete queue")
                delete_queue.append(resource)

    def _log_final_state(self, dr):
        if len(self.undeletable_resources) > 0:
            logger.error(
                f"There were {len(self.undeletable_resources)} {dr.name_plural()} that were not deleted"
            )
            for item in self.undeletable_resources:
                logger.error(
                    f"{item.resource_type} {item.resource_name} was not deleted because {item.reason}"
                )
                self.undeleteable_resources.append(item)
        else:
            logger.debug(f"All {dr.name_plural()} were deleted successfully")

    def _delete_resource_type(self, dr: _DeleteableResource):
        delete_queue = deque(dr.list())
        if len(delete_queue) == 0:
            logger.debug(f"No {dr.name_plural()} to delete")
            return

        while len(delete_queue) > 0:
            logger.debug(f"There are {len(delete_queue)} {dr.name_plural()} left to delete")
            time.sleep(self.sleep_interval)
            resource = delete_queue.popleft()
            label = f"{dr.name()} {resource.name}"

            logger.debug(f"Processing {label}")

            # Get the latest description of the resource
            state, should_continue = self._get_state(dr, resource, delete_queue)
            if should_continue:
                continue

            # If the resource is in the terminating state, add it to the back of the delete queue to recheck later
            should_continue = self._check_if_terminating(state, dr, resource, delete_queue)
            if should_continue:
                continue

            # If the index is not in a deleteable state, add it to the back of the delete queue
            should_continue = self._check_if_deletable(state, dr, resource, delete_queue)
            if should_continue:
                continue

            # If the resource is deletable, delete it
            self._attempt_delete(dr, resource, delete_queue)

        self._log_final_state(dr)

    def delete_all_indexes(self, force_delete=False):
        index_list = self.pc.db.index.list()
        index_with_deletion_protection = [
            index for index in index_list if index.deletion_protection == "enabled"
        ]
        if not force_delete and len(index_with_deletion_protection) > 0:
            raise Exception(
                f"Indexes with deletion protection enabled cannot be deleted: {[i.name for i in index_with_deletion_protection]}"
            )

        for index in index_with_deletion_protection:
            logger.debug(f"Disabling deletion protection for Index {index.name}")
            time.sleep(self.sleep_interval)
            try:
                self.pc.db.index.configure(name=index.name, deletion_protection="disabled")
            except Exception as e:
                logger.error(f"Error disabling deletion protection for Index {index.name}: {e}")
                self.undeleteable_resources.append(
                    DeletionFailure(
                        resource_type="index",
                        resource_name=index.name,
                        reason=f"Failed to disable deletion protection: {e}",
                    )
                )

        index_eraser = _DeleteableIndex(pc=self.pc)
        return self._delete_resource_type(index_eraser)

    def delete_all_collections(self):
        collection_eraser = _DeleteableCollection(pc=self.pc)
        return self._delete_resource_type(collection_eraser)

    def delete_all_backups(self):
        backup_eraser = _DeleteableBackup(pc=self.pc)
        return self._delete_resource_type(backup_eraser)

    def retry_needed(self):
        if len(self.undeleteable_resources) > 0:
            logger.debug(
                f"Retry needed for {len(self.undeleteable_resources)} undeleteable resources"
            )
            return True
        else:
            return False
