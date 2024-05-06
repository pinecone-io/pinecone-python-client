import time
import os
from typing import Optional

from pinecone.features.feature import PineconeFeature

class DeleteIndexFeature(PineconeFeature):
    def __init__(self):
        pass

    def location(self):
        return os.path.dirname(__file__)

    def target(self):
        return 'Pinecone'

    def methods(self):
        return {
            "delete_index": delete_index,
        }


def delete_index(self, name: str, timeout: Optional[int] = None):
    """Deletes a Pinecone index.

    Deleting an index is an irreversible operation. All data in the index will be lost.
    When you use this command, a request is sent to the Pinecone control plane to delete 
    the index, but the termination is not synchronous because resources take a few moments to
    be released. 
    
    You can check the status of the index by calling the `describe_index()` command.
    With repeated polling of the describe_index command, you will see the index transition to a 
    `Terminating` state before eventually resulting in a 404 after it has been removed.

    :param name: the name of the index.
    :type name: str
    :param timeout: Number of seconds to poll status checking whether the index has been deleted. If None, 
        wait indefinitely; if >=0, time out after this many seconds;
        if -1, return immediately and do not wait. Default: None
    :type timeout: int, optional
    """
    api_instance = self.index_api
    api_instance.delete_index(name)
    self.index_host_store.delete_host(self.config, name)

    def get_remaining():
        return name in self.list_indexes().names()

    if timeout == -1:
        return

    if timeout is None:
        while get_remaining():
            time.sleep(5)
    else:
        while get_remaining() and timeout >= 0:
            time.sleep(5)
            timeout -= 5
    if timeout and timeout < 0:
        raise (
            TimeoutError(
                "Please call the list_indexes API ({}) to confirm if index is deleted".format(
                    "https://www.pinecone.io/docs/api/operation/list_indexes/"
                )
            )
        )

