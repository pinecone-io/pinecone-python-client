#
# Copyright (c) 2020-2021 Pinecone Systems Inc. All right reserved.
#
from pathlib import Path

from pinecone.legacy.functions import Function
from pinecone.legacy.utils.constants import NODE_TYPE_RESOURCES
from abc import abstractmethod
from typing import Dict, Optional


class Index(Function):

    """
    Generic interface for defining an engine
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @abstractmethod
    def size(self) -> int:
        raise NotImplementedError

    def get_stats(self) -> Dict:
        return {'size': self.size()}

    @property
    def stats_frequency_seconds(self):
        return 60

    @property
    def memory_request(self) -> int:
        return NODE_TYPE_RESOURCES[self._node_type]['memory']

    @property
    def cpu_request(self) -> int:
        return NODE_TYPE_RESOURCES[self._node_type]['cpu']

    @property
    def volume_request(self) -> Optional[int]:
        return NODE_TYPE_RESOURCES[self._node_type]['disk']

    @abstractmethod
    def save(self, path: Path):
        raise NotImplementedError

    @abstractmethod
    def load(self, path: Path):
        raise NotImplementedError
