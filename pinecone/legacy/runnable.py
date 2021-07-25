#
# Copyright (c) 2020-2021 Pinecone Systems Inc. All right reserved.
#

import abc
from typing import List, Optional


class Runnable:

    @abc.abstractmethod
    def to_args(self):
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def from_args(cls, args):
        raise NotImplementedError

    @abc.abstractmethod
    def start(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def image(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def replicas(self) -> int:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def shards(self) -> int:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def ports(self) -> List[int]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def ext_port(self) -> Optional[int]:
        raise NotImplementedError

    @abc.abstractproperty
    def metrics_port(self) -> Optional[int]:
        raise NotImplementedError

    @property
    def memory_request(self) -> int:
        return 200

    @property
    def cpu_request(self) -> int:
        """In 1/1000th of cpu"""
        return 100

    @property
    def volume_request(self) -> Optional[int]:
        """
        Either the size of requested persistent volume in GB or None if ephemeral
        :return:
        """
        return None
