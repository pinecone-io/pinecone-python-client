#
# Copyright (c) 2020-2021 Pinecone Systems Inc. All right reserved.
#

from typing import Optional, List

from pinecone.legacy.specs import Spec
from pinecone.legacy.specs.service import Service
from pinecone.utils import validate_dns_name

from pinecone import logger


class TrafficRouter(Spec):

    def __init__(self, name: str, services: List[str], active: Optional[str]):
        """
        :param port: Port to run the gRPC gateway on
        :param native: Whether to run with native python processes or kubernetes
        """
        if active:
            assert active in services
        self._name = name
        self.active_service = active
        self.all_services = services

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str):
        validate_dns_name(value)
        self._name = value

    def set_active(self, service: str):
        assert service in self.all_services
        self.active_service = service

    def add_service(self, service: str):
        if service in self.all_services:
            logger.error("Service already in traffic router")
            return
        self.all_services.append(service)

    def remove_service(self, service: Service):
        self.all_services.remove(service)
        if self.active_service == service:
            self.active_service = None

    def validate(self):
        """Validates the router.

        Perform the following validations:

        - Router name should be DNS compatible
        """
        validate_dns_name(self._name)

    @classmethod
    def from_obj(cls, obj: dict) -> "TrafficRouter":
        assert obj['kind'] == 'TrafficRouter'
        metadata = obj['metadata']
        spec = obj['spec']
        logger.debug(str(spec))

        router = cls(metadata['name'], spec['services'], spec['active'])
        return router

    def to_obj(self) -> dict:
        metadata = {'name': self.name}
        spec = {
            'active': self.active_service,
            'services': self.all_services
        }
        return {
            'version': 'pinecone/v1alpha1',
            'kind': 'TrafficRouter',
            'metadata': metadata,
            'spec': spec
        }
