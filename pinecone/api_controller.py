#
# Copyright (c) 2020-2021 Pinecone Systems Inc. All right reserved.
#

from typing import List
from pinecone import logger
from pinecone.api_base import BaseAPI


__all__ = ["ControllerAPI"]


class ControllerAPI(BaseAPI):
    """API calls to the controller."""

    def get_service(self, service_name: str) -> str:
        """Returns service specs."""
        return self.get("/services/{}".format(service_name))["service"]

    def list_services(self) -> List[str]:
        """Returns the names of all of the services."""
        return self.get("/services")

    def get_status(self, service_name: str) -> dict:
        """Returns service status"""
        # TODO: service status should be an enum
        return self.get("/services/{}/status".format(service_name))

    def stop(self, service_name: str):
        """Stops an service."""
        response = self.delete("/services/{}".format(service_name))
        if not response["success"]:
            logger.error("Failed to stop the service '{}'. It probably wasn't running!".format(service_name))
        return response

    def deploy(self, service_json: str):
        """Deploys an service."""
        response = self.post("/services", json={'service': service_json})
        if response["success"]:
            logger.success("Successfully deployed {}".format(self.host))
        else:
            if "already exists" in response["msg"]:
                raise RuntimeError("Service already exists.")
            else:
                raise RuntimeError("Failed to deploy: {}".format(response["msg"]))
        return response
