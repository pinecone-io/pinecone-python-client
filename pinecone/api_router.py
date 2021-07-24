#
# Copyright (c) 2020-2021 Pinecone Systems Inc. All right reserved.
#

from typing import List
from pinecone import logger
from pinecone.api_base import BaseAPI

__all__ = ["RouterAPI"]


class RouterAPI(BaseAPI):
    """API calls to the controller."""

    def get_router(self, router_name):
        """Returns the spec of a router."""
        return self.get("/routers/{}".format(router_name))['router']

    def deploy(self, router_json: str):
        """Creates or updates a traffic router."""
        res_json = self.post("/routers", json={"router": router_json})
        if res_json["success"]:
            logger.success("Successfully deployed to {}".format(self.host))
        else:
            # TODO: improve error handling
            if "already exists" in res_json["msg"]:
                raise RuntimeError("Service already exists.")
            else:
                raise RuntimeError("Failed to deploy: {}".format(res_json["msg"]))
        return res_json

    def list(self) -> List[str]:
        """Returns the names of all traffic routers."""
        return self.get("/routers")

    def stop(self, router_name: str):
        """Stops a traffic router."""
        response = self.delete("/routers/{}".format(router_name))
        if not response["success"]:
            logger.error("Failed to stop {0}, it probably wasn't running!".format(router_name))
        return response

    def get_status(self, router_name: str) -> dict:
        """Gets status of a traffic router."""
        return self.get("/routers/{}/status".format(router_name))
