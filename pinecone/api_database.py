#
# Copyright (c) 2020-2021 Pinecone Systems Inc. All right reserved.
#

from typing import List
from pinecone import logger
from pinecone.api_base import BaseAPI


class DatabaseAPI(BaseAPI):
    """API calls to the controller."""

    def list_services(self) -> List[str]:
        """Returns the names of all of the indexes."""
        return self.get("/databases")

    def get_database(self, db_name: str):
        """Returns the database spec"""
        return self.get("/databases/{}".format(db_name))["database"]

    def get_status(self, db_name: str) -> dict:
        """Returns index status"""
        return self.get("/databases/{}/status".format(db_name))

    def stop(self, db_name: str):
        """Stops an index."""
        response = self.delete("/databases/{}".format(db_name))
        if not response["success"]:
            logger.error("Failed to stop the index '{}'. It probably wasn't running!".format(db_name))
        return response

    def deploy(self, db_json: str):
        """Deploys an index."""
        response = self.post("/databases", json={'database': db_json})
        if response["success"]:
            logger.success("Successfully deployed {}".format(self.host))
        else:
            if "already exists" in response["msg"]:
                raise RuntimeError("Index already exists.")
            else:
                raise RuntimeError("Failed to deploy: {}".format(response["msg"]))
        return response

    def update(self, db_name: str, replicas: int):
        """Updates an index"""
        response = self.patch("/databases/{}".format(db_name), json={'replicas': replicas})
        if response["success"]:
            logger.success("Succesfully updated {}".format(self.host))
        else:
            raise RuntimeError("Failed to update: {}".format(response["msg"]))
        return response
