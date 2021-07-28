#
# Copyright (c) 2020-2021 Pinecone Systems Inc. All right reserved.
#

from typing import NamedTuple
from pinecone import __version__
from pinecone.api_base import BaseAPI

__all__ = ["ActionAPI"]


class WhoAmIResponse(NamedTuple):
    username: str
    user_label: str
    projectname: str


class VersionResponse(NamedTuple):
    server: str
    client: str


class ActionAPI(BaseAPI):
    """User related API calls."""

    def whoami(self) -> WhoAmIResponse:
        """Returns user information."""
        response = self.get("/actions/whoami")
        return WhoAmIResponse(
            username=response.get("user_name", "UNDEFINED"),
            projectname=response.get("project_name", "UNDEFINED"),
            user_label=response.get("user_label", "UNDEFINED"),
        )

    def version(self) -> VersionResponse:
        """Returns version information."""
        response = self.get("/actions/version")
        return VersionResponse(server=response.get("version", "UNKNOWN"),
                               client=__version__)
