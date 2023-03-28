#
# Copyright (c) 2020-2021 Pinecone Systems Inc. All right reserved.
#

from typing import NamedTuple
from pinecone.core.api_base import BaseAPI

__all__ = ["ActionAPI", "VersionResponse", "WhoAmIResponse"]

from pinecone.core.utils import get_version


class WhoAmIResponse(NamedTuple):
    """
    Represents the response returned by the Pinecone WhoAmI API.

    Example response:
        {"project_name":"5d54712","user_label":"default","user_name":"18bd872"}

    Fields:
        - username: the user ID of the Pinecone account, rather than username
        - user_label: the label of the user in the Pinecone account
        - projectname: the project ID of the Pinecone account, rather projectname
    """
    username: str = 'UNKNOWN'  # TODO:  should be named as user_id
    user_label: str = 'UNKNOWN'
    projectname: str = 'UNKNOWN'  # TODO:  should be named as project_id


class VersionResponse(NamedTuple):
    server: str
    client: str


class ActionAPI(BaseAPI):
    """User related API calls."""
    client_version = get_version()

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
                               client=self.client_version)
