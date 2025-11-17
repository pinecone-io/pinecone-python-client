from pinecone.config import OpenApiConfiguration, Config
from pinecone.openapi_support import ApiClient
from pinecone.core.openapi.oauth import API_VERSION
from pinecone.core.openapi.oauth.apis import OAuthApi
from pinecone.core.openapi.oauth.models import TokenRequest
from pinecone.utils import get_user_agent
import os
from copy import deepcopy


class Admin:
    """
    A class for accessing the Pinecone Admin API.

    A prerequisite for using this class is to have a `service account <https://docs.pinecone.io/guides/organizations/manage-service-accounts>`_. To create a service
    account, visit the `Pinecone web console <https://app.pinecone.io>`_ and navigate to
    the ``Access > Service Accounts`` section.

    After creating a service account, you will be provided with a client ID and secret.
    These values can be passed to the Admin constructor or set the ``PINECONE_CLIENT_ID``
    and ``PINECONE_CLIENT_SECRET`` environment variables.


    :param client_id: The client ID for the Pinecone API. To obtain a client ID and secret,
        you must create a service account via the Pinecone web console. This value can be
        passed using keyword arguments or set the ``PINECONE_CLIENT_ID`` environment variable.
    :type client_id: Optional[str]
    :param client_secret: The client secret for the Pinecone API. To obtain a client ID
        and secret, you must create a service account via the Pinecone web console. This value
        can be passed using keyword arguments or set the ``PINECONE_CLIENT_SECRET`` environment
        variable.
    :type client_secret: Optional[str]
    :param additional_headers: Additional headers to use for the Pinecone API. This is a
        dictionary of key-value pairs. This is primarily used for internal testing
        purposes.
    :type additional_headers: Optional[dict[str, str]]
    """

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        additional_headers: dict[str, str] | None = None,
    ):
        """
        Initialize the ``Admin`` class.

        :param client_id: The client ID for the Pinecone API. To obtain a client ID and secret,
          you must create a service account via the Pinecone web console. This value can be
          passed using keyword arguments or set the ``PINECONE_CLIENT_ID`` environment variable.
        :type client_id: Optional[str]
        :param client_secret: The client secret for the Pinecone API. To obtain a client ID
          and secret, you must create a service account via the Pinecone web console. This value
          can be passed using keyword arguments or set the ``PINECONE_CLIENT_SECRET`` environment
          variable.
        :type client_secret: Optional[str]
        :param additional_headers: Additional headers to use for the Pinecone API. This is a
          dictionary of key-value pairs. This is primarily used for internal testing
          purposes.
        :type additional_headers: Optional[dict[str, str]]

        Examples
        --------

        .. code-block:: python
            :caption: Initialize Admin using environment variables

            import os
            from pinecone import Admin

            # Set environment variables
            os.environ["PINECONE_CLIENT_ID"] = "your-client-id"
            os.environ["PINECONE_CLIENT_SECRET"] = "your-client-secret"

            # Initialize Admin (reads from environment variables)
            admin = Admin()

        .. code-block:: python
            :caption: Initialize Admin with explicit credentials

            from pinecone import Admin

            # Initialize Admin with explicit credentials
            admin = Admin(
                client_id="your-client-id",
                client_secret="your-client-secret"
            )

        .. code-block:: python
            :caption: Initialize Admin with additional headers

            from pinecone import Admin

            # Initialize Admin with additional headers for testing
            admin = Admin(
                client_id="your-client-id",
                client_secret="your-client-secret",
                additional_headers={"X-Custom-Header": "value"}
            )

        """

        if client_id is not None:
            self._client_id = client_id
        else:
            self._client_id = os.environ.get("PINECONE_CLIENT_ID", "")
        if self._client_id is None or self._client_id == "":
            raise ValueError(
                "client_id is not set or is empty. Pass client_id to the Admin constructor or set the PINECONE_CLIENT_ID environment variable."
            )

        if client_secret is not None:
            self._client_secret = client_secret
        else:
            self._client_secret = os.environ.get("PINECONE_CLIENT_SECRET", "")
        if self._client_secret is None or self._client_secret == "":
            raise ValueError(
                "client_secret is not set or is empty. Pass client_secret to the Admin constructor or set the PINECONE_CLIENT_SECRET environment variable."
            )

        if additional_headers is None:
            additional_headers = {}

        _oauth_api_config = OpenApiConfiguration(host="https://login.pinecone.io")

        _oauth_api_client = ApiClient(configuration=_oauth_api_config)
        _oauth_api_client.set_default_header("X-Pinecone-Api-Version", API_VERSION)
        for key, value in additional_headers.items():
            _oauth_api_client.set_default_header(key, value)
        _oauth_api_client.user_agent = get_user_agent(Config())

        _oauth_api = OAuthApi(_oauth_api_client)
        token_request = TokenRequest(
            client_id=self._client_id,
            client_secret=self._client_secret,
            grant_type="client_credentials",
            audience="https://api.pinecone.io/",
        )
        token_response = _oauth_api.get_token(token_request)
        self._token = token_response.access_token

        _child_api_config = deepcopy(_oauth_api_config)
        _child_api_config.host = "https://api.pinecone.io"
        _child_api_config.api_key_prefix = {"BearerAuth": "Bearer"}
        _child_api_config.api_key = {"BearerAuth": self._token}

        self._child_api_client = ApiClient(configuration=_child_api_config)
        self._child_api_client.set_default_header("X-Pinecone-Api-Version", API_VERSION)
        for key, value in additional_headers.items():
            self._child_api_client.set_default_header(key, value)
        self._child_api_client.user_agent = get_user_agent(Config())

        # Lazily initialize resources
        from typing import TYPE_CHECKING

        if TYPE_CHECKING:
            from pinecone.admin.resources import (
                ProjectResource,
                ApiKeyResource,
                OrganizationResource,
            )

            self._project: ProjectResource | None = None
            self._api_key: ApiKeyResource | None = None
            self._organization: OrganizationResource | None = None
        else:
            self._project = None  # type: ignore[assignment]
            self._api_key = None  # type: ignore[assignment]
            self._organization = None  # type: ignore[assignment]

    @property
    def project(self):
        """A namespace for project-related operations

        Alias for :func:`projects`.

        To learn about all project-related operations, see :func:`pinecone.admin.resources.ProjectResource`.

        Examples
        --------

        .. code-block:: python
            :caption: Creating a project

            from pinecone import Admin

            # Using environment variables to pass PINECONE_CLIENT_ID and PINECONE_CLIENT_SECRET
            admin = Admin()

            # Create a project with no quota for pod indexes
            admin.project.create(
                name="my-project",
                max_pods=0
            )

        .. code-block:: python
            :caption: Listing all projects

            from pinecone import Admin

            admin = Admin()
            admin.projects.list()

        .. code-block:: python
            :caption: Deleting a project

            from pinecone import Admin

            admin = Admin()
            project = admin.project.get(name="my-project")
            admin.project.delete(project_id=project.id)

        """
        if self._project is None:
            from pinecone.admin.resources import ProjectResource

            self._project = ProjectResource(self._child_api_client)
        return self._project

    @property
    def projects(self):
        """Alias for :func:`project`"""
        return self.project

    @property
    def api_key(self):
        """A namespace for api key-related operations

        Alias for :func:`api_keys`.

        To learn about all api key-related operations, see :func:`pinecone.admin.resources.ApiKeyResource`.

        Examples
        --------

        .. code-block:: python
            :caption: Creating an API key

            from pinecone import Admin

            admin = Admin()

            project = admin.project.get(name="my-project")

            admin.api_key.create(
                name="my-api-key",
                project_id=project.id,
                description="my-api-key-description",
                roles=["ProjectEditor"]
            )

        .. code-block:: python
            :caption: Listing all API keys for a project

            from pinecone import Admin

            admin = Admin()
            project = admin.project.get(name="my-project")
            admin.api_key.list(project_id=project.id)

        .. code-block:: python
            :caption: Deleting an API key

            from pinecone import Admin

            admin = Admin()
            project = admin.project.get(name="my-project")

            # List api keys for the project
            keys_list = admin.api_key.list(project_id=project.id)

            # Delete the first api key in the list
            admin.api_key.delete(api_key_id=keys_list[0].id)

        """
        if self._api_key is None:
            from pinecone.admin.resources import ApiKeyResource

            self._api_key = ApiKeyResource(self._child_api_client)
        return self._api_key

    @property
    def api_keys(self):
        """Alias for :func:`api_key`"""
        return self.api_key

    @property
    def organization(self):
        """A namespace for organization-related operations

        Alias for :func:`organizations`.

        To learn about all organization-related operations, see :func:`pinecone.admin.resources.OrganizationResource`.

        Examples
        --------

        .. code-block:: python
            :caption: Listing all organizations

            from pinecone import Admin

            # Using environment variables to pass PINECONE_CLIENT_ID and PINECONE_CLIENT_SECRET
            admin = Admin()

            # List all organizations
            organizations_response = admin.organization.list()
            for org in organizations_response.data:
                print(org.id)
                print(org.name)

        .. code-block:: python
            :caption: Fetching an organization

            from pinecone import Admin

            admin = Admin()
            organization = admin.organization.get(organization_id="my-organization-id")
            print(organization.name)
            print(organization.plan)

        .. code-block:: python
            :caption: Updating an organization

            from pinecone import Admin

            admin = Admin()
            organization = admin.organization.update(
                organization_id="my-organization-id",
                name="updated-organization-name"
            )
            print(organization.name)

        """
        if self._organization is None:
            from pinecone.admin.resources import OrganizationResource

            self._organization = OrganizationResource(self._child_api_client)
        return self._organization

    @property
    def organizations(self):
        """Alias for :func:`organization`"""
        return self.organization
