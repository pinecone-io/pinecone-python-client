from __future__ import annotations

from typing import List
from pinecone.openapi_support import ApiClient
from pinecone.core.openapi.admin.apis import APIKeysApi
from pinecone.utils import require_kwargs, parse_non_empty_args
from pinecone.core.openapi.admin.models import CreateAPIKeyRequest, UpdateAPIKeyRequest


class ApiKeyResource:
    """
    This class is used to create, delete, list, fetch, and update API keys.

    .. note::
        The class should not be instantiated directly. Instead, access this classes
        methods through the :class:`pinecone.Admin` class's
        :attr:`api_key` or :attr:`api_keys` attributes.

        .. code-block:: python

            from pinecone import Admin

            admin = Admin()

            project = admin.project.get(name='my-project-name')
            api_keys = admin.api_keys.list(project_id=project.id)
    """

    def __init__(self, api_client: ApiClient):
        self._api_keys_api = APIKeysApi(api_client=api_client)

    @require_kwargs
    def list(self, project_id: str):
        """
        List all API keys for a project.

        To find the ``project_id`` for your project, use
        :func:`pinecone.admin.resources.ProjectResource.list`
        or :func:`pinecone.admin.resources.ProjectResource.get`.

        The value of the API key is not returned. The value is only returned
        when a new API key is being created.

        :param project_id: The project_id of the project to list API keys for.
        :type project_id: str
        :return: An object with a list of API keys.
        :rtype: {"data": [APIKey]}

        Examples
        --------

        .. code-block:: python
            :caption: List all API keys for a project
            :emphasize-lines: 9

            from pinecone import Admin

            # Credentials read from PINECONE_CLIENT_ID and
            # PINECONE_CLIENT_SECRET environment variables
            admin = Admin()

            project = admin.project.get(name='my-project-name')

            api_keys = admin.api_key.list(project_id=project.id)
            for api_key in api_keys.data:
                print(api_key.id)
                print(api_key.name)
                print(api_key.description)
                print(api_key.roles)

        """
        return self._api_keys_api.list_project_api_keys(project_id=project_id)

    @require_kwargs
    def fetch(self, api_key_id: str):
        """
        Fetch an API key by ``api_key_id``.

        The value of the API key is not returned. The value is only returned
        when a new API key is being created.

        :param api_key_id: The id of the API key to fetch.
        :type api_key_id: str
        :return: The API key.
        :rtype: APIKey

        Examples
        --------

        .. code-block:: python
            :caption: Fetch an API key by ``api_key_id``
            :emphasize-lines: 7

            from pinecone import Admin

            # Credentials read from PINECONE_CLIENT_ID and
            # PINECONE_CLIENT_SECRET environment variables
            admin = Admin()

            api_key = admin.api_key.fetch(api_key_id='my-api-key-id')
            print(api_key.id)
            print(api_key.name)
            print(api_key.description)
            print(api_key.roles)
            print(api_key.created_at)

        """
        return self._api_keys_api.fetch_api_key(api_key_id=api_key_id)

    @require_kwargs
    def get(self, api_key_id: str):
        """Alias for :func:`fetch`

        Examples
        --------

        .. code-block:: python
            :caption: Get an API key by api_key_id

            from pinecone import Admin

            # Credentials read from PINECONE_CLIENT_ID and
            # PINECONE_CLIENT_SECRET environment variables
            admin = Admin()

            api_key = admin.api_key.get(api_key_id='my-api-key-id')
            print(api_key.id)
            print(api_key.name)
            print(api_key.description)
            print(api_key.roles)
            print(api_key.created_at)

        """
        return self.fetch(api_key_id=api_key_id)

    @require_kwargs
    def describe(self, api_key_id: str):
        """Alias for :func:`fetch`

        Examples
        --------

        .. code-block:: python
            :caption: Describe an API key by api_key_id

            from pinecone import Admin

            # Credentials read from PINECONE_CLIENT_ID and
            # PINECONE_CLIENT_SECRET environment variables
            admin = Admin()

            api_key = admin.api_key.describe(api_key_id='my-api-key-id')
            print(api_key.id)
            print(api_key.name)
            print(api_key.description)
            print(api_key.roles)
            print(api_key.created_at)

        """
        return self.fetch(api_key_id=api_key_id)

    @require_kwargs
    def delete(self, api_key_id: str):
        """
        Delete an API key by ``api_key_id``.

        :param api_key_id: The id of the API key to delete.
        :type api_key_id: str
        :return: ``None``

        Examples
        --------

        .. code-block:: python
            :caption: Delete an API key by api_key_id
            :emphasize-lines: 7

            from pinecone import Admin

            # Credentials read from PINECONE_CLIENT_ID and
            # PINECONE_CLIENT_SECRET environment variables
            admin = Admin()

            admin.api_key.delete(api_key_id='my-api-key-id')

            try:
                admin.api_key.fetch(api_key_id='my-api-key-id')
            except NotFoundException:
                print("API key deleted successfully")

        """
        return self._api_keys_api.delete_api_key(api_key_id=api_key_id)

    @require_kwargs
    def create(
        self,
        project_id: str,
        name: str,
        description: str | None = None,
        roles: List[str] | None = None,
    ):
        """
        Create an API key for a project.

        The value of the API key is returned in the create response.
        This is the only time the value is returned.

        :param project_id: The project_id of the project to create the API key for.
        :type project_id: str
        :param name: The name of the API key.
        :type name: str
        :param description: The description of the API key.
        :type description: Optional[str]
        :param roles: The roles of the API key. Available roles include:
            ``ProjectEditor``, ``ProjectViewer``, ``ControlPlaneEditor``,
            ``ControlPlaneViewer``, ``DataPlaneEditor``, ``DataPlaneViewer``
        :type roles: Optional[list[str]]
        :return: The created API key object and value.
        :rtype: {"key": APIKey, "value": str}

        Examples
        --------

        .. code-block:: python
            :caption: Create an API key for a project
            :emphasize-lines: 9-14

            from pinecone import Admin

            # Credentials read from PINECONE_CLIENT_ID and
            # PINECONE_CLIENT_SECRET environment variables
            admin = Admin()

            project = admin.project.get(name='my-project-name')

            api_key_response = admin.api_key.create(
                project_id=project.id,
                name='ci-key',
                description='Key for CI testing',
                roles=['ProjectEditor']
            )
            api_key = api_key_response.key
            print(api_key.id)
            print(api_key.name)
            print(api_key.description)
            print(api_key.roles)

            api_key_value = api_key_response.value
            print(api_key_value)

        """
        args = [("name", name), ("description", description), ("roles", roles)]
        create_api_key_request = CreateAPIKeyRequest(**parse_non_empty_args(args))
        return self._api_keys_api.create_api_key(
            project_id=project_id, create_api_key_request=create_api_key_request
        )

    @require_kwargs
    def update(self, api_key_id: str, name: str | None = None, roles: List[str] | None = None):
        """
        Update an API key.

        :param api_key_id: The id of the API key to update.
        :type api_key_id: str
        :param name: A new name for the API key. The name must be 1-80 characters long.
            If omitted, the name will not be updated.
        :type name: Optional[str]
        :param roles: A new set of roles for the API key. Available roles include:
            ``ProjectEditor``, ``ProjectViewer``, ``ControlPlaneEditor``,
            ``ControlPlaneViewer``, ``DataPlaneEditor``, ``DataPlaneViewer``.
            Existing roles will be removed if not included. If this field is omitted,
            the roles will not be updated.
        :type roles: Optional[list[str]]
        :return: The updated API key.
        :rtype: APIKey

        Examples
        --------

        .. code-block:: python
            :caption: Update an API key's name
            :emphasize-lines: 7-10

            from pinecone import Admin

            # Credentials read from PINECONE_CLIENT_ID and
            # PINECONE_CLIENT_SECRET environment variables
            admin = Admin()

            api_key = admin.api_key.update(
                api_key_id='my-api-key-id',
                name='updated-api-key-name'
            )
            print(api_key.name)

        .. code-block:: python
            :caption: Update an API key's roles
            :emphasize-lines: 7-10

            from pinecone import Admin

            admin = Admin()

            api_key = admin.api_key.update(
                api_key_id='my-api-key-id',
                roles=['ProjectViewer']
            )
            print(api_key.roles)

        .. code-block:: python
            :caption: Update both name and roles
            :emphasize-lines: 7-12

            from pinecone import Admin

            admin = Admin()

            api_key = admin.api_key.update(
                api_key_id='my-api-key-id',
                name='updated-name',
                roles=['ProjectEditor', 'DataPlaneEditor']
            )
            print(api_key.name)
            print(api_key.roles)

        """
        args = [("name", name), ("roles", roles)]
        update_request = UpdateAPIKeyRequest(**parse_non_empty_args(args))
        return self._api_keys_api.update_api_key(
            api_key_id=api_key_id, update_api_key_request=update_request
        )
