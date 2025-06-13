from typing import Optional, List
from pinecone.openapi_support import ApiClient
from pinecone.core.openapi.admin.apis import APIKeysApi
from pinecone.utils import require_kwargs, parse_non_empty_args
from pinecone.core.openapi.admin.models import CreateAPIKeyRequest


class ApiKeyResource:
    """
    This class is used to create, delete, list, and fetch API keys.

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
        return self._api_keys_api.list_api_keys(project_id=project_id)

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
        """Alias for :func:`fetch`"""
        return self.fetch(api_key_id=api_key_id)

    @require_kwargs
    def describe(self, api_key_id: str):
        """Alias for :func:`fetch`"""
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
        description: Optional[str] = None,
        roles: Optional[List[str]] = None,
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
        :type roles: Optional[List[str]]
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
