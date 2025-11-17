from pinecone.openapi_support import ApiClient
from pinecone.core.openapi.admin.apis import OrganizationsApi
from pinecone.utils import require_kwargs, parse_non_empty_args
from pinecone.core.openapi.admin.models import UpdateOrganizationRequest


class OrganizationResource:
    """
    This class is used to list, fetch, update, and delete organizations.

    .. note::
        The class should not be instantiated directly. Instead, access this classes
        methods through the :class:`pinecone.Admin` class's
        :attr:`organization` or :attr:`organizations` attributes.

        .. code-block:: python

            from pinecone import Admin

            admin = Admin()
            organization = admin.organization.get(organization_id="my-organization-id")
    """

    def __init__(self, api_client: ApiClient):
        """
        Initialize the OrganizationResource.

        .. warning::
            This class should not be instantiated directly. Instead, access this classes
            methods through the :class:`pinecone.Admin` class's
            :attr:`organization` or :attr:`organizations` attributes.

        :param api_client: The API client to use.
        :type api_client: ApiClient
        """
        self._organizations_api = OrganizationsApi(api_client=api_client)
        self._api_client = api_client

    @require_kwargs
    def list(self):
        """
        List all organizations associated with the account.

        :return: An object with a list of organizations.
        :rtype: {"data": [Organization]}

        Examples
        --------

        .. code-block:: python
            :caption: List all organizations
            :emphasize-lines: 8

            from pinecone import Admin

            # Credentials read from PINECONE_CLIENT_ID and
            # PINECONE_CLIENT_SECRET environment variables
            admin = Admin()

            # List all organizations
            organizations_response = admin.organization.list()
            for organization in organizations_response.data:
                print(organization.id)
                print(organization.name)
                print(organization.plan)
                print(organization.payment_status)

        """
        return self._organizations_api.list_organizations()

    @require_kwargs
    def fetch(self, organization_id: str):
        """
        Fetch an organization by organization_id.

        :param organization_id: The organization_id of the organization to fetch.
        :type organization_id: str
        :return: The organization.
        :rtype: Organization

        Examples
        --------

        .. code-block:: python
            :caption: Fetch an organization by organization_id
            :emphasize-lines: 7-9

            from pinecone import Admin

            # Credentials read from PINECONE_CLIENT_ID and
            # PINECONE_CLIENT_SECRET environment variables
            admin = Admin()

            organization = admin.organization.fetch(
                organization_id="42ca341d-43bf-47cb-9f27-e645dbfabea6"
            )
            print(organization.id)
            print(organization.name)
            print(organization.plan)
            print(organization.payment_status)
            print(organization.created_at)
            print(organization.support_tier)

        """
        return self._organizations_api.fetch_organization(organization_id=organization_id)

    @require_kwargs
    def get(self, organization_id: str):
        """Alias for :func:`fetch`

        Examples
        --------

        .. code-block:: python
            :caption: Get an organization by organization_id
            :emphasize-lines: 7-9

            from pinecone import Admin

            # Credentials read from PINECONE_CLIENT_ID and
            # PINECONE_CLIENT_SECRET environment variables
            admin = Admin()

            organization = admin.organization.get(
                organization_id="42ca341d-43bf-47cb-9f27-e645dbfabea6"
            )
            print(organization.id)
            print(organization.name)

        """
        return self.fetch(organization_id=organization_id)

    @require_kwargs
    def describe(self, organization_id: str):
        """Alias for :func:`fetch`

        Examples
        --------

        .. code-block:: python
            :caption: Describe an organization by organization_id
            :emphasize-lines: 7-9

            from pinecone import Admin

            # Credentials read from PINECONE_CLIENT_ID and
            # PINECONE_CLIENT_SECRET environment variables
            admin = Admin()

            organization = admin.organization.describe(
                organization_id="42ca341d-43bf-47cb-9f27-e645dbfabea6"
            )
            print(organization.id)
            print(organization.name)

        """
        return self.fetch(organization_id=organization_id)

    @require_kwargs
    def update(self, organization_id: str, name: str | None = None):
        """
        Update an organization.

        :param organization_id: The organization_id of the organization to update.
        :type organization_id: str
        :param name: The new name for the organization. If omitted, the name will not be updated.
        :type name: Optional[str]
        :return: The updated organization.
        :rtype: Organization

        Examples
        --------

        .. code-block:: python
            :caption: Update an organization's name
            :emphasize-lines: 7-10

            from pinecone import Admin

            # Credentials read from PINECONE_CLIENT_ID and
            # PINECONE_CLIENT_SECRET environment variables
            admin = Admin()

            organization = admin.organization.update(
                organization_id="42ca341d-43bf-47cb-9f27-e645dbfabea6",
                name="updated-organization-name"
            )
            print(organization.name)

        """
        args = [("name", name)]
        update_request = UpdateOrganizationRequest(**parse_non_empty_args(args))
        return self._organizations_api.update_organization(
            organization_id=organization_id, update_organization_request=update_request
        )

    @require_kwargs
    def delete(self, organization_id: str):
        """
        Delete an organization by organization_id.

        .. warning::
            Deleting an organization is a permanent and irreversible operation.
            Please be very sure you want to delete the organization and everything
            associated with it before calling this function.

        Before deleting an organization, you must delete all projects (including indexes,
        assistants, backups, and collections) associated with the organization.

        :param organization_id: The organization_id of the organization to delete.
        :type organization_id: str
        :return: ``None``

        Examples
        --------

        .. code-block:: python
            :caption: Delete an organization by organization_id
            :emphasize-lines: 7-9

            from pinecone import Admin

            # Credentials read from PINECONE_CLIENT_ID and
            # PINECONE_CLIENT_SECRET environment variables
            admin = Admin()

            admin.organization.delete(
                organization_id="42ca341d-43bf-47cb-9f27-e645dbfabea6"
            )

        """
        return self._organizations_api.delete_organization(organization_id=organization_id)
