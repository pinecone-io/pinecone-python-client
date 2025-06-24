from typing import Optional
from pinecone.exceptions import NotFoundException, PineconeException
from pinecone.openapi_support import ApiClient
from pinecone.core.openapi.admin.apis import ProjectsApi
from pinecone.utils import parse_non_empty_args, require_kwargs
from pinecone.core.openapi.admin.models import CreateProjectRequest, UpdateProjectRequest
import logging
import time

logger = logging.getLogger(__name__)


class ProjectResource:
    """
    This class is used to create, delete, list, fetch, and update projects.

    .. note::
        The class should not be instantiated directly. Instead, access this classes
        methods through the :class:`pinecone.Admin` class's
        :attr:`project` or :attr:`projects` attributes.

        .. code-block:: python

            from pinecone import Admin

            # Credentials read from PINECONE_CLIENT_ID and
            # PINECONE_CLIENT_SECRET environment variables
            admin = Admin()

            # Now call project methods on the projects namespace
            project = admin.projects.create(
                name="my-project",
                max_pods=10,
                force_encryption_with_cmek=False
            )
    """

    def __init__(self, api_client: ApiClient):
        """
        Initialize the ProjectResource.

        .. warning::
            This class should not be instantiated directly. Instead, access this classes
            methods through the :class:`pinecone.Admin` class's
            :attr:`project` or :attr:`projects` attributes.

        :param api_client: The API client to use.
        :type api_client: ApiClient
        """
        self._projects_api = ProjectsApi(api_client=api_client)
        self._api_client = api_client

    @require_kwargs
    def list(self):
        """
        List all projects in the organization.

        :return: An object with a list of projects.
        :rtype: {"data": [Project]}

        .. code-block:: python
            :caption: List all projects in the organization
            :emphasize-lines: 8

            from pinecone import Admin

            # Credentials read from PINECONE_CLIENT_ID and
            # PINECONE_CLIENT_SECRET environment variables
            admin = Admin()

            # List all projects in the organization
            projects_response = admin.projects.list()
            for project in projects_response.data:
                print(project.id)
                print(project.name)
                print(project.max_pods)
                print(project.force_encryption_with_cmek)
        """
        return self._projects_api.list_projects()

    @require_kwargs
    def fetch(self, project_id: Optional[str] = None, name: Optional[str] = None):
        """
        Fetch a project by project_id or name.

        :param project_id: The project_id of the project to fetch.
        :type project_id: str
        :param name: The name of the project to fetch.
        :type name: str
        :return: The project.
        :rtype: Project

        Examples
        --------

        .. code-block:: python
            :caption: Fetch a project by project_id
            :emphasize-lines: 7-9

            from pinecone import Admin

            # Credentials read from PINECONE_CLIENT_ID and
            # PINECONE_CLIENT_SECRET environment variables
            admin = Admin()

            project = admin.projects.fetch(
                project_id="42ca341d-43bf-47cb-9f27-e645dbfabea6"
            )
            print(project.id)
            print(project.name)
            print(project.max_pods)
            print(project.force_encryption_with_cmek)
            print(project.organization_id)
            print(project.created_at)

        .. code-block:: python
            :caption: Fetch a project by name
            :emphasize-lines: 7

            from pinecone import Admin

            # Credentials read from PINECONE_CLIENT_ID and
            # PINECONE_CLIENT_SECRET environment variables
            admin = Admin()

            project = admin.projects.fetch(name="my-project-name")
            print(project.id)
            print(project.name)
            print(project.max_pods)
            print(project.force_encryption_with_cmek)
            print(project.organization_id)
            print(project.created_at)
        """
        if project_id is not None and name is not None:
            raise ValueError("Either project_id or name must be provided but not both")
        elif project_id is None and name is None:
            raise ValueError("Either project_id or name must be provided")

        if project_id is not None:
            return self._projects_api.fetch_project(project_id=project_id)
        else:
            projects = self.list().data
            projects = [project for project in projects if project.name == name]
            if len(projects) == 0:
                raise NotFoundException(f"Project with name '{name}' not found")
            elif len(projects) > 1:
                ids = [project.id for project in projects]
                raise PineconeException(
                    f"Multiple projects found with name '{name}'. Please use project_id to fetch a specific project. Matching project ids: {ids}"
                )
            else:
                return projects[0]

    @require_kwargs
    def get(self, project_id: Optional[str] = None, name: Optional[str] = None):
        """Alias for :func:`fetch`

        Examples
        --------

        .. code-block:: python
            :caption: Get a project by project_id
            :emphasize-lines: 7-9

            from pinecone import Admin

            # Credentials read from PINECONE_CLIENT_ID and
            # PINECONE_CLIENT_SECRET environment variables
            admin = Admin()

            project = admin.project.get(
                project_id="42ca341d-43bf-47cb-9f27-e645dbfabea6"
            )
            print(project.id)
            print(project.name)
            print(project.max_pods)
            print(project.force_encryption_with_cmek)
        """
        return self.fetch(project_id=project_id, name=name)

    @require_kwargs
    def describe(self, project_id: Optional[str] = None, name: Optional[str] = None):
        """Alias for :func:`fetch`

        Examples
        --------

        .. code-block:: python
            :caption: Describe a project by project_id
            :emphasize-lines: 7-9

            from pinecone import Admin

            # Credentials read from PINECONE_CLIENT_ID and
            # PINECONE_CLIENT_SECRET environment variables
            admin = Admin()

            project = admin.project.describe(
                project_id="42ca341d-43bf-47cb-9f27-e645dbfabea6"
            )
            print(project.id)
            print(project.name)
            print(project.max_pods)
            print(project.force_encryption_with_cmek)
        """
        return self.fetch(project_id=project_id, name=name)

    @require_kwargs
    def exists(self, project_id: Optional[str] = None, name: Optional[str] = None):
        """
        Check if a project exists by project_id or name.

        :param project_id: The project_id of the project to check.
        :type project_id: str
        :param name: The name of the project to check.
        :type name: str
        :return: True if the project exists, False otherwise.
        :rtype: bool

        :raises ValueError: If both project_id and name are provided.

        Examples
        --------

        .. code-block:: python
            :caption: Check if a project exists by project name
            :emphasize-lines: 8

            from pinecone import Admin

            # Credentials read from PINECONE_CLIENT_ID and
            # PINECONE_CLIENT_SECRET environment variables
            admin = Admin()

            project_name = "my-project-name"
            if admin.project.exists(name=project_name):
                print(f"Project {project_name} exists")
            else:
                admin.project.create(
                    name=project_name,
                    max_pods=10,
                    force_encryption_with_cmek=False
                )

        .. code-block:: python
            :caption: Check if a project exists by project_id
            :emphasize-lines: 8

            from pinecone import Admin

            # Credentials read from PINECONE_CLIENT_ID and
            # PINECONE_CLIENT_SECRET environment variables
            admin = Admin()

            project_id = "42ca341d-43bf-47cb-9f27-e645dbfabea6"
            if admin.project.exists(project_id=project_id):
                print(f"Project {project_id} exists")
            else:
                print(f"Project {project_id} does not exist")
        """
        if project_id is not None and name is not None:
            raise ValueError("Either project_id or name must be provided but not both")
        elif project_id is None and name is None:
            raise ValueError("Either project_id or name must be provided")

        try:
            args = [("project_id", project_id), ("name", name)]
            self.fetch(**parse_non_empty_args(args))
            return True
        except NotFoundException:
            return False

    @require_kwargs
    def create(
        self,
        name: str,
        max_pods: Optional[int] = None,
        force_encryption_with_cmek: Optional[bool] = None,
    ):
        """
        Create a project.

        :param name: The name of the project to create.
        :type name: str
        :param max_pods: The maximum number of pods for the project.
        :type max_pods: int
        :param force_encryption_with_cmek: Whether to force encryption with CMEK.
        :type force_encryption_with_cmek: bool
        :return: The created project.
        :rtype: Project

        Examples
        --------

        .. code-block:: python
            :caption: Create a project
            :emphasize-lines: 7-11

            from pinecone import Admin

            # Credentials read from PINECONE_CLIENT_ID and
            # PINECONE_CLIENT_SECRET environment variables
            admin = Admin()

            project = admin.project.create(
                name="my-project-name",
                max_pods=10,
                force_encryption_with_cmek=False
            )

            print(project.id)
            print(project.name)
            print(project.organization_id)
            print(project.max_pods)
            print(project.force_encryption_with_cmek)
            print(project.created_at)

        """
        args = [
            ("name", name),
            ("max_pods", max_pods),
            ("force_encryption_with_cmek", force_encryption_with_cmek),
        ]
        create_request = CreateProjectRequest(**parse_non_empty_args(args))
        return self._projects_api.create_project(create_project_request=create_request)

    @require_kwargs
    def update(
        self,
        project_id: str,
        name: Optional[str] = None,
        max_pods: Optional[int] = None,
        force_encryption_with_cmek: Optional[bool] = None,
    ):
        """
        Update a project.

        :param project_id: The project_id of the project to update.
        :type project_id: str
        :param name: The name of the project to update.
        :type name: str
        :param max_pods: The maximum number of pods for the project.
        :type max_pods: int
        :param force_encryption_with_cmek: Whether to force encryption with CMEK.
        :type force_encryption_with_cmek: bool
        :return: The updated project.
        :rtype: Project

        Examples
        --------

        .. code-block:: python
            :caption: Update a project by project_id
            :emphasize-lines: 10-13, 16-19

            from pinecone import Admin

            # Credentials read from PINECONE_CLIENT_ID and
            # PINECONE_CLIENT_SECRET environment variables
            admin = Admin()

            project = admin.project.get(name='my-project-name')

            # Update max pods to 10
            project = admin.project.update(
                project_id=project.id,
                max_pods=10
            )

            # Update force_encryption_with_cmek to True
            project = admin.project.update(
                project_id=project.id,
                force_encryption_with_cmek=True
            )

        """
        args = [
            ("name", name),
            ("max_pods", max_pods),
            ("force_encryption_with_cmek", force_encryption_with_cmek),
        ]
        update_request = UpdateProjectRequest(**parse_non_empty_args(args))
        return self._projects_api.update_project(
            project_id=project_id, update_project_request=update_request
        )

    @require_kwargs
    def delete(
        self,
        project_id: str,
        delete_all_indexes: bool = False,
        delete_all_collections: bool = False,
        delete_all_backups: bool = False,
    ):
        """

        .. warning::
            Deleting a project is a permanent and irreversible operation.
            Please be very sure you want to delete the project and everything
            associated with it before calling this function.


        Projects can only be deleted if they are empty. The delete operation
        will fail if the project contains any resources such as indexes,
        collections, or backups.

        If you pass additional options such as ``delete_all_indexes=True``,
        ``delete_all_collections=True``, or ``delete_all_backups=True``, this function
        will attempt to delete all of these resources before deleting the project itself.
        **These deletions are permanent and cannot be undone.**

        :param project_id: The project_id of the project to delete.
        :type project_id: str
        :param delete_all_indexes: Attempt to delete all indexes associated with the project.
        :type delete_all_indexes: bool
        :param delete_all_collections: Attempt to delete all collections associated with the project.
        :type delete_all_collections: bool
        :param delete_all_backups: Attempt to delete all backups associated with the project.
        :type delete_all_backups: bool
        :return: ``None``

        Examples
        --------

        .. code-block:: python
            :caption: Delete a project by project_id
            :emphasize-lines: 9

            from pinecone import Admin

            # Credentials read from PINECONE_CLIENT_ID and
            # PINECONE_CLIENT_SECRET environment variables
            admin = Admin()

            project = admin.project.get(name='my-project-name')

            admin.project.delete(project_id=project.id)

        .. code-block:: python
            :caption: Delete a project that still contains indexes, collections, and backups
            :emphasize-lines: 7-12

            from pinecone import Admin

            admin = Admin()

            project = admin.project.get(name='my-project-name')

            admin.project.delete(
                project_id=project.id,
                delete_all_indexes=True,
                delete_all_collections=True,
                delete_all_backups=True
            )

            if not admin.project.exists(project_id=project.id):
                print("Project deleted successfully")
            else:
                print("Project deletion failed")
        """
        project = self.get(project_id=project_id)

        if not (delete_all_indexes or delete_all_collections or delete_all_backups):
            return self._projects_api.delete_project(project_id=project_id)

        from .api_key import ApiKeyResource

        api_key_resource = ApiKeyResource(self._api_client)
        logger.debug(f"Creating API key 'cleanup-key' for project {project.id}")
        key_create_response = api_key_resource.create(
            project_id=project.id, name="cleanup-key", roles=["ProjectEditor"]
        )
        api_key = key_create_response.value

        try:
            from ..eraser.project_eraser import _ProjectEraser

            done = False
            retries = 0

            while not done and retries < 5:
                project_eraser = _ProjectEraser(api_key=api_key)

                if delete_all_collections:
                    project_eraser.delete_all_collections()
                if delete_all_backups:
                    project_eraser.delete_all_backups()
                if delete_all_indexes:
                    project_eraser.delete_all_indexes()

                done = not project_eraser.retry_needed()
                retries += 1
                if not done:
                    logger.debug(
                        f"Retrying deletion of resources for project {project.id}. There were {len(project_eraser.undeleteable_resources)} undeleteable resources"
                    )
                    time.sleep(30)
        finally:
            logger.debug(f"Deleting API key 'cleanup-key' for project {project.id}")
            api_key_resource.delete(api_key_id=key_create_response.key.id)

        logger.info(f"Deleting project {project_id}")
        return self._projects_api.delete_project(project_id=project_id)
