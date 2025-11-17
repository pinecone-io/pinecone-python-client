from typing import TYPE_CHECKING

from pinecone.core.openapi.db_control.api.manage_indexes_api import ManageIndexesApi
from pinecone.core.openapi.db_control.model.create_backup_request import CreateBackupRequest
from pinecone.db_control.models import BackupModel, BackupList
from pinecone.utils import parse_non_empty_args, require_kwargs, PluginAware

if TYPE_CHECKING:
    from pinecone.config import Config, OpenApiConfiguration


class BackupResource(PluginAware):
    def __init__(
        self,
        index_api: ManageIndexesApi,
        config: "Config",
        openapi_config: "OpenApiConfiguration",
        pool_threads: int,
    ):
        self._index_api = index_api
        """ :meta private: """

        self.config = config
        """ :meta private: """

        self._openapi_config = openapi_config
        """ :meta private: """

        self._pool_threads = pool_threads
        """ :meta private: """

        super().__init__()  # Initialize PluginAware

    @require_kwargs
    def list(
        self,
        *,
        index_name: str | None = None,
        limit: int | None = 10,
        pagination_token: str | None = None,
    ) -> BackupList:
        """
        List backups for an index or for the project.

        :param index_name: The name of the index to list backups for. If not provided, list all backups for the project.
        :type index_name: str, optional
        :param limit: The maximum number of backups to return.
        :type limit: int, optional
        :param pagination_token: The pagination token to use for the next page of backups.
        :type pagination_token: str, optional
        :return: A list of backups.
        :rtype: BackupList
        """
        if index_name is not None:
            args = parse_non_empty_args(
                [
                    ("index_name", index_name),
                    ("limit", limit),
                    ("pagination_token", pagination_token),
                ]
            )
            return BackupList(self._index_api.list_index_backups(**args))
        else:
            args = parse_non_empty_args([("limit", limit), ("pagination_token", pagination_token)])
            return BackupList(self._index_api.list_project_backups(**args))

    @require_kwargs
    def create(self, *, index_name: str, backup_name: str, description: str = "") -> BackupModel:
        """
        Create a backup for an index.

        :param index_name: The name of the index to create a backup for.
        :type index_name: str
        :param backup_name: The name of the backup to create.
        :type backup_name: str
        :param description: The description of the backup.
        :type description: str, optional
        :return: The created backup.
        :rtype: BackupModel
        """
        req = CreateBackupRequest(name=backup_name, description=description)
        return BackupModel(
            self._index_api.create_backup(index_name=index_name, create_backup_request=req)
        )

    @require_kwargs
    def describe(self, *, backup_id: str) -> BackupModel:
        """
        Describe a backup.

        :param backup_id: The ID of the backup to describe.
        :type backup_id: str
        :return: The described backup.
        :rtype: BackupModel
        """
        return BackupModel(self._index_api.describe_backup(backup_id=backup_id))

    @require_kwargs
    def get(self, *, backup_id: str) -> BackupModel:
        """Alias for describe"""
        return self.describe(backup_id=backup_id)

    @require_kwargs
    def delete(self, *, backup_id: str) -> None:
        """
        Delete a backup.

        :param backup_id: The ID of the backup to delete.
        :type backup_id: str
        """
        self._index_api.delete_backup(backup_id=backup_id)
        return None
