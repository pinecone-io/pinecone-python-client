from __future__ import annotations

import json
from typing import TYPE_CHECKING
from pinecone.core.openapi.db_control.model.backup_model import BackupModel as OpenAPIBackupModel
from pinecone.utils.repr_overrides import custom_serializer

if TYPE_CHECKING:
    from pinecone.core.openapi.db_control.model.backup_model_schema import BackupModelSchema


class BackupModel:
    """Represents a Pinecone backup configuration and status.

    The BackupModel describes the configuration and status of a Pinecone backup,
    including metadata about the source index, backup location, and schema
    configuration.
    """

    def __init__(self, backup: OpenAPIBackupModel):
        self._backup = backup

    @property
    def schema(self) -> "BackupModelSchema" | None:
        """Schema for the behavior of Pinecone's internal metadata index.

        This property defines which metadata fields are indexed and filterable
        in the backup. By default, all metadata is indexed. When ``schema`` is
        present, only fields which are present in the ``fields`` object with
        ``filterable: true`` are indexed.

        The schema is a map of metadata field names to their configuration,
        where each field configuration specifies whether the field is filterable.

        :type: BackupModelSchema, optional
        :returns: The metadata schema configuration, or None if not set.
        """
        return getattr(self._backup, "schema", None)

    def __getattr__(self, attr):
        return getattr(self._backup, attr)

    def __getitem__(self, key):
        return self.__getattr__(key)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=4, default=custom_serializer)

    def to_dict(self):
        return self._backup.to_dict()
