from pinecone.core.openapi.db_control.model.backup_model import BackupModel as OpenAPIBackupModel


class BackupModel:
    def __init__(self, backup: OpenAPIBackupModel):
        self._backup = backup

    def __str__(self):
        return str(self._backup)

    def __getattr__(self, attr):
        return getattr(self._backup, attr)

    def __getitem__(self, key):
        return self.__getattr__(key)

    def to_dict(self):
        return self._backup.to_dict()
