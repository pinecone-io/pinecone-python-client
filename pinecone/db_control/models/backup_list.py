import json
from pinecone.core.openapi.db_control.model.backup_list import BackupList as OpenAPIBackupList
from .backup_model import BackupModel
from typing import List


class BackupList:
    def __init__(self, backup_list: OpenAPIBackupList):
        self._backup_list = backup_list
        self._backups = [BackupModel(b) for b in self._backup_list.data]

    def names(self) -> List[str]:
        return [i.name for i in self._backups]

    def __getitem__(self, key):
        return self.indexes[key]

    def __len__(self):
        return len(self._backups)

    def __iter__(self):
        return iter(self._backups)

    def __str__(self):
        return str(self._backups)

    def __repr__(self):
        return json.dumps([i.to_dict() for i in self._backups], indent=4)

    def __getattr__(self, attr):
        return getattr(self._backup_list, attr)
