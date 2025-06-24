from ._deleteable_resource import _DeleteableResource
from pinecone import Pinecone


class _DeleteableBackup(_DeleteableResource):
    def __init__(self, pc: Pinecone):
        self.pc = pc

    def name(self):
        return "backup"

    def name_plural(self):
        return "backups"

    def delete(self, name):
        backup = self._get_backup_by_name(name)
        return self.pc.db.backup.delete(backup_id=backup.backup_id)

    def get_state(self, name):
        backup = self._get_backup_by_name(name)
        return backup.status

    def list(self):
        return self.pc.db.backup.list()

    def _get_backup_by_name(self, name):
        for backup in self.pc.db.backup.list():
            if backup.name == name:
                return backup
        raise Exception(f"Backup {name} not found")
