from pinecone import Pinecone
from ._deleteable_resource import _DeleteableResource


class _DeleteableIndex(_DeleteableResource):
    def __init__(self, pc: Pinecone):
        self.pc = pc

    def name(self):
        return "index"

    def name_plural(self):
        return "indexes"

    def delete(self, name):
        return self.pc.db.index.delete(name=name)

    def get_state(self, name):
        desc = self.pc.db.index.describe(name=name)
        return desc["status"]["state"]

    def list(self):
        return self.pc.db.index.list()
