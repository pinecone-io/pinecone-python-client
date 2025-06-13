from ._deleteable_resource import _DeleteableResource
from pinecone import Pinecone


class _DeleteableCollection(_DeleteableResource):
    def __init__(self, pc: Pinecone):
        self.pc = pc

    def name(self):
        return "collection"

    def name_plural(self):
        return "collections"

    def get_state(self, name):
        desc = self.pc.db.collection.describe(name=name)
        return desc["status"]

    def delete(self, name):
        return self.pc.db.collection.delete(name=name)

    def list(self):
        return self.pc.db.collection.list()
