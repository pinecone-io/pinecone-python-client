from pinecone.core.openapi.db_control.models import IndexModel as OpenAPIIndexModel


class IndexModel:
    def __init__(self, index: OpenAPIIndexModel):
        self.index = index
        self.deletion_protection = index.deletion_protection.value

    def __str__(self):
        return str(self.index)

    def __getattr__(self, attr):
        return getattr(self.index, attr)

    def __getitem__(self, key):
        return self.__getattr__(key)

    def to_dict(self):
        return self.index.to_dict()
