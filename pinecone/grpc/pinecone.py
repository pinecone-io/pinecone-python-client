from ..control.pinecone import Pinecone
from .index_grpc import GRPCIndex

class Pinecone(Pinecone):
    def Index(self, name: str):
        index_host = self.index_host_store.get_host(self.index_api, self.config, name)
        return GRPCIndex(api_key=self.config.API_KEY, host=index_host)
