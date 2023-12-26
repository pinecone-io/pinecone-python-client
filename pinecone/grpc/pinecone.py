from ..control.pinecone import Pinecone
from ..config.config import ConfigBuilder
from .index_grpc import GRPCIndex
from typing import Optional

class PineconeGRPC(Pinecone):
    def Index(self, name: str, host: Optional[str] = None):
        if host is None:
            host = self.index_host_store.get_host(self.index_api, self.config, name)
        config = ConfigBuilder.build(api_key=self.config.api_key, host=host)
        return GRPCIndex(index_name=name, config=config)
