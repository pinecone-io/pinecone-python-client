from ..control.pinecone import Pinecone
from ..config.config import ConfigBuilder
from .index_grpc import GRPCIndex

class PineconeGRPC(Pinecone):
    def Index(self, name: str = '', host: str = ''):
        if name == '' and host == '':
            raise ValueError("Either name or host must be specified")

        if host != '':
            # Use host if it is provided
            config = ConfigBuilder.build(api_key=self.config.api_key, host=host)
            return GRPCIndex(index_name=name, config=config)

        if name != '':
            # Otherwise, get host url from describe_index using the index name
            index_host = self.index_host_store.get_host(self.index_api, self.config, name)
            config = ConfigBuilder.build(api_key=self.config.api_key, host=index_host)
            return GRPCIndex(index_name=name, config=config)
