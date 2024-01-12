from .index_grpc import GRPCIndex
from .pinecone import PineconeGRPC

from pinecone.core.grpc.protos.vector_service_pb2 import (
    Vector as GRPCVector,
    QueryVector as GRPCQueryVector,
    SparseValues as GRPCSparseValues,
    Vector,
    QueryVector,
    SparseValues
)