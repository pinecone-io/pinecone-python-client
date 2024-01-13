from .index_grpc import GRPCIndex
from .pinecone import PineconeGRPC

from pinecone.core.grpc.protos.vector_service_pb2 import (
    Vector as GRPCVector,
    SparseValues as GRPCSparseValues,
    Vector,
    SparseValues
)