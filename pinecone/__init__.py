"""
.. include:: ../README.md
"""
from .config import *
from .exceptions import *
from .control.pinecone import Pinecone
from .data.index import *

try:
    from .data.grpc.index_grpc import *
except ImportError:
    pass  # ignore for non-[grpc] installations
