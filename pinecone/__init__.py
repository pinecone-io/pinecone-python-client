"""
.. include:: ../README.md
"""
from .config.config import *
from .exceptions import *
from .control.pinecone import Pinecone
from .index import *

try:
    from .grpc.index_grpc import *
except ImportError:
    pass  # ignore for non-[grpc] installations
