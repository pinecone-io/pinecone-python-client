"""
.. include:: ../README.md
"""
from pinecone.core.utils.constants import CLIENT_VERSION as __version__
from .config import *
from .exceptions import *
from .info import *
from .manage import *
from .index import *

try:
    from .core.grpc.index_grpc import *
except ImportError:
    pass  # ignore for non-[grpc] installations

# Kept for backwards-compatibility
UpsertResult = None
"""@private"""
DeleteResult = None
"""@private"""
QueryResult = None
"""@private"""
FetchResult = None
"""@private"""
InfoResult = None
"""@private"""
