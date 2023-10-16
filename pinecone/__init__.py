"""
.. include:: ../README.md
"""
from .config import *
from .exceptions import *
from .manage import *
from .index import *

try:
    from .grpc.index_grpc import *
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
