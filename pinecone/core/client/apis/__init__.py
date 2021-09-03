#
# Copyright (c) 2020-2021 Pinecone Systems Inc. All right reserved.
#

# flake8: noqa

# Import all APIs into this package.
# If you have many APIs here with many many models used in each API this may
# raise a `RecursionError`.
# In order to avoid this, import only the API that you directly need like:
#
#   from .api.database_service_api import DatabaseServiceApi
#
# or import this package, but before doing it, use:
#
#   import sys
#   sys.setrecursionlimit(n)

# Import APIs into API package:
from pinecone.core.client.api.database_service_api import DatabaseServiceApi
from pinecone.core.client.api.vector_service_api import VectorServiceApi
