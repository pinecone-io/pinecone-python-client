import warnings

from pinecone.db_data.features import *

warnings.warn(
    "The module at `pinecone.data.features` has moved to `pinecone.db_data.features`. "
    "Please update your imports. "
    "This warning will become an error in a future version of the Pinecone Python SDK.",
    DeprecationWarning,
)
