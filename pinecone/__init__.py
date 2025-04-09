"""
.. include:: ../pdoc/README.md
"""

from .deprecated_plugins import check_for_deprecated_plugins
from .deprecation_warnings import *
from .pinecone import Pinecone
from .pinecone_asyncio import PineconeAsyncio
from .exceptions import *
# from .config import *
# from .db_control import *
# from .db_data import *

from .utils import __version__

import logging

# Raise an exception if the user is attempting to use the SDK with
# deprecated plugins installed in their project.
check_for_deprecated_plugins()

# Silence annoying log messages from the plugin interface
logging.getLogger("pinecone_plugin_interface").setLevel(logging.CRITICAL)
