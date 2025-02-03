"""
.. include:: ../README.md
"""

from .deprecated_plugins import check_for_deprecated_plugins
from .deprecation_warnings import *
from .config import *
from .exceptions import *
from .control import *
from .data import *
from .models import *
from .enums import *

from .utils import __version__

import logging

# Raise an exception if the user is attempting to use the SDK with
# deprecated plugins installed in their project.
check_for_deprecated_plugins()

# Silence annoying log messages from the plugin interface
logging.getLogger("pinecone_plugin_interface").setLevel(logging.CRITICAL)
