"""
.. include:: ../README.md
"""
import warnings
from tqdm import TqdmExperimentalWarning

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

from .deprecation_warnings import *
from .config import *
from .exceptions import *
from .control import *
from .data import *
from .models import *

from .core.client.models import (
    IndexModel,
)

from .utils import __version__


import pinecone.features

for name in pinecone.features.__all__:
    feat = getattr(pinecone.features, name)()
    feat.validate_generated()
    target = globals()[feat.target()]

    # Install methods on target
    for method_name, method in feat.methods().items():
        if hasattr(target, method_name):
            raise ValueError(f"Method {method_name} already exists on target {target}. Cannot install the feature method from {name}.")
        setattr(target, method_name, method)