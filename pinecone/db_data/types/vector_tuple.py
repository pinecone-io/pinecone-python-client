from .vector_metadata_dict import VectorMetadataTypedDict
from typing import Tuple

VectorTuple = Tuple[str, list[float]]
VectorTupleWithMetadata = Tuple[str, list[float], VectorMetadataTypedDict]
