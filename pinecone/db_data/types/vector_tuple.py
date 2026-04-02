from collections.abc import Sequence

from .vector_metadata_dict import VectorMetadataTypedDict

VectorTuple = tuple[str, Sequence[float]]
VectorTupleWithMetadata = tuple[str, Sequence[float], VectorMetadataTypedDict]
