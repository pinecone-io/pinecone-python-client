from .vector_metadata_dict import VectorMetadataTypedDict

VectorTuple = tuple[str, list[float]]
VectorTupleWithMetadata = tuple[str, list[float], VectorMetadataTypedDict]
