from typing import Dict, List

VectorDictMetadataValue = str | int | float | List[str] | List[int] | List[float]
VectorMetadataTypedDict = Dict[str, VectorDictMetadataValue]
