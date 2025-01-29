from typing import TypedDict, Dict, Union
from ...enums import Metric
from ...data.features.inference import EmbedModel


class CreateIndexForModelEmbedTypedDict(TypedDict):
    model: Union[EmbedModel, str]
    field_map: Dict
    metric: Union[Metric, str]
    read_parameters: Dict
    write_parameters: Dict
