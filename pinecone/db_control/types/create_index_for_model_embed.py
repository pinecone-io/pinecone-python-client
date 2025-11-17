from typing import TypedDict, Dict
from pinecone.db_control.enums import Metric
from pinecone.inference import EmbedModel


class CreateIndexForModelEmbedTypedDict(TypedDict):
    model: EmbedModel | str
    field_map: Dict
    metric: Metric | str
    read_parameters: Dict
    write_parameters: Dict
