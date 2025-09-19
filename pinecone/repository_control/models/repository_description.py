from typing import NamedTuple, Dict, Literal


class ServerlessSpecDefinition(NamedTuple):
    cloud: str
    region: str


ServerlessKey = Literal["serverless"]
ServerlessSpec = Dict[ServerlessKey, ServerlessSpecDefinition]
