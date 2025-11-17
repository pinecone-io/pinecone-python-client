from typing import TypedDict, Dict, Any


class ConfigureIndexEmbed(TypedDict):
    model: str
    field_map: Dict[str, str]
    read_parameters: Dict[str, Any] | None
    write_parameters: Dict[str, Any] | None
