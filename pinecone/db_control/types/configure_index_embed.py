from typing import TypedDict, Dict, Any, Optional


class ConfigureIndexEmbed(TypedDict):
    model: str
    field_map: Dict[str, str]
    read_parameters: Optional[Dict[str, Any]]
    write_parameters: Optional[Dict[str, Any]]
