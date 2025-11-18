from typing import TypedDict, Any


class ConfigureIndexEmbed(TypedDict):
    model: str
    field_map: dict[str, str]
    read_parameters: dict[str, Any] | None
    write_parameters: dict[str, Any] | None
