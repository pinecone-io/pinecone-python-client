from enum import Enum


def convert_enum_to_string(value: Enum | str) -> str:
    if isinstance(value, Enum):
        return str(value.value)
    return value
