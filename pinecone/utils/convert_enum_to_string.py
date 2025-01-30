from typing import Union
from enum import Enum


def convert_enum_to_string(value: Union[Enum, str]) -> str:
    if isinstance(value, Enum):
        return value.value
    return value
