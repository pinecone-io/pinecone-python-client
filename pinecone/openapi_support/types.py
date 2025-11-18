from typing import TypedDict


class PropertyValidationTypedDict(TypedDict, total=False):
    max_length: int
    min_length: int
    max_items: int
    min_items: int
    exclusive_maximum: int | float
    inclusive_maximum: int | float
    exclusive_minimum: int | float
    inclusive_minimum: int | float
    regex: dict[str, str]
    multiple_of: int
