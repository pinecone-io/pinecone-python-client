from typing import Any


def parse_non_empty_args(args: list[tuple[str, Any]]) -> dict[str, Any]:
    return {arg_name: val for arg_name, val in args if val is not None}
