from typing import Tuple, Any


def parse_non_empty_args(args: list[Tuple[str, Any]]) -> dict[str, Any]:
    return {arg_name: val for arg_name, val in args if val is not None}
