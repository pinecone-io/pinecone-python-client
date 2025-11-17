from typing import Any


class DictLike:
    def __getitem__(self, key: str) -> Any:
        if hasattr(self, "__dataclass_fields__") and key in getattr(
            self, "__dataclass_fields__", {}
        ):
            return getattr(self, key)
        raise KeyError(f"{key} is not a valid field")

    def __setitem__(self, key: str, value: Any) -> None:
        if hasattr(self, "__dataclass_fields__") and key in getattr(
            self, "__dataclass_fields__", {}
        ):
            setattr(self, key, value)
        else:
            raise KeyError(f"{key} is not a valid field")

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-like get method for compatibility with tests that use .get()"""
        try:
            return self[key]
        except KeyError:
            return default
