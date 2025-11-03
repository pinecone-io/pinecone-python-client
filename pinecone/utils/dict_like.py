"""
Dictionary-like object that supports both square bracket and dot notation access.

This utility class allows you to access dictionary values using either:
- Square bracket notation: obj['key']
- Dot notation: obj.key

It also supports nested access for dictionaries within the structure.
"""

from typing import Any, Dict, Union, Iterator, KeysView, ValuesView, ItemsView, MutableMapping


class DictLike(MutableMapping[str, Any]):
    """
    A dictionary-like object that supports both square bracket and dot notation access.

    This class wraps a dictionary and provides attribute-style access to its keys.
    Nested dictionaries are automatically wrapped in DictLike objects as well.

    Example:
        >>> data = {'name': 'test', 'config': {'timeout': 30, 'retries': 3}}
        >>> obj = DictLike(data)
        >>> obj.name  # 'test'
        >>> obj['name']  # 'test'
        >>> obj.config.timeout  # 30
        >>> obj['config']['timeout']  # 30
    """

    def __init__(self, data: Dict[str, Any]):
        """
        Initialize the DictLike object with a dictionary.

        Args:
            data: The dictionary to wrap
        """
        self._data = data

    def __getattr__(self, key: str) -> Any:
        """
        Provide dot notation access to dictionary keys.

        Args:
            key: The key to access

        Returns:
            The value associated with the key, wrapped in DictLike if it's a dict

        Raises:
            AttributeError: If the key doesn't exist
        """
        if key.startswith("_"):
            # Don't interfere with private attributes
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

        if key not in self._data:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

        value = self._data[key]
        # Recursively wrap nested dictionaries
        if isinstance(value, dict):
            return DictLike(value)
        return value

    def __getitem__(self, key: str) -> Any:
        """
        Provide square bracket access to dictionary keys.

        Args:
            key: The key to access

        Returns:
            The value associated with the key, wrapped in DictLike if it's a dict

        Raises:
            KeyError: If the key doesn't exist
        """
        if key not in self._data:
            raise KeyError(key)

        value = self._data[key]
        # Recursively wrap nested dictionaries
        if isinstance(value, dict):
            return DictLike(value)
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        """
        Set attributes, distinguishing between internal attributes and data keys.

        Args:
            key: The key to set
            value: The value to set
        """
        if key.startswith("_"):
            # Set internal attributes normally
            super().__setattr__(key, value)
        else:
            # Set data keys
            if not hasattr(self, "_data"):
                super().__setattr__("_data", {})
            self._data[key] = value

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Set dictionary keys using square bracket notation.

        Args:
            key: The key to set
            value: The value to set
        """
        self._data[key] = value

    def __delattr__(self, key: str) -> None:
        """
        Delete attributes using dot notation.

        Args:
            key: The key to delete

        Raises:
            AttributeError: If the key doesn't exist
        """
        if key.startswith("_"):
            super().__delattr__(key)
        elif key in self._data:
            del self._data[key]
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

    def __delitem__(self, key: str) -> None:
        """
        Delete dictionary keys using square bracket notation.

        Args:
            key: The key to delete

        Raises:
            KeyError: If the key doesn't exist
        """
        if key not in self._data:
            raise KeyError(key)
        del self._data[key]

    def __contains__(self, key: str) -> bool:
        """
        Check if a key exists in the dictionary.

        Args:
            key: The key to check

        Returns:
            True if the key exists, False otherwise
        """
        return key in self._data

    def __iter__(self) -> Iterator[str]:
        """
        Iterate over the dictionary keys.

        Returns:
            An iterator over the keys
        """
        return iter(self._data)

    def __len__(self) -> int:
        """
        Get the number of items in the dictionary.

        Returns:
            The number of items
        """
        return len(self._data)

    def __repr__(self) -> str:
        """
        String representation of the DictLike object.

        Returns:
            A string representation showing the wrapped dictionary
        """
        return f"DictLike({self._data})"

    def __str__(self) -> str:
        """
        String representation of the DictLike object.

        Returns:
            A string representation of the wrapped dictionary
        """
        return str(self._data)

    def keys(self) -> KeysView[str]:
        """Return the keys of the dictionary."""
        return self._data.keys()

    def values(self) -> ValuesView[Any]:
        """Return the values of the dictionary."""
        return self._data.values()

    def items(self) -> ItemsView[str, Any]:
        """Return the items of the dictionary."""
        return self._data.items()

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value with a default if the key doesn't exist.

        Args:
            key: The key to get
            default: The default value if key doesn't exist

        Returns:
            The value or default
        """
        value = self._data.get(key, default)
        if isinstance(value, dict):
            return DictLike(value)
        return value

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the DictLike object back to a regular dictionary.

        Returns:
            The underlying dictionary
        """
        return self._data.copy()

    def update(self, other: Union[Dict[str, Any], "DictLike"] = None, **kwargs) -> None:
        """
        Update the dictionary with elements from another dictionary or iterable.

        Args:
            other: Another dictionary or DictLike object to update from
            **kwargs: Additional key-value pairs to update
        """
        if other is not None:
            if hasattr(other, "items"):
                # Handle both dict and DictLike objects
                for key, value in other.items():
                    self._data[key] = value
            else:
                # Handle iterable of key-value pairs
                for key, value in other:
                    self._data[key] = value

        # Update with keyword arguments
        for key, value in kwargs.items():
            self._data[key] = value

    def pop(self, key: str, default: Any = None) -> Any:
        """
        Remove and return a value from the dictionary.

        Args:
            key: The key to remove
            default: The default value if key doesn't exist

        Returns:
            The value that was removed, or default if key doesn't exist
        """
        return self._data.pop(key, default)

    def popitem(self) -> tuple[str, Any]:
        """
        Remove and return a (key, value) pair from the dictionary.

        Returns:
            A (key, value) pair

        Raises:
            KeyError: If the dictionary is empty
        """
        return self._data.popitem()

    def setdefault(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the dictionary, setting it to default if it doesn't exist.

        Args:
            key: The key to get/set
            default: The default value to set if key doesn't exist

        Returns:
            The value for the key
        """
        if key not in self._data:
            self._data[key] = default
        return self._data[key]

    def clear(self) -> None:
        """Remove all items from the dictionary."""
        self._data.clear()

    def copy(self) -> "DictLike":
        """
        Create a shallow copy of the DictLike object.

        Returns:
            A new DictLike object with a copy of the data
        """
        return DictLike(self._data.copy())

    def __eq__(self, other: Any) -> bool:
        """
        Check equality with another object.

        Args:
            other: The object to compare with

        Returns:
            True if equal, False otherwise
        """
        if isinstance(other, DictLike):
            return self._data == other._data
        elif isinstance(other, dict):
            return self._data == other
        return False

    def __ne__(self, other: Any) -> bool:
        """
        Check inequality with another object.

        Args:
            other: The object to compare with

        Returns:
            True if not equal, False otherwise
        """
        return not self.__eq__(other)

    def __hash__(self) -> int:
        """
        Hash the DictLike object.

        Note: This makes DictLike objects unhashable, which is consistent
        with regular dictionaries.

        Returns:
            The hash value

        Raises:
            TypeError: DictLike objects are unhashable
        """
        raise TypeError(f"unhashable type: '{self.__class__.__name__}'")
