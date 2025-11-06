class DictLike:
    def __getitem__(self, key):
        if key in self.__dataclass_fields__:
            return getattr(self, key)
        raise KeyError(f"{key} is not a valid field")

    def __setitem__(self, key, value):
        if key in self.__dataclass_fields__:
            setattr(self, key, value)
        else:
            raise KeyError(f"{key} is not a valid field")

    def get(self, key, default=None):
        """Dict-like get method for compatibility with tests that use .get()"""
        try:
            return self[key]
        except KeyError:
            return default
