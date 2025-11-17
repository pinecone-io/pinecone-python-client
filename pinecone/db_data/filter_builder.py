from typing import Any, cast
from .types.query_filter import FilterTypedDict, FieldValue, NumericFieldValue, SimpleFilter


class FilterBuilder:
    """
    A fluent builder for constructing Pinecone metadata filters.

    The FilterBuilder helps prevent common filter construction errors such as
    misspelled operator names or invalid filter structures. It supports all
    Pinecone filter operators and provides operator overloading for combining
    conditions with AND (``&``) and OR (``|``) logic.

    Examples:

    .. code-block:: python

        # Simple equality filter
        filter1 = FilterBuilder().eq("genre", "drama").build()
        # Returns: {"genre": "drama"}

        # Multiple conditions with AND using & operator
        filter2 = (FilterBuilder().eq("genre", "drama") &
                   FilterBuilder().gt("year", 2020)).build()
        # Returns: {"$and": [{"genre": "drama"}, {"year": {"$gt": 2020}}]}

        # Multiple conditions with OR using | operator
        filter3 = (FilterBuilder().eq("genre", "comedy") |
                   FilterBuilder().eq("genre", "drama")).build()
        # Returns: {"$or": [{"genre": "comedy"}, {"genre": "drama"}]}

        # Complex nested conditions
        filter4 = ((FilterBuilder().eq("genre", "drama") &
                    FilterBuilder().gt("year", 2020)) |
                   (FilterBuilder().eq("genre", "comedy") &
                    FilterBuilder().lt("year", 2000))).build()

        # Using $exists
        filter5 = FilterBuilder().exists("genre", True).build()
        # Returns: {"genre": {"$exists": True}}

    """

    def __init__(self, filter_dict: (SimpleFilter | dict[str, Any]) | None = None) -> None:
        """
        Initialize a FilterBuilder.

        Args:
            filter_dict: Optional initial filter dictionary. Used internally
                        for combining filters with operators.
        """
        self._filter: (SimpleFilter | dict[str, Any]) | None = filter_dict

    def eq(self, field: str, value: FieldValue) -> "FilterBuilder":
        """
        Add an equality condition.

        Matches records where the specified field equals the given value.

        Args:
            field: The metadata field name.
            value: The value to match. Can be str, int, float, or bool.

        Returns:
            A new FilterBuilder instance with this condition added.

        Examples:

        .. code-block:: python

            FilterBuilder().eq("genre", "drama").build()
            # Returns: {"genre": "drama"}
        """
        return FilterBuilder({field: value})

    def ne(self, field: str, value: FieldValue) -> "FilterBuilder":
        """
        Add a not-equal condition.

        Matches records where the specified field does not equal the given value.

        Args:
            field: The metadata field name.
            value: The value to exclude. Can be str, int, float, or bool.

        Returns:
            A new FilterBuilder instance with this condition added.

        Examples:

        .. code-block:: python

            FilterBuilder().ne("genre", "drama").build()
            # Returns: {"genre": {"$ne": "drama"}}
        """
        return FilterBuilder({field: {"$ne": value}})

    def gt(self, field: str, value: NumericFieldValue) -> "FilterBuilder":
        """
        Add a greater-than condition.

        Matches records where the specified numeric field is greater than
        the given value.

        Args:
            field: The metadata field name.
            value: The numeric value to compare against. Must be int or float.

        Returns:
            A new FilterBuilder instance with this condition added.

        Examples:

        .. code-block:: python

            FilterBuilder().gt("year", 2020).build()
            # Returns: {"year": {"$gt": 2020}}
        """
        return FilterBuilder({field: {"$gt": value}})

    def gte(self, field: str, value: NumericFieldValue) -> "FilterBuilder":
        """
        Add a greater-than-or-equal condition.

        Matches records where the specified numeric field is greater than
        or equal to the given value.

        Args:
            field: The metadata field name.
            value: The numeric value to compare against. Must be int or float.

        Returns:
            A new FilterBuilder instance with this condition added.

        Examples:

        .. code-block:: python

            FilterBuilder().gte("year", 2020).build()
            # Returns: {"year": {"$gte": 2020}}
        """
        return FilterBuilder({field: {"$gte": value}})

    def lt(self, field: str, value: NumericFieldValue) -> "FilterBuilder":
        """
        Add a less-than condition.

        Matches records where the specified numeric field is less than
        the given value.

        Args:
            field: The metadata field name.
            value: The numeric value to compare against. Must be int or float.

        Returns:
            A new FilterBuilder instance with this condition added.

        Examples:

        .. code-block:: python

            FilterBuilder().lt("year", 2000).build()
            # Returns: {"year": {"$lt": 2000}}
        """
        return FilterBuilder({field: {"$lt": value}})

    def lte(self, field: str, value: NumericFieldValue) -> "FilterBuilder":
        """
        Add a less-than-or-equal condition.

        Matches records where the specified numeric field is less than
        or equal to the given value.

        Args:
            field: The metadata field name.
            value: The numeric value to compare against. Must be int or float.

        Returns:
            A new FilterBuilder instance with this condition added.

        Examples:

        .. code-block:: python

            FilterBuilder().lte("year", 2000).build()
            # Returns: {"year": {"$lte": 2000}}
        """
        return FilterBuilder({field: {"$lte": value}})

    def in_(self, field: str, values: list[FieldValue]) -> "FilterBuilder":
        """
        Add an in-list condition.

        Matches records where the specified field's value is in the given list.

        Args:
            field: The metadata field name.
            values: List of values to match against. Each value can be
                   str, int, float, or bool.

        Returns:
            A new FilterBuilder instance with this condition added.

        Examples:

        .. code-block:: python

            FilterBuilder().in_("genre", ["comedy", "drama"]).build()
            # Returns: {"genre": {"$in": ["comedy", "drama"]}}
        """
        return FilterBuilder({field: {"$in": values}})

    def nin(self, field: str, values: list[FieldValue]) -> "FilterBuilder":
        """
        Add a not-in-list condition.

        Matches records where the specified field's value is not in the
        given list.

        Args:
            field: The metadata field name.
            values: List of values to exclude. Each value can be
                   str, int, float, or bool.

        Returns:
            A new FilterBuilder instance with this condition added.

        Examples:

        .. code-block:: python

            FilterBuilder().nin("genre", ["comedy", "drama"]).build()
            # Returns: {"genre": {"$nin": ["comedy", "drama"]}}
        """
        return FilterBuilder({field: {"$nin": values}})

    def exists(self, field: str, exists: bool) -> "FilterBuilder":
        """
        Add an exists condition.

        Matches records where the specified field exists (or does not exist)
        in the metadata.

        Args:
            field: The metadata field name.
            exists: True to match records where the field exists,
                   False to match records where the field does not exist.

        Returns:
            A new FilterBuilder instance with this condition added.

        Examples:

        .. code-block:: python

            FilterBuilder().exists("genre", True).build()
            # Returns: {"genre": {"$exists": True}}
        """
        return FilterBuilder({field: {"$exists": exists}})

    def __and__(self, other: "FilterBuilder") -> "FilterBuilder":
        """
        Combine two FilterBuilder instances with AND logic.

        This method is called when using the ``&`` operator between two
        FilterBuilder instances.

        Args:
            other: Another FilterBuilder instance to combine with.

        Returns:
            A new FilterBuilder instance combining both conditions with AND.

        Examples:

        .. code-block:: python

            (FilterBuilder().eq("genre", "drama") &
             FilterBuilder().gt("year", 2020)).build()
            # Returns: {"$and": [{"genre": "drama"}, {"year": {"$gt": 2020}}]}
        """
        left_condition = self._get_filter_condition()
        right_condition = other._get_filter_condition()

        # If both sides are already $and, merge their conditions
        left_has_and = isinstance(self._filter, dict) and "$and" in self._filter
        right_has_and = isinstance(other._filter, dict) and "$and" in other._filter

        if left_has_and and right_has_and:
            left_and_dict = cast(dict[str, list[Any]], self._filter)
            right_and_dict = cast(dict[str, list[Any]], other._filter)
            conditions = left_and_dict["$and"] + right_and_dict["$and"]
            return FilterBuilder({"$and": conditions})

        # If either side is already an $and, merge the conditions
        if left_has_and:
            and_dict = cast(dict[str, list[Any]], self._filter)
            conditions = and_dict["$and"] + [right_condition]
            return FilterBuilder({"$and": conditions})
        if right_has_and:
            and_dict = cast(dict[str, list[Any]], other._filter)
            conditions = [left_condition] + and_dict["$and"]
            return FilterBuilder({"$and": conditions})
        return FilterBuilder({"$and": [left_condition, right_condition]})

    def __or__(self, other: "FilterBuilder") -> "FilterBuilder":
        """
        Combine two FilterBuilder instances with OR logic.

        This method is called when using the ``|`` operator between two
        FilterBuilder instances.

        Args:
            other: Another FilterBuilder instance to combine with.

        Returns:
            A new FilterBuilder instance combining both conditions with OR.

        Examples:

        .. code-block:: python

            (FilterBuilder().eq("genre", "comedy") |
             FilterBuilder().eq("genre", "drama")).build()
            # Returns: {"$or": [{"genre": "comedy"}, {"genre": "drama"}]}
        """
        left_condition = self._get_filter_condition()
        right_condition = other._get_filter_condition()

        # If both sides are already $or, merge their conditions
        left_has_or = isinstance(self._filter, dict) and "$or" in self._filter
        right_has_or = isinstance(other._filter, dict) and "$or" in other._filter

        if left_has_or and right_has_or:
            left_or_dict = cast(dict[str, list[Any]], self._filter)
            right_or_dict = cast(dict[str, list[Any]], other._filter)
            conditions = left_or_dict["$or"] + right_or_dict["$or"]
            return FilterBuilder({"$or": conditions})

        # If either side is already an $or, merge the conditions
        if left_has_or:
            or_dict = cast(dict[str, list[Any]], self._filter)
            conditions = or_dict["$or"] + [right_condition]
            return FilterBuilder({"$or": conditions})
        if right_has_or:
            or_dict = cast(dict[str, list[Any]], other._filter)
            conditions = [left_condition] + or_dict["$or"]
            return FilterBuilder({"$or": conditions})
        return FilterBuilder({"$or": [left_condition, right_condition]})

    def _get_filter_condition(self) -> SimpleFilter | dict[str, Any]:
        """
        Get the filter condition representation of this builder.

        Returns either a SimpleFilter for single conditions, or the full
        $and/$or structure for compound filters. This allows nesting
        of $and/$or structures even though the type system doesn't
        perfectly support it.

        Returns:
            A filter condition (SimpleFilter or compound structure).
        """
        if self._filter is None:
            raise ValueError("FilterBuilder must have at least one condition")
        return self._filter

    def build(self) -> FilterTypedDict:
        """
        Build and return the final filter dictionary.

        Returns:
            A FilterTypedDict that can be used with Pinecone query methods.
            Note: The return type may be more permissive than FilterTypedDict
            to support nested $and/$or structures that Pinecone accepts.

        Raises:
            ValueError: If the builder has no conditions.

        Examples:

        .. code-block:: python

            filter_dict = FilterBuilder().eq("genre", "drama").build()
            index.query(vector=embedding, top_k=10, filter=filter_dict)
        """
        if self._filter is None:
            raise ValueError("FilterBuilder must have at least one condition")
        # Type cast to FilterTypedDict - the actual structure may support
        # nested $and/$or even though the type system doesn't fully capture it
        return self._filter
