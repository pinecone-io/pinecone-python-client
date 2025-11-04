import pytest
from pinecone.db_data.filter_builder import FilterBuilder


class TestFilterBuilderSimpleFilters:
    """Test simple single-condition filters."""

    def test_eq_string(self):
        """Test equality filter with string value."""
        result = FilterBuilder().eq("genre", "drama").build()
        assert result == {"genre": "drama"}

    def test_eq_int(self):
        """Test equality filter with integer value."""
        result = FilterBuilder().eq("year", 2020).build()
        assert result == {"year": 2020}

    def test_eq_float(self):
        """Test equality filter with float value."""
        result = FilterBuilder().eq("rating", 4.5).build()
        assert result == {"rating": 4.5}

    def test_eq_bool(self):
        """Test equality filter with boolean value."""
        result = FilterBuilder().eq("active", True).build()
        assert result == {"active": True}

    def test_ne_string(self):
        """Test not-equal filter with string value."""
        result = FilterBuilder().ne("genre", "comedy").build()
        assert result == {"genre": {"$ne": "comedy"}}

    def test_ne_int(self):
        """Test not-equal filter with integer value."""
        result = FilterBuilder().ne("year", 2019).build()
        assert result == {"year": {"$ne": 2019}}

    def test_gt_int(self):
        """Test greater-than filter with integer value."""
        result = FilterBuilder().gt("year", 2020).build()
        assert result == {"year": {"$gt": 2020}}

    def test_gt_float(self):
        """Test greater-than filter with float value."""
        result = FilterBuilder().gt("rating", 4.0).build()
        assert result == {"rating": {"$gt": 4.0}}

    def test_gte_int(self):
        """Test greater-than-or-equal filter with integer value."""
        result = FilterBuilder().gte("year", 2020).build()
        assert result == {"year": {"$gte": 2020}}

    def test_gte_float(self):
        """Test greater-than-or-equal filter with float value."""
        result = FilterBuilder().gte("rating", 4.5).build()
        assert result == {"rating": {"$gte": 4.5}}

    def test_lt_int(self):
        """Test less-than filter with integer value."""
        result = FilterBuilder().lt("year", 2000).build()
        assert result == {"year": {"$lt": 2000}}

    def test_lt_float(self):
        """Test less-than filter with float value."""
        result = FilterBuilder().lt("rating", 3.0).build()
        assert result == {"rating": {"$lt": 3.0}}

    def test_lte_int(self):
        """Test less-than-or-equal filter with integer value."""
        result = FilterBuilder().lte("year", 2000).build()
        assert result == {"year": {"$lte": 2000}}

    def test_lte_float(self):
        """Test less-than-or-equal filter with float value."""
        result = FilterBuilder().lte("rating", 3.5).build()
        assert result == {"rating": {"$lte": 3.5}}

    def test_in_strings(self):
        """Test in-list filter with string values."""
        result = FilterBuilder().in_("genre", ["comedy", "drama", "action"]).build()
        assert result == {"genre": {"$in": ["comedy", "drama", "action"]}}

    def test_in_ints(self):
        """Test in-list filter with integer values."""
        result = FilterBuilder().in_("year", [2019, 2020, 2021]).build()
        assert result == {"year": {"$in": [2019, 2020, 2021]}}

    def test_in_mixed(self):
        """Test in-list filter with mixed value types."""
        result = FilterBuilder().in_("value", ["string", 42, 3.14, True]).build()
        assert result == {"value": {"$in": ["string", 42, 3.14, True]}}

    def test_nin_strings(self):
        """Test not-in-list filter with string values."""
        result = FilterBuilder().nin("genre", ["comedy", "drama"]).build()
        assert result == {"genre": {"$nin": ["comedy", "drama"]}}

    def test_nin_ints(self):
        """Test not-in-list filter with integer values."""
        result = FilterBuilder().nin("year", [2019, 2020]).build()
        assert result == {"year": {"$nin": [2019, 2020]}}

    def test_exists_true(self):
        """Test exists filter with True."""
        result = FilterBuilder().exists("genre", True).build()
        assert result == {"genre": {"$exists": True}}

    def test_exists_false(self):
        """Test exists filter with False."""
        result = FilterBuilder().exists("genre", False).build()
        assert result == {"genre": {"$exists": False}}


class TestFilterBuilderAndOperator:
    """Test AND operator overloading."""

    def test_and_two_conditions(self):
        """Test combining two conditions with AND."""
        result = (FilterBuilder().eq("genre", "drama") & FilterBuilder().gt("year", 2020)).build()
        assert result == {"$and": [{"genre": "drama"}, {"year": {"$gt": 2020}}]}

    def test_and_three_conditions(self):
        """Test combining three conditions with AND."""
        f1 = FilterBuilder().eq("genre", "drama")
        f2 = FilterBuilder().gt("year", 2020)
        f3 = FilterBuilder().lt("rating", 5.0)
        result = ((f1 & f2) & f3).build()
        assert result == {
            "$and": [{"genre": "drama"}, {"year": {"$gt": 2020}}, {"rating": {"$lt": 5.0}}]
        }

    def test_and_merge_existing_and(self):
        """Test merging with existing $and structure."""
        f1 = FilterBuilder().eq("genre", "drama") & FilterBuilder().gt("year", 2020)
        f2 = FilterBuilder().lt("rating", 5.0)
        result = (f1 & f2).build()
        assert result == {
            "$and": [{"genre": "drama"}, {"year": {"$gt": 2020}}, {"rating": {"$lt": 5.0}}]
        }

    def test_and_merge_both_sides(self):
        """Test merging when both sides have $and."""
        f1 = FilterBuilder().eq("genre", "drama") & FilterBuilder().gt("year", 2020)
        f2 = FilterBuilder().lt("rating", 5.0) & FilterBuilder().exists("active", True)
        result = (f1 & f2).build()
        assert result == {
            "$and": [
                {"genre": "drama"},
                {"year": {"$gt": 2020}},
                {"rating": {"$lt": 5.0}},
                {"active": {"$exists": True}},
            ]
        }

    def test_and_chained(self):
        """Test chained AND operations."""
        result = (
            FilterBuilder().eq("genre", "drama")
            & FilterBuilder().gt("year", 2020)
            & FilterBuilder().lt("rating", 5.0)
        ).build()
        assert result == {
            "$and": [{"genre": "drama"}, {"year": {"$gt": 2020}}, {"rating": {"$lt": 5.0}}]
        }


class TestFilterBuilderOrOperator:
    """Test OR operator overloading."""

    def test_or_two_conditions(self):
        """Test combining two conditions with OR."""
        result = (
            FilterBuilder().eq("genre", "comedy") | FilterBuilder().eq("genre", "drama")
        ).build()
        assert result == {"$or": [{"genre": "comedy"}, {"genre": "drama"}]}

    def test_or_three_conditions(self):
        """Test combining three conditions with OR."""
        f1 = FilterBuilder().eq("genre", "comedy")
        f2 = FilterBuilder().eq("genre", "drama")
        f3 = FilterBuilder().eq("genre", "action")
        result = ((f1 | f2) | f3).build()
        assert result == {"$or": [{"genre": "comedy"}, {"genre": "drama"}, {"genre": "action"}]}

    def test_or_merge_existing_or(self):
        """Test merging with existing $or structure."""
        f1 = FilterBuilder().eq("genre", "comedy") | FilterBuilder().eq("genre", "drama")
        f2 = FilterBuilder().eq("genre", "action")
        result = (f1 | f2).build()
        assert result == {"$or": [{"genre": "comedy"}, {"genre": "drama"}, {"genre": "action"}]}

    def test_or_merge_both_sides(self):
        """Test merging when both sides have $or."""
        f1 = FilterBuilder().eq("genre", "comedy") | FilterBuilder().eq("genre", "drama")
        f2 = FilterBuilder().eq("genre", "action") | FilterBuilder().eq("genre", "thriller")
        result = (f1 | f2).build()
        assert result == {
            "$or": [
                {"genre": "comedy"},
                {"genre": "drama"},
                {"genre": "action"},
                {"genre": "thriller"},
            ]
        }

    def test_or_chained(self):
        """Test chained OR operations."""
        result = (
            FilterBuilder().eq("genre", "comedy")
            | FilterBuilder().eq("genre", "drama")
            | FilterBuilder().eq("genre", "action")
        ).build()
        assert result == {"$or": [{"genre": "comedy"}, {"genre": "drama"}, {"genre": "action"}]}


class TestFilterBuilderComplexNested:
    """Test complex nested filter structures."""

    def test_nested_and_or(self):
        """Test nested AND and OR operations."""
        # (genre == "drama" AND year > 2020) OR (genre == "comedy" AND year < 2000)
        result = (
            (FilterBuilder().eq("genre", "drama") & FilterBuilder().gt("year", 2020))
            | (FilterBuilder().eq("genre", "comedy") & FilterBuilder().lt("year", 2000))
        ).build()
        assert result == {
            "$or": [
                {"$and": [{"genre": "drama"}, {"year": {"$gt": 2020}}]},
                {"$and": [{"genre": "comedy"}, {"year": {"$lt": 2000}}]},
            ]
        }

    def test_nested_or_and(self):
        """Test nested OR and AND operations."""
        # (genre == "drama" OR genre == "comedy") AND year > 2020
        result = (
            (FilterBuilder().eq("genre", "drama") | FilterBuilder().eq("genre", "comedy"))
            & FilterBuilder().gt("year", 2020)
        ).build()
        assert result == {
            "$and": [{"$or": [{"genre": "drama"}, {"genre": "comedy"}]}, {"year": {"$gt": 2020}}]
        }

    def test_deeply_nested(self):
        """Test deeply nested filter structure."""
        # ((a AND b) OR (c AND d)) AND e
        a = FilterBuilder().eq("field1", "value1")
        b = FilterBuilder().eq("field2", "value2")
        c = FilterBuilder().eq("field3", "value3")
        d = FilterBuilder().eq("field4", "value4")
        e = FilterBuilder().eq("field5", "value5")

        result = (((a & b) | (c & d)) & e).build()
        assert result == {
            "$and": [
                {
                    "$or": [
                        {"$and": [{"field1": "value1"}, {"field2": "value2"}]},
                        {"$and": [{"field3": "value3"}, {"field4": "value4"}]},
                    ]
                },
                {"field5": "value5"},
            ]
        }

    def test_mixed_operators(self):
        """Test mixing different operators in nested structure."""
        result = (
            (
                FilterBuilder().eq("genre", "drama")
                & FilterBuilder().gt("year", 2020)
                & FilterBuilder().in_("tags", ["award-winning", "critically-acclaimed"])
            )
            | (
                FilterBuilder().eq("genre", "comedy")
                & FilterBuilder().lt("year", 2000)
                & FilterBuilder().exists("rating", True)
            )
        ).build()
        assert result == {
            "$or": [
                {
                    "$and": [
                        {"genre": "drama"},
                        {"year": {"$gt": 2020}},
                        {"tags": {"$in": ["award-winning", "critically-acclaimed"]}},
                    ]
                },
                {
                    "$and": [
                        {"genre": "comedy"},
                        {"year": {"$lt": 2000}},
                        {"rating": {"$exists": True}},
                    ]
                },
            ]
        }


class TestFilterBuilderEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_build_raises_error(self):
        """Test that building an empty filter raises ValueError."""
        builder = FilterBuilder()
        with pytest.raises(ValueError, match="FilterBuilder must have at least one condition"):
            builder.build()

    def test_single_condition(self):
        """Test that a single condition works correctly."""
        result = FilterBuilder().eq("genre", "drama").build()
        assert result == {"genre": "drama"}

    def test_empty_list_in(self):
        """Test in-list with empty list."""
        result = FilterBuilder().in_("genre", []).build()
        assert result == {"genre": {"$in": []}}

    def test_empty_list_nin(self):
        """Test not-in-list with empty list."""
        result = FilterBuilder().nin("genre", []).build()
        assert result == {"genre": {"$nin": []}}

    def test_single_item_list_in(self):
        """Test in-list with single item."""
        result = FilterBuilder().in_("genre", ["drama"]).build()
        assert result == {"genre": {"$in": ["drama"]}}

    def test_large_list_in(self):
        """Test in-list with many items."""
        items = [f"item{i}" for i in range(100)]
        result = FilterBuilder().in_("field", items).build()
        assert result == {"field": {"$in": items}}

    def test_all_value_types(self):
        """Test all supported value types."""
        result = FilterBuilder().eq("str_field", "string").build()
        assert result == {"str_field": "string"}

        result = FilterBuilder().eq("int_field", 42).build()
        assert result == {"int_field": 42}

        result = FilterBuilder().eq("float_field", 3.14).build()
        assert result == {"float_field": 3.14}

        result = FilterBuilder().eq("bool_field", True).build()
        assert result == {"bool_field": True}

    def test_numeric_operators_with_float(self):
        """Test numeric operators accept float values."""
        result = FilterBuilder().gt("rating", 4.5).build()
        assert result == {"rating": {"$gt": 4.5}}

        result = FilterBuilder().gte("rating", 4.5).build()
        assert result == {"rating": {"$gte": 4.5}}

        result = FilterBuilder().lt("rating", 3.5).build()
        assert result == {"rating": {"$lt": 3.5}}

        result = FilterBuilder().lte("rating", 3.5).build()
        assert result == {"rating": {"$lte": 3.5}}

    def test_numeric_operators_with_int(self):
        """Test numeric operators accept int values."""
        result = FilterBuilder().gt("year", 2020).build()
        assert result == {"year": {"$gt": 2020}}

        result = FilterBuilder().gte("year", 2020).build()
        assert result == {"year": {"$gte": 2020}}

        result = FilterBuilder().lt("year", 2000).build()
        assert result == {"year": {"$lt": 2000}}

        result = FilterBuilder().lte("year", 2000).build()
        assert result == {"year": {"$lte": 2000}}


class TestFilterBuilderRealWorldExamples:
    """Test real-world filter examples."""

    def test_movie_search_example(self):
        """Example: Find movies that are dramas from 2020 or later, or comedies from before 2000."""
        result = (
            (FilterBuilder().eq("genre", "drama") & FilterBuilder().gte("year", 2020))
            | (FilterBuilder().eq("genre", "comedy") & FilterBuilder().lt("year", 2000))
        ).build()
        assert result == {
            "$or": [
                {"$and": [{"genre": "drama"}, {"year": {"$gte": 2020}}]},
                {"$and": [{"genre": "comedy"}, {"year": {"$lt": 2000}}]},
            ]
        }

    def test_product_search_example(self):
        """Example: Find products in certain categories with price range."""
        result = (
            FilterBuilder().in_("category", ["electronics", "computers"])
            & FilterBuilder().gte("price", 100.0)
            & FilterBuilder().lte("price", 1000.0)
        ).build()
        assert result == {
            "$and": [
                {"category": {"$in": ["electronics", "computers"]}},
                {"price": {"$gte": 100.0}},
                {"price": {"$lte": 1000.0}},
            ]
        }

    def test_exclude_certain_values_example(self):
        """Example: Exclude certain values and require existence of a field."""
        result = (
            FilterBuilder().nin("status", ["deleted", "archived"])
            & FilterBuilder().exists("published_at", True)
        ).build()
        assert result == {
            "$and": [
                {"status": {"$nin": ["deleted", "archived"]}},
                {"published_at": {"$exists": True}},
            ]
        }
