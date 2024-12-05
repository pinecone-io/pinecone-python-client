from pinecone.core.openapi.db_control.models import (
    IndexModel,
    IndexModelStatus,
    IndexModelSpec,
    DeletionProtection,
)
from pinecone.openapi_support import ApiClient
from datetime import date, datetime


class TestSanitization:
    def test_sanitize_for_serialization_returns_basic_types(self):
        # Return basic types without modification
        assert ApiClient.sanitize_for_serialization(None) is None
        assert ApiClient.sanitize_for_serialization(1) == 1
        assert ApiClient.sanitize_for_serialization(-1) == -1
        assert ApiClient.sanitize_for_serialization("string") == "string"
        assert ApiClient.sanitize_for_serialization(True) is True
        assert ApiClient.sanitize_for_serialization(False) is False
        assert ApiClient.sanitize_for_serialization(1.1) == 1.1

    def test_sanitize_for_serialization_stringifies_dates(self):
        assert ApiClient.sanitize_for_serialization(date(2021, 1, 1)) == "2021-01-01"
        assert (
            ApiClient.sanitize_for_serialization(datetime(2021, 1, 1, 0, 0, 0))
            == "2021-01-01T00:00:00"
        )

    def sanitize_for_serialization_serializes_dicts(self):
        assert ApiClient.sanitize_for_serialization({"key": "value"}) == {"key": "value"}
        assert ApiClient.sanitize_for_serialization({"key": {"inner_key": "foo"}}) == {
            "key": {"inner_key": "foo"}
        }

    def test_sanitize_for_serialization_serializes_lists(self):
        assert ApiClient.sanitize_for_serialization([1, 2, 3]) == [1, 2, 3]
        assert ApiClient.sanitize_for_serialization([date(2021, 1, 1), date(2021, 1, 2)]) == [
            "2021-01-01",
            "2021-01-02",
        ]

    def test_sanitize_for_serialization_serializes_tuples(self):
        assert ApiClient.sanitize_for_serialization((1, 2, 3)) == [1, 2, 3]
        assert ApiClient.sanitize_for_serialization((date(2021, 1, 1), date(2021, 1, 2))) == [
            "2021-01-01",
            "2021-01-02",
        ]
        assert ApiClient.sanitize_for_serialization((None, None, "1", 23.3)) == [
            None,
            None,
            "1",
            23.3,
        ]

    def test_sanitization_for_serialization_serializes_io(self):
        import io

        assert ApiClient.sanitize_for_serialization(io.BytesIO(b"test")) == b"test"

    def test_sanitize_for_serialization_serializes_model_normal(self):
        m = IndexModel(
            name="myindex",
            dimension=10,
            metric="cosine",
            host="localhost",
            spec=IndexModelSpec(),
            status=IndexModelStatus(ready=True, state="Ready"),
            vector_type="dense",
        )
        assert ApiClient.sanitize_for_serialization(m) == {
            "name": "myindex",
            "dimension": 10,
            "metric": "cosine",
            "host": "localhost",
            "spec": {},
            "status": {"ready": True, "state": "Ready"},
            "vector_type": "dense",
        }

        m2 = IndexModel(
            name="myindex2",
            metric="cosine",
            host="localhost",
            spec=IndexModelSpec(),
            status=IndexModelStatus(ready=True, state="Ready"),
            vector_type="sparse",
            deletion_protection=DeletionProtection(value="enabled"),
        )
        assert ApiClient.sanitize_for_serialization(m2) == {
            "name": "myindex2",
            "metric": "cosine",
            "host": "localhost",
            "spec": {},
            "status": {"ready": True, "state": "Ready"},
            "vector_type": "sparse",
            "deletion_protection": "enabled",
        }

    def test_sanitize_for_serialization_serializes_model_simple(self):
        # ModelSimple is used to model named values which are not objects
        m = DeletionProtection(value="enabled")
        assert ApiClient.sanitize_for_serialization(m) == "enabled"


class TestParametersToTuples:
    def test_parameters_to_tuples_converts_dict_to_list_of_tuples(self):
        params = {"key1": "value1", "key2": "value2"}
        assert ApiClient.parameters_to_tuples(params, None) == [
            ("key1", "value1"),
            ("key2", "value2"),
        ]
        assert ApiClient.parameters_to_tuples({}, None) == []

    def test_parameters_to_tuples_tuple_input(self):
        params = [("key1", "value1"), ("key2", "value2")]
        assert ApiClient.parameters_to_tuples(params, None) == params
        assert ApiClient.parameters_to_tuples([], None) == []

    def test_parameters_to_tuples_accepts_dict_values(self):
        params = {"key1": {"inner_key": "value1"}, "key2": {"inner_key": "value2"}}
        assert ApiClient.parameters_to_tuples(params, None) == [
            ("key1", {"inner_key": "value1"}),
            ("key2", {"inner_key": "value2"}),
        ]

    def test_parameters_to_tuples_accepts_list_values(self):
        params = {"key1": ["value1", "value2"], "key2": ["value3", "value4"]}
        assert ApiClient.parameters_to_tuples(params, None) == [
            ("key1", ["value1", "value2"]),
            ("key2", ["value3", "value4"]),
        ]

    def test_parameters_to_tuples_with_collection_format(self):
        collection_formats = {"key1": "csv", "key2": "ssv", "key3": "tsv", "key4": "pipes"}
        params = {
            "key1": ["value1", "value2", "value3"],
            "key2": ["value1", "value2", "value3"],
            "key3": ["value1", "value2", "value3"],
            "key4": ["value1", "value2", "value3"],
        }

        assert ApiClient.parameters_to_tuples(params, collection_formats) == [
            ("key1", "value1,value2,value3"),
            ("key2", "value1 value2 value3"),
            ("key3", "value1\tvalue2\tvalue3"),
            ("key4", "value1|value2|value3"),
        ]

    def test_parameters_to_tuples_with_collection_format_multi(self):
        collection_formats = {"key1": "multi", "key2": "csv"}
        params = {"key1": ["value1", "value2", "value3"], "key2": ["value4", "value5"]}

        assert ApiClient.parameters_to_tuples(params, collection_formats) == [
            ("key1", "value1"),
            ("key1", "value2"),
            ("key1", "value3"),
            ("key2", "value4,value5"),
        ]

    def test_tuple_with_collection_format_multi(self):
        collection_formats = {"key1": "multi", "key2": "csv"}
        params = [("key1", ["value1", "value2", "value3"]), ("key2", ["value4", "value5"])]

        assert ApiClient.parameters_to_tuples(params, collection_formats) == [
            ("key1", "value1"),
            ("key1", "value2"),
            ("key1", "value3"),
            ("key2", "value4,value5"),
        ]

    def test_casts_to_string(self):
        collection_formats = {"key1": "multi", "key2": "csv"}
        params = {"key1": [1, 2, 3], "key2": [1, 2]}

        # This seems kinda crazy, but I'm just characterizing the current behavior
        assert ApiClient.parameters_to_tuples(params, collection_formats) == [
            ("key1", 1),
            ("key1", 2),
            ("key1", 3),
            ("key2", "1,2"),
        ]
