from pinecone.core.openapi.db_control.models import IndexModelStatus
from pinecone.core.openapi.db_data.models import VectorValues
from pinecone.openapi_support.serializer import Serializer
from pinecone.openapi_support.api_client_utils import parameters_to_tuples
from datetime import date, datetime


class TestSanitization:
    def test_sanitize_for_serialization_returns_basic_types(self):
        # Return basic types without modification
        assert Serializer.sanitize_for_serialization(None) is None
        assert Serializer.sanitize_for_serialization(1) == 1
        assert Serializer.sanitize_for_serialization(-1) == -1
        assert Serializer.sanitize_for_serialization("string") == "string"
        assert Serializer.sanitize_for_serialization(True) is True
        assert Serializer.sanitize_for_serialization(False) is False
        assert Serializer.sanitize_for_serialization(1.1) == 1.1

    def test_sanitize_for_serialization_stringifies_dates(self):
        assert Serializer.sanitize_for_serialization(date(2021, 1, 1)) == "2021-01-01"
        assert (
            Serializer.sanitize_for_serialization(datetime(2021, 1, 1, 0, 0, 0))
            == "2021-01-01T00:00:00"
        )

    def sanitize_for_serialization_serializes_dicts(self):
        assert Serializer.sanitize_for_serialization({"key": "value"}) == {"key": "value"}
        assert Serializer.sanitize_for_serialization({"key": {"inner_key": "foo"}}) == {
            "key": {"inner_key": "foo"}
        }

    def test_sanitize_for_serialization_serializes_lists(self):
        assert Serializer.sanitize_for_serialization([1, 2, 3]) == [1, 2, 3]
        assert Serializer.sanitize_for_serialization([date(2021, 1, 1), date(2021, 1, 2)]) == [
            "2021-01-01",
            "2021-01-02",
        ]

    def test_sanitize_for_serialization_serializes_tuples(self):
        assert Serializer.sanitize_for_serialization((1, 2, 3)) == [1, 2, 3]
        assert Serializer.sanitize_for_serialization((date(2021, 1, 1), date(2021, 1, 2))) == [
            "2021-01-01",
            "2021-01-02",
        ]
        assert Serializer.sanitize_for_serialization((None, None, "1", 23.3)) == [
            None,
            None,
            "1",
            23.3,
        ]

    def test_sanitization_for_serialization_serializes_io(self):
        import io

        assert Serializer.sanitize_for_serialization(io.BytesIO(b"test")) == b"test"

    def test_sanitize_for_serialization_serializes_model_normal(self):
        from tests.fixtures import make_index_model

        m = make_index_model(
            name="myindex",
            dimension=10,
            metric="cosine",
            host="localhost",
            status=IndexModelStatus(ready=True, state="Ready"),
        ).index  # Get the underlying OpenAPI model

        serialized = Serializer.sanitize_for_serialization(m)
        # New API format: uses schema and deployment
        assert serialized["name"] == "myindex"
        assert serialized["schema"]["fields"]["_values"]["dimension"] == 10
        assert serialized["schema"]["fields"]["_values"]["metric"] == "cosine"
        assert serialized["host"] == "localhost"
        assert serialized["deployment"]["deployment_type"] == "serverless"
        assert serialized["status"] == {"ready": True, "state": "Ready"}

        m2 = make_index_model(
            name="myindex2",
            metric="cosine",
            host="localhost",
            status=IndexModelStatus(ready=True, state="Ready"),
            deletion_protection="enabled",
        ).index

        serialized2 = Serializer.sanitize_for_serialization(m2)
        assert serialized2["name"] == "myindex2"
        assert serialized2["schema"]["fields"] == {}  # No dimension means empty schema
        assert serialized2["host"] == "localhost"
        assert serialized2["deployment"]["deployment_type"] == "serverless"
        assert serialized2["status"] == {"ready": True, "state": "Ready"}
        assert serialized2["deletion_protection"] == "enabled"

    def test_sanitize_for_serialization_serializes_model_simple(self):
        # ModelSimple is used to model named values which are not objects
        m = VectorValues(value=[1.0, 2.0, 3.0])
        assert Serializer.sanitize_for_serialization(m) == [1.0, 2.0, 3.0]


class TestParametersToTuples:
    def test_parameters_to_tuples_converts_dict_to_list_of_tuples(self):
        params = {"key1": "value1", "key2": "value2"}
        assert parameters_to_tuples(params, None) == [("key1", "value1"), ("key2", "value2")]
        assert parameters_to_tuples({}, None) == []

    def test_parameters_to_tuples_tuple_input(self):
        params = [("key1", "value1"), ("key2", "value2")]
        assert parameters_to_tuples(params, None) == params
        assert parameters_to_tuples([], None) == []

    def test_parameters_to_tuples_accepts_dict_values(self):
        params = {"key1": {"inner_key": "value1"}, "key2": {"inner_key": "value2"}}
        assert parameters_to_tuples(params, None) == [
            ("key1", {"inner_key": "value1"}),
            ("key2", {"inner_key": "value2"}),
        ]

    def test_parameters_to_tuples_accepts_list_values(self):
        params = {"key1": ["value1", "value2"], "key2": ["value3", "value4"]}
        assert parameters_to_tuples(params, None) == [
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

        assert parameters_to_tuples(params, collection_formats) == [
            ("key1", "value1,value2,value3"),
            ("key2", "value1 value2 value3"),
            ("key3", "value1\tvalue2\tvalue3"),
            ("key4", "value1|value2|value3"),
        ]

    def test_parameters_to_tuples_with_collection_format_multi(self):
        collection_formats = {"key1": "multi", "key2": "csv"}
        params = {"key1": ["value1", "value2", "value3"], "key2": ["value4", "value5"]}

        assert parameters_to_tuples(params, collection_formats) == [
            ("key1", "value1"),
            ("key1", "value2"),
            ("key1", "value3"),
            ("key2", "value4,value5"),
        ]

    def test_tuple_with_collection_format_multi(self):
        collection_formats = {"key1": "multi", "key2": "csv"}
        params = [("key1", ["value1", "value2", "value3"]), ("key2", ["value4", "value5"])]

        assert parameters_to_tuples(params, collection_formats) == [
            ("key1", "value1"),
            ("key1", "value2"),
            ("key1", "value3"),
            ("key2", "value4,value5"),
        ]

    def test_casts_to_string(self):
        collection_formats = {"key1": "multi", "key2": "csv"}
        params = {"key1": [1, 2, 3], "key2": [1, 2]}

        # This seems kinda crazy, but I'm just characterizing the current behavior
        assert parameters_to_tuples(params, collection_formats) == [
            ("key1", 1),
            ("key1", 2),
            ("key1", 3),
            ("key2", "1,2"),
        ]
