import pytest
import warnings

from urllib3 import BaseHTTPResponse

from pinecone.core_ea.openapi.db_data.api.bulk_operations_api import BulkOperationsApi
from pinecone.core_ea.openapi.db_data.models import ImportModel, StartImportResponse
from pinecone.core_ea.openapi.shared.api_client import ApiClient
from pinecone.core_ea.openapi.shared.exceptions import PineconeApiException

from pinecone.data.features.bulk_import import ImportFeatureMixin


def build_api_w_faked_response(mocker, body: str, status: int = 200) -> BaseHTTPResponse:
    response = mocker.Mock()
    response.headers = {"content-type": "application/json"}
    response.status = status
    response.data = body.encode("utf-8")

    api_client = ApiClient()
    mocker.patch.object(api_client.rest_client.pool_manager, "request", return_value=response)
    return BulkOperationsApi(api_client=api_client)


def build_client_w_faked_response(mocker, body: str, status: int = 200):
    api_client = build_api_w_faked_response(mocker, body, status)
    return ImportFeatureMixin(__import_operations_api=api_client, api_key="asdf", host="asdf")


class TestBulkImportStartImport:
    def test_start_import(self, mocker):
        body = """
        {
            "id": "1"
        }
        """
        client = build_client_w_faked_response(mocker, body)

        with pytest.warns(UserWarning, match="prerelease"):
            my_import = client.start_import("s3://path/to/file.parquet")
            assert my_import.id == "1"
            assert my_import["id"] == "1"
            assert my_import.to_dict() == {"id": "1"}
            assert my_import.__class__ == StartImportResponse

    def test_start_import_with_kwargs(self, mocker):
        body = """
        {
            "id": "1"
        }
        """
        client = build_client_w_faked_response(mocker, body)

        with pytest.warns(UserWarning, match="prerelease"):
            my_import = client.start_import(uri="s3://path/to/file.parquet")
            assert my_import.id == "1"
            assert my_import["id"] == "1"
            assert my_import.to_dict() == {"id": "1"}
            assert my_import.__class__ == StartImportResponse

    def test_start_invalid_uri(self, mocker):
        body = """
        {
            "code": "3",
            "message": "Bulk import URIs must start with the scheme of a supported storage provider",
            "details": []
        }
        """
        client = build_client_w_faked_response(mocker, body, 400)

        with pytest.warns(UserWarning, match="prerelease"):
            with pytest.raises(PineconeApiException) as e:
                my_import = client.start_import(uri="invalid path")

        assert e.value.status == 400
        assert e.value.body == body
        assert "Bulk import URIs must start with the scheme of a supported storage provider" in str(e.value)

    def test_no_arguments(self, mocker):
        client = build_client_w_faked_response(mocker, "")

        with pytest.warns(UserWarning, match="prerelease"):
            with pytest.raises(TypeError) as e:
                client.start_import()

        assert "missing 1 required positional argument" in str(e.value)

class TestDescribeImport:
    def test_describe_import(self, mocker):
        # body = """
        # {
        #     "id": "1",
        #     "records_imported": 1000,
        #     "uri": "s3://path/to/file.parquet",
        #     "status": "In Progress",
        #     "error_mode": "CONTINUE",
        #     "created_at": "2021-01-01T00:00:00Z",
        #     "updated_at": "2021-01-01T00:00:00Z",
        #     "integration": "s3",
        #     "error_message": ""
        #     "percent_complete": 43.2
        # }
        # """
        body = """
        {
            "id": "1",
        }
        """
        client = build_client_w_faked_response(mocker, body)

        with pytest.warns(UserWarning, match="prerelease"):
            my_import = client.describe_import(id="1")
            assert my_import.id == "1"
            assert my_import["id"] == "1"
            assert my_import.to_dict() == {
                "id": "1",
                "records_imported": 1000,
                "uri": "s3://path/to/file.parquet",
                "status": "In Progress",
                "error_mode": "CONTINUE",
                "created_at": "2021-01-01T00:00:00Z",
                "updated_at": "2021-01-01T00:00:00Z",
                "integration": "s3",
                "error_message": "",
                "percent_complete": 43.2,
            }