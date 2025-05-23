import pytest

from pinecone.openapi_support import ApiClient, PineconeApiException
from pinecone.core.openapi.db_data.models import (
    StartImportResponse,
    ImportErrorMode as ImportErrorModeGeneratedClass,
)

from pinecone.db_data.resources.sync.bulk_import import BulkImportResource, ImportErrorMode


def build_client_w_faked_response(mocker, body: str, status: int = 200):
    response = mocker.Mock()
    response.headers = {"content-type": "application/json"}
    response.status = status
    response.data = body.encode("utf-8")

    api_client = ApiClient()
    mock_request = mocker.patch.object(
        api_client.rest_client.pool_manager, "request", return_value=response
    )
    return BulkImportResource(api_client=api_client), mock_request


class TestBulkImportStartImport:
    def test_start_minimal(self, mocker):
        body = """
        {
            "id": "1"
        }
        """
        client, mock_req = build_client_w_faked_response(mocker, body)

        my_import = client.start("s3://path/to/file.parquet")

        # We made some overrides to the print behavior, so we need to
        # call it to ensure it doesn't raise an exception
        print(my_import)

        assert my_import.id == "1"
        assert my_import["id"] == "1"
        assert my_import.to_dict() == {"id": "1"}
        assert my_import.__class__ == StartImportResponse

    def test_start_with_kwargs(self, mocker):
        body = """
        {
            "id": "1"
        }
        """
        client, mock_req = build_client_w_faked_response(mocker, body)

        my_import = client.start(uri="s3://path/to/file.parquet", integration_id="123-456-789")
        assert my_import.id == "1"
        assert my_import["id"] == "1"
        assert my_import.to_dict() == {"id": "1"}
        assert my_import.__class__ == StartImportResponse

        # By default, use continue error mode
        _, call_kwargs = mock_req.call_args
        assert (
            call_kwargs["body"]
            == '{"uri": "s3://path/to/file.parquet", "integrationId": "123-456-789", "errorMode": {"onError": "continue"}}'
        )

    @pytest.mark.parametrize(
        "error_mode_input", [ImportErrorMode.CONTINUE, "Continue", "continue", "cONTINUE"]
    )
    def test_start_with_explicit_error_mode(self, mocker, error_mode_input):
        body = """
        {
            "id": "1"
        }
        """
        client, mock_req = build_client_w_faked_response(mocker, body)

        client.start(uri="s3://path/to/file.parquet", error_mode=error_mode_input)
        _, call_kwargs = mock_req.call_args
        assert (
            call_kwargs["body"]
            == '{"uri": "s3://path/to/file.parquet", "errorMode": {"onError": "continue"}}'
        )

    def test_start_with_abort_error_mode(self, mocker):
        body = """
        {
            "id": "1"
        }
        """
        client, mock_req = build_client_w_faked_response(mocker, body)

        client.start(uri="s3://path/to/file.parquet", error_mode=ImportErrorMode.ABORT)
        _, call_kwargs = mock_req.call_args
        assert (
            call_kwargs["body"]
            == '{"uri": "s3://path/to/file.parquet", "errorMode": {"onError": "abort"}}'
        )

    def test_start_with_unknown_error_mode(self, mocker):
        body = """
        {
            "id": "1"
        }
        """
        client, mock_req = build_client_w_faked_response(mocker, body)

        with pytest.raises(ValueError) as e:
            client.start(uri="s3://path/to/file.parquet", error_mode="unknown")

        assert "Invalid error_mode: unknown" in str(e.value)

    def test_start_invalid_uri(self, mocker):
        body = """
        {
            "code": "3",
            "message": "Bulk import URIs must start with the scheme of a supported storage provider",
            "details": []
        }
        """
        client, mock_req = build_client_w_faked_response(mocker, body, 400)

        with pytest.raises(PineconeApiException) as e:
            client.start(uri="invalid path")

        assert e.value.status == 400
        assert e.value.body == body
        assert "Bulk import URIs must start with the scheme of a supported storage provider" in str(
            e.value
        )

    def test_no_arguments(self, mocker):
        client, mock_req = build_client_w_faked_response(mocker, "")

        with pytest.raises(TypeError) as e:
            client.start()

        assert "missing 1 required positional argument" in str(e.value)

    def test_enums_are_aligned(self):
        modes = dir(ImportErrorMode)
        for key, _ in ImportErrorModeGeneratedClass().allowed_values[("on_error",)].items():
            assert key in modes


class TestDescribeImport:
    def test_describe_import(self, mocker):
        body = """
        {
            "id": "1",
            "records_imported": 1000,
            "uri": "s3://path/to/file.parquet",
            "status": "InProgress",
            "error_mode": "CONTINUE",
            "created_at": "2021-01-01T00:00:00Z",
            "updated_at": "2021-01-01T00:00:00Z",
            "integration": "s3",
            "error_message": "",
            "percent_complete": 43.2
        }
        """
        client, mock_req = build_client_w_faked_response(mocker, body)

        my_import = client.describe(id="1")

        # We made some overrides to the print behavior, so we need to
        # call it to ensure it doesn't raise an exception
        print(my_import)

        assert my_import.id == "1"
        assert my_import["id"] == "1"
        desc = my_import.to_dict()
        assert desc["id"] == "1"
        assert desc["records_imported"] == 1000
        assert desc["uri"] == "s3://path/to/file.parquet"
        assert desc["status"] == "InProgress"
