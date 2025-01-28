from urllib3 import BaseHTTPResponse, HTTPResponse

from datetime import datetime, date

from pinecone.core.openapi.db_data.api.bulk_operations_api import BulkOperationsApi
from pinecone.openapi_support import ApiClient, RESTResponse


def fake_response(mocker, body: str, status: int = 200) -> BaseHTTPResponse:
    r = HTTPResponse(
        body=body.encode("utf-8"),
        headers={"content-type": "application/json"},
        status=status,
        reason="OK",
        preload_content=True,
    )
    api_client = ApiClient()
    return_value = RESTResponse(r.status, r.data, r.headers, r.reason)
    mocker.patch.object(api_client, "request", return_value=return_value)
    return api_client


class TestBulkImport:
    def test_parsing_datetime_fields(self, mocker):
        body = """
        {
            "id": "1",
            "uri": "s3://pinecone-bulk-import-dataset/cc-news/cc-news-part1.parquet",
            "status": "Pending",
            "percentComplete": 0,
            "recordsImported": 0,
            "createdAt": "2024-08-27T17:10:32.206413+00:00"
        }
        """
        api_client = fake_response(mocker, body, 200)
        api = BulkOperationsApi(api_client=api_client)

        r = api.describe_bulk_import(id="1")
        assert r.created_at.year == 2024
        assert r.created_at.month == 8
        assert r.created_at.date() == date(year=2024, month=8, day=27)
        assert r.created_at.time() == datetime.strptime("17:10:32.206413", "%H:%M:%S.%f").time()
