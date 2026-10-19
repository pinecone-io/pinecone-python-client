"""An ``http://`` index host warns on the REST data plane, as it does on gRPC.

Plain ``http`` has to keep working for local simulators, so the host is kept as
given and the warning is the only signal that the API key crosses a public
network unencrypted.
"""

from __future__ import annotations

import warnings

import httpx
import pytest
import respx

from pinecone import AsyncPinecone, Pinecone
from pinecone._internal.config import PineconeConfig, RetryConfig
from pinecone._internal.data_plane_helpers import _warn_if_plaintext_public_host
from pinecone._internal.http_client import AsyncHTTPClient, HTTPClient
from pinecone.async_client.async_index import AsyncIndex
from pinecone.errors.exceptions import PineconeConnectionError, PineconeTimeoutError
from pinecone.index import Index

PUBLIC_HOST = "idx-abc123.svc.aped-4627-b74a.pinecone.io"


def _plaintext_warnings(record: list[warnings.WarningMessage]) -> list[warnings.WarningMessage]:
    return [w for w in record if issubclass(w.category, RuntimeWarning)]


class TestSyncIndexWarning:
    def test_http_public_host_warns_once_naming_host_and_scheme(self) -> None:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            index = Pinecone(api_key="x").index(host=f"http://{PUBLIC_HOST}")

        found = _plaintext_warnings(record)
        assert len(found) == 1
        message = str(found[0].message)
        assert PUBLIC_HOST in message
        assert "http" in message
        assert "unencrypted" in message
        assert index.host == f"http://{PUBLIC_HOST}"

    def test_direct_construction_warns(self) -> None:
        with pytest.warns(RuntimeWarning, match=PUBLIC_HOST):
            Index(host=f"http://{PUBLIC_HOST}", api_key="x")

    @pytest.mark.parametrize(
        "host",
        [
            PUBLIC_HOST,
            f"https://{PUBLIC_HOST}",
            "http://localhost:5080",
            "http://127.0.0.1:5080",
            "http://10.1.2.3",
            "http://172.20.0.5:8080",
            "http://192.168.1.10",
            "http://sim.localhost:5080",
        ],
    )
    def test_no_warning(self, host: str) -> None:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            Pinecone(api_key="x").index(host=host)
        assert _plaintext_warnings(record) == []

    def test_userinfo_is_left_out_of_the_message(self) -> None:
        with pytest.warns(RuntimeWarning) as record:
            Index(host=f"http://user:hunter2@{PUBLIC_HOST}", api_key="x")
        assert "hunter2" not in str(record[0].message)

    def test_public_ip_over_http_warns(self) -> None:
        with pytest.warns(RuntimeWarning, match="8.8.8.8"):
            Pinecone(api_key="x").index(host="http://8.8.8.8:5080")

    @pytest.mark.parametrize("port", ["99999", "80a"])
    def test_malformed_port_does_not_raise_from_the_helper(self, port: str) -> None:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            _warn_if_plaintext_public_host(f"http://{PUBLIC_HOST}:{port}")
        assert _plaintext_warnings(record) == []

    def test_out_of_range_port_does_not_fail_construction(self) -> None:
        host = f"http://{PUBLIC_HOST}:99999"
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            index = Index(host=host, api_key="x")
        assert index.host == host
        assert _plaintext_warnings(record) == []

    def test_grpc_index_gets_no_rest_warning(self) -> None:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            Pinecone(api_key="x").index(host=f"http://{PUBLIC_HOST}", grpc=True)
        assert not [w for w in record if "uses http" in str(w.message)]


class TestAsyncIndexWarning:
    @pytest.mark.asyncio
    async def test_http_public_host_warns_once_naming_host_and_scheme(self) -> None:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            index = await AsyncPinecone(api_key="x").index(host=f"http://{PUBLIC_HOST}")

        found = _plaintext_warnings(record)
        assert len(found) == 1
        message = str(found[0].message)
        assert PUBLIC_HOST in message
        assert "http" in message
        assert "unencrypted" in message
        assert index.host == f"http://{PUBLIC_HOST}"

    def test_direct_construction_warns(self) -> None:
        with pytest.warns(RuntimeWarning, match=PUBLIC_HOST):
            AsyncIndex(host=f"http://{PUBLIC_HOST}", api_key="x")

    def test_out_of_range_port_does_not_fail_construction(self) -> None:
        host = f"http://{PUBLIC_HOST}:99999"
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            index = AsyncIndex(host=host, api_key="x")
        assert index.host == host
        assert _plaintext_warnings(record) == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "host",
        [
            PUBLIC_HOST,
            f"https://{PUBLIC_HOST}",
            "http://localhost:5080",
            "http://127.0.0.1:5080",
            "http://10.1.2.3",
            "http://172.20.0.5:8080",
            "http://192.168.1.10",
            "http://sim.localhost:5080",
        ],
    )
    async def test_no_warning(self, host: str) -> None:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            await AsyncPinecone(api_key="x").index(host=host)
        assert _plaintext_warnings(record) == []

    def test_userinfo_is_left_out_of_the_message(self) -> None:
        with pytest.warns(RuntimeWarning) as record:
            AsyncIndex(host=f"http://user:hunter2@{PUBLIC_HOST}", api_key="x")
        assert "hunter2" not in str(record[0].message)

    @pytest.mark.asyncio
    async def test_public_ip_over_http_warns(self) -> None:
        with pytest.warns(RuntimeWarning, match="8.8.8.8"):
            await AsyncPinecone(api_key="x").index(host="http://8.8.8.8:5080")


class TestTransportErrorsNameTheRequestOrigin:
    URL = f"http://{PUBLIC_HOST}"

    def _sync(self) -> HTTPClient:
        return HTTPClient(
            PineconeConfig(api_key="k", host=self.URL, retry_config=RetryConfig(max_retries=0)),
            "2026-07",
        )

    def _async(self) -> AsyncHTTPClient:
        return AsyncHTTPClient(
            PineconeConfig(api_key="k", host=self.URL, retry_config=RetryConfig(max_retries=0)),
            "2026-07",
        )

    def test_sync_timeout_names_scheme_and_host(self) -> None:
        with respx.mock:
            respx.post(f"{self.URL}/documents/fetch").mock(
                side_effect=httpx.ConnectTimeout("timed out")
            )
            with pytest.raises(PineconeTimeoutError) as exc_info:
                self._sync().post("/documents/fetch", json={})
        assert f"http://{PUBLIC_HOST}" in str(exc_info.value)
        assert "timed out" in str(exc_info.value)

    def test_sync_connection_error_names_scheme_and_host(self) -> None:
        with respx.mock:
            respx.get(f"{self.URL}/x").mock(side_effect=httpx.ConnectError("refused"))
            with pytest.raises(PineconeConnectionError) as exc_info:
                self._sync().get("/x")
        assert f"http://{PUBLIC_HOST}" in str(exc_info.value)

    def test_message_leaves_out_path_and_query(self) -> None:
        with respx.mock:
            respx.get(f"{self.URL}/secret/path").mock(side_effect=httpx.ConnectError("refused"))
            with pytest.raises(PineconeConnectionError) as exc_info:
                self._sync().get("/secret/path", params={"token": "abc"})
        assert "secret" not in str(exc_info.value)
        assert "token" not in str(exc_info.value)

    def test_message_leaves_out_userinfo(self) -> None:
        client = HTTPClient(
            PineconeConfig(
                api_key="k",
                host=f"http://user:hunter2@{PUBLIC_HOST}",
                retry_config=RetryConfig(max_retries=0),
            ),
            "2026-07",
        )
        with respx.mock:
            respx.get(f"{self.URL}/x").mock(side_effect=httpx.ConnectError("refused"))
            with pytest.raises(PineconeConnectionError) as exc_info:
                client.get("/x")
        assert f"http://{PUBLIC_HOST}" in str(exc_info.value)
        assert "hunter2" not in str(exc_info.value)

    def test_port_is_kept(self) -> None:
        client = HTTPClient(
            PineconeConfig(
                api_key="k",
                host="http://example.test:8080",
                retry_config=RetryConfig(max_retries=0),
            ),
            "2026-07",
        )
        with respx.mock:
            respx.get("http://example.test:8080/x").mock(side_effect=httpx.ConnectError("refused"))
            with pytest.raises(PineconeConnectionError, match=r"http://example\.test:8080"):
                client.get("/x")

    @pytest.mark.asyncio
    async def test_async_timeout_names_scheme_and_host(self) -> None:
        with respx.mock:
            respx.post(f"{self.URL}/documents/fetch").mock(side_effect=httpx.ReadTimeout(""))
            with pytest.raises(PineconeTimeoutError) as exc_info:
                await self._async().post("/documents/fetch", json={})
        assert f"http://{PUBLIC_HOST}" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_async_connection_error_names_scheme_and_host(self) -> None:
        with respx.mock:
            respx.get(f"{self.URL}/x").mock(side_effect=httpx.ConnectError("refused"))
            with pytest.raises(PineconeConnectionError) as exc_info:
                await self._async().get("/x")
        assert f"http://{PUBLIC_HOST}" in str(exc_info.value)
