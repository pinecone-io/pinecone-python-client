import os
import pytest
from urllib3 import make_headers
from urllib3.exceptions import InsecureRequestWarning

PROXY1_URL_HTTPS = "https://localhost:8080"
PROXY1_URL_HTTP = "http://localhost:8080"

PROXY2_URL = "https://localhost:8081"


def exercise_all_apis(client, index_name):
    # Control plane
    client.list_indexes()
    # Data plane
    index = client.Index(index_name)
    index.describe_index_stats()


class TestProxyConfig:
    @pytest.mark.skipif(
        os.getenv("USE_GRPC") != "false", reason="gRPC doesn't support 'https://' proxy URLs"
    )
    def test_https_proxy_with_self_signed_cert(self, client_cls, api_key, index_name, proxy1):
        ssl_ca_certs = os.path.join(proxy1["ssl_ca_certs"], "mitmproxy-ca-cert.pem")
        pc = client_cls(api_key=api_key, proxy_url=PROXY1_URL_HTTPS, ssl_ca_certs=ssl_ca_certs)
        exercise_all_apis(pc, index_name)

    def test_http_proxy_with_self_signed_cert(self, client_cls, api_key, index_name, proxy1):
        ssl_ca_certs = os.path.join(proxy1["ssl_ca_certs"], "mitmproxy-ca-cert.pem")
        pc = client_cls(api_key=api_key, proxy_url=PROXY1_URL_HTTP, ssl_ca_certs=ssl_ca_certs)
        exercise_all_apis(pc, index_name)

    @pytest.mark.skipif(
        os.getenv("USE_GRPC") != "false", reason="gRPC doesn't support disabling ssl_verify"
    )
    def test_proxy_with_ssl_verification_disabled_emits_warning(
        self, client_cls, api_key, index_name
    ):
        pc = client_cls(api_key=api_key, proxy_url=PROXY1_URL_HTTPS, ssl_verify=False)

        with pytest.warns(InsecureRequestWarning):
            pc.list_indexes()

    def test_proxy_with_incorrect_cert_path(self, client_cls, api_key):
        with pytest.raises(Exception) as e:
            pc = client_cls(
                api_key=api_key, proxy_url=PROXY1_URL_HTTPS, ssl_ca_certs="~/incorrect/path"
            )
            pc.list_indexes()

        assert "No such file or directory" in str(e.value)

    def test_proxy_with_valid_path_to_incorrect_cert(self, client_cls, api_key, proxy2):
        ssl_ca_certs = os.path.join(proxy2["ssl_ca_certs"], "mitmproxy-ca-cert.pem")
        with pytest.raises(Exception) as e:
            pc = client_cls(api_key=api_key, proxy_url=PROXY1_URL_HTTPS, ssl_ca_certs=ssl_ca_certs)
            pc.list_indexes()

        assert "CERTIFICATE_VERIFY_FAILED" in str(e.value)

    @pytest.mark.skipif(os.getenv("USE_GRPC") != "false", reason="gRPC doesn't support proxy auth")
    def test_proxy_that_requires_proxyauth(self, client_cls, api_key, index_name, proxy2):
        ssl_ca_certs = os.path.join(proxy2["ssl_ca_certs"], "mitmproxy-ca-cert.pem")
        username = proxy2["auth"][0]
        password = proxy2["auth"][1]
        pc = client_cls(
            api_key=api_key,
            proxy_url=PROXY2_URL,
            proxy_headers=make_headers(proxy_basic_auth=f"{username}:{password}"),
            ssl_ca_certs=ssl_ca_certs,
        )
        exercise_all_apis(pc, index_name)
