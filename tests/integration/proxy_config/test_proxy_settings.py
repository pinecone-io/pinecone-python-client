import os
import pytest
from pinecone import Pinecone
from urllib3 import make_headers
from urllib3.exceptions import InsecureRequestWarning

PROXY_URL_HTTPS = 'https://127.0.0.1:8080'
PROXY_URL_HTTP = 'http://127.0.0.1:8080'

PROXY_AUTH_CREDS = 'testuser:testpassword'
PROXY_W_PROXY_AUTH = 'https://127.0.0.1:8081'

def exercise_all_apis(client, index_name):
    # Control plane
    client.list_indexes()

    # Data plane
    index = client.Index(index_name)
    index.describe_index_stats()

class TestProxyConfig:
    def test_https_proxy_with_self_signed_cert(self, api_key, index_name):
        pc = Pinecone(
            api_key=api_key, 
            proxy_url=PROXY_URL_HTTPS,
            ssl_ca_certs='./.mitm/proxy1/mitmproxy-ca-cert.pem',
        )
        exercise_all_apis(pc, index_name)

    def test_http_proxy_with_self_signed_cert(self, api_key, index_name):
        pc = Pinecone(
            api_key=api_key, 
            proxy_url=PROXY_URL_HTTP,
            ssl_ca_certs='./.mitm/proxy1/mitmproxy-ca-cert.pem',
        )
        exercise_all_apis(pc, index_name)

    def test_proxy_with_ssl_verification_disabled_emits_warning(self, api_key):
        pc = Pinecone(
            api_key=api_key, 
            proxy_url=PROXY_URL_HTTPS,
            ssl_verify=False,
        )

        with pytest.warns(InsecureRequestWarning):
            pc.list_indexes()

    def test_proxy_with_incorrect_cert_path(self, api_key):
        with pytest.raises(Exception) as e:
            pc = Pinecone(
                api_key=api_key,
                proxy_url=PROXY_URL_HTTPS,
                ssl_ca_certs='~/incorrect/path',
            )
            pc.list_indexes()

        assert 'No such file or directory' in str(e.value)

    def test_proxy_with_valid_path_to_incorrect_cert(self, api_key):
        with pytest.raises(Exception) as e:
            pc = Pinecone(
                api_key=api_key,
                proxy_url=PROXY_URL_HTTPS,
                ssl_ca_certs='./.mitm/proxy2/mitmproxy-ca-cert.pem',
            )
            pc.list_indexes()

        assert 'CERTIFICATE_VERIFY_FAILED' in str(e.value)

    def test_proxy_that_requires_proxyauth(self, api_key, index_name):
        pc = Pinecone(
            api_key=api_key,
            proxy_url=PROXY_W_PROXY_AUTH,
            proxy_headers=make_headers(proxy_basic_auth=PROXY_AUTH_CREDS),
            ssl_ca_certs='./.mitm/proxy2/mitmproxy-ca-cert.pem'
        )
        exercise_all_apis(pc, index_name)

