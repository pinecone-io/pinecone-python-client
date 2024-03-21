import pytest
import re
from pinecone import ConfigBuilder, Pinecone

class TestIndexClientInitialization():
    @pytest.mark.parametrize(
        'additional_headers', 
        [
            None, 
            {}
        ]
    )
    def test_no_additional_headers_leaves_useragent_only(self, additional_headers):
        pc = Pinecone(api_key='YOUR_API_KEY')
        index = pc.Index(host='myhost', additional_headers=additional_headers)
        assert len(index._vector_api.api_client.default_headers) == 1
        assert 'User-Agent' in index._vector_api.api_client.default_headers
        assert 'python-client-' in index._vector_api.api_client.default_headers['User-Agent']

    def test_additional_headers_one_additional(self):
        pc = Pinecone(api_key='YOUR_API_KEY')
        index = pc.Index(
            host='myhost', 
            additional_headers={'test-header': 'test-header-value'}
        )
        assert 'test-header' in index._vector_api.api_client.default_headers
        assert len(index._vector_api.api_client.default_headers) == 2

    def test_multiple_additional_headers(self):
        pc = Pinecone(api_key='YOUR_API_KEY')
        index = pc.Index(
            host='myhost', 
            additional_headers={
                'test-header': 'test-header-value', 
                'test-header2': 'test-header-value2'
            }
        )
        assert 'test-header' in index._vector_api.api_client.default_headers
        assert 'test-header2' in index._vector_api.api_client.default_headers
        assert len(index._vector_api.api_client.default_headers) == 3

    def test_overwrite_useragent(self):
        # This doesn't seem like a common use case, but we may want to allow this
        # when embedding the client in other pinecone tools such as canopy.
        pc = Pinecone(api_key='YOUR_API_KEY')
        index = pc.Index(
            host='myhost', 
            additional_headers={
                'User-Agent': 'test-user-agent'
            }
        )
        assert len(index._vector_api.api_client.default_headers) == 1
        assert 'User-Agent' in index._vector_api.api_client.default_headers
        assert index._vector_api.api_client.default_headers['User-Agent'] == 'test-user-agent'

    def test_set_source_tag(self):
        pc = Pinecone(api_key="123-456-789", source_tag="test_source_tag")
        index = pc.Index(host='myhost')
        assert re.search(r"source_tag=test_source_tag", pc.index_api.api_client.user_agent) is not None

    def test_set_source_tag_via_config(self):
        config = ConfigBuilder.build(api_key='YOUR_API_KEY', host='https://my-host', source_tag='my_source_tag')
        pc = Pinecone(config=config)
        index = pc.Index(host='myhost')
        assert re.search(r"source_tag=my_source_tag", pc.index_api.api_client.user_agent) is not None