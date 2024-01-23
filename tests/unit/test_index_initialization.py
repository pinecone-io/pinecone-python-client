import pytest
from pinecone import Pinecone

class TestIndexInitialization():
    def test_additional_headers(self):
        pc = Pinecone(api_key='YOUR_API_KEY')
        index = pc.Index('my-index', additional_headers={'test-header': 'test-header-value'})


