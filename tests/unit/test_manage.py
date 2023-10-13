import pytest
import pinecone

class TestManage:
    
    def test_get_api_instance_without_host(self):
        pinecone.init(api_key="123-456-789", environment="my-environment")
        api_instance = pinecone.manage._get_api_instance()
        assert api_instance.api_client.configuration.host == "https://controller.my-environment.pinecone.io"

    def test_get_api_instance_with_host(self):
        pinecone.init(api_key="123-456-789", environment="my-environment", host="my-host")
        api_instance = pinecone.manage._get_api_instance()
        assert api_instance.api_client.configuration.host == "my-host"

