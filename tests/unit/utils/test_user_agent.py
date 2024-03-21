import re
from pinecone.utils.user_agent import get_user_agent
from pinecone.config import ConfigBuilder

class TestUserAgent():
    def test_user_agent(self):
        config = ConfigBuilder.build(api_key="my-api-key", host="https://my-controller-host")
        useragent = get_user_agent(config)
        assert re.search(r"python-client-\d+\.\d+\.\d+", useragent) is not None
        assert re.search(r"urllib3:\d+\.\d+\.\d+", useragent) is not None

    def test_user_agent_with_source_tag(self):
        config = ConfigBuilder.build(api_key="my-api-key", host="https://my-controller-host", source_tag="my_source_tag")
        useragent = get_user_agent(config)
        assert re.search(r"python-client-\d+\.\d+\.\d+", useragent) is not None
        assert re.search(r"urllib3:\d+\.\d+\.\d+", useragent) is not None
        assert re.search(r"source_tag=my_source_tag", useragent) is not None

    def test_source_tag_is_normalized(self):
        config = ConfigBuilder.build(api_key="my-api-key", host="https://my-controller-host", source_tag="my source tag!!!!")
        useragent = get_user_agent(config)
        assert re.search(r"source_tag=my_source_tag", useragent) is not None

        config = ConfigBuilder.build(api_key="my-api-key", host="https://my-controller-host", source_tag="My Source Tag")
        useragent = get_user_agent(config)
        assert re.search(r"source_tag=my_source_tag", useragent) is not None

        config = ConfigBuilder.build(api_key="my-api-key", host="https://my-controller-host", source_tag="   My Source Tag  123  ")
        useragent = get_user_agent(config)
        assert re.search(r"source_tag=my_source_tag_123", useragent) is not None
