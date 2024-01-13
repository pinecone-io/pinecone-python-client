import re
from pinecone.utils.user_agent import get_user_agent

class TestUserAgent():
    def test_user_agent(self):
        useragent = get_user_agent()
        assert re.search(r"python-client-\d+\.\d+\.\d+", useragent) is not None
        assert re.search(r"urllib3:\d+\.\d+\.\d+", useragent) is not None