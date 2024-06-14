import pytest
import requests
from pinecone.utils import docslinks

urls = list(docslinks.values())


@pytest.mark.parametrize("url", urls)
def test_valid_links(url):
    response = requests.get(url)
    assert response.status_code == 200, f"Docs link is invalid: {url}"
