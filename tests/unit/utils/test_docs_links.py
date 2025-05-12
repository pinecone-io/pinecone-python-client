import pytest
import requests
from pinecone.utils import docslinks
from pinecone import __version__

urls = list(docslinks.values())


@pytest.mark.parametrize("url", urls)
def test_valid_links(url):
    if isinstance(url, str):
        response = requests.get(url)
        assert response.status_code == 200, f"Docs link is invalid: {url}"
    else:
        versioned_url = url(__version__)
        response = requests.get(versioned_url)
        assert response.status_code == 200, f"Docs link is invalid: {versioned_url}"
