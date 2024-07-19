import pytest
from ..helpers import get_environment_var

@pytest.fixture()
def api_key():
    return get_environment_var("PINECONE_API_KEY")
