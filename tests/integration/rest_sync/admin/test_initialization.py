import pytest
from pinecone import Admin


class TestAdminInitialization:
    def test_initialization_missing_client_id(self):
        with pytest.raises(ValueError):
            admin = Admin(client_id="", client_secret="asdf")
            assert admin is not None

    def test_initialization_missing_client_secret(self):
        with pytest.raises(ValueError):
            admin = Admin(client_id="asdf", client_secret="")
            assert admin is not None

    def test_initialization_missing_client_id_and_client_secret(self):
        with pytest.raises(ValueError):
            admin = Admin(client_id="", client_secret="")
            assert admin is not None
