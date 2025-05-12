import pytest


class TestReorganization:
    def test_data(self):
        with pytest.warns(DeprecationWarning) as warning_info:
            from pinecone.data import Index

            assert Index is not None
            assert len(warning_info) > 0
            assert "has moved to" in str(warning_info[0].message)

    def test_config(self):
        with pytest.warns(DeprecationWarning) as warning_info:
            from pinecone.config import PineconeConfig

            assert PineconeConfig is not None
            assert len(warning_info) > 0
            assert "has moved to" in str(warning_info[0].message)
