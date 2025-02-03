import pytest


class TestIndexInstantiation:
    def test_invalid_host(self):
        from pinecone import Pinecone

        pc = Pinecone(api_key="key")

        with pytest.raises(ValueError) as e:
            pc.Index(host="invalid")
        assert "You passed 'invalid' as the host but this does not appear to be valid" in str(
            e.value
        )

        with pytest.raises(ValueError) as e:
            pc.Index(host="my-index")
        assert "You passed 'my-index' as the host but this does not appear to be valid" in str(
            e.value
        )

        # Can instantiate with realistic host
        pc.Index(host="test-bt8x3su.svc.apw5-4e34-81fa.pinecone.io")

        # Can instantiate with localhost address
        pc.Index(host="localhost:8080")
