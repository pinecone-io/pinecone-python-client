import pytest

from pinecone.data.index import _Index


class TestUpsert:
    def test_upsert_informative_error_when_unknown_kwarg(self):
        with pytest.raises(ValueError) as e:
            _Index("api-key", "host").update(id="1", value="value")
        assert "Unexpected keyword arguments: value" in str(e.value)
