import pinecone
import pytest


class TestIndexInstantiationUX:
    def test_index_instantiation_ux(self):
        with pytest.raises(Exception) as e:
            pinecone.Index(name="my-index", host="host")

        assert (
            "You are attempting to access the Index client directly from the pinecone module."
            in str(e.value)
        )
