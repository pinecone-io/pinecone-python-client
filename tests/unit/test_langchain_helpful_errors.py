import pytest
from pinecone import Pinecone


class TestLangchainErrorMessages:
    def test_error_from_texts_positional_args(self):
        with pytest.raises(AttributeError) as e:
            Pinecone.from_texts("texts", "id")
        assert "from_texts is not a top-level attribute of the Pinecone class" in str(e.value)

    def test_error_from_texts_kwargs(self):
        with pytest.raises(AttributeError) as e:
            Pinecone.from_texts(foo="texts", bar="id", num_threads=1)
        assert "from_texts is not a top-level attribute of the Pinecone class" in str(e.value)

    def test_error_from_documents(self):
        with pytest.raises(AttributeError) as e:
            Pinecone.from_documents("documents", "id")
        assert "from_documents is not a top-level attribute of the Pinecone class" in str(e.value)
