import re
import pinecone

def test_version():
    assert re.search(r"\d+\.\d+\.\d+", pinecone.__version__) is not None