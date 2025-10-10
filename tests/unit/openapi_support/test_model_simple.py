from pinecone.core.openapi.db_data.models import VectorValues


def test_simple_model_instantiation():
    vv = VectorValues(value=[1.0, 2.0, 3.0])
    assert vv.value == [1.0, 2.0, 3.0]
