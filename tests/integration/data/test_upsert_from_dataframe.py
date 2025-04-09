import pandas as pd
from pinecone.db_data import _Index
from ..helpers import embedding_values, random_string


class TestUpsertFromDataFrame:
    def test_upsert_from_dataframe(self, idx: _Index):
        # Create sample data for testing.
        data = {
            "id": ["1", "2", "3"],
            "values": [embedding_values(), embedding_values(), embedding_values()],
            "sparse_values": [
                {"indices": [1], "values": [0.234]},
                {"indices": [2], "values": [0.432]},
                {"indices": [3], "values": [0.543]},
            ],
            "metadata": [
                {"source": "generated", "quality": "high"},
                {"source": "generated", "quality": "medium"},
                {"source": "generated", "quality": "low"},
            ],
        }

        # Create the DataFrame
        df = pd.DataFrame(data)

        ns = random_string(10)
        idx.upsert_from_dataframe(df=df, namespace=ns)
