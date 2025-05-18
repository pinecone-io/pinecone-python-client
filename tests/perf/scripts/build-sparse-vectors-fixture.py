import random
import string
import pandas as pd
import numpy as np
import os
import uuid


def random_string(length):
    return "".join(random.choice(string.ascii_lowercase) for i in range(length))


def random_embedding_values(dimension):
    return np.random.rand(dimension).tolist()


def build_random_df(num_rows=1000):
    df = pd.DataFrame(columns=["id", "values", "sparse_indices", "sparse_values", "metadata"])
    for i in range(num_rows):
        num_elements = random.randint(50, 100)
        df.loc[i] = {
            "id": random_string(10),
            "values": [],
            "sparse_indices": [random.randint(1, 100000) for _ in range(num_elements)],
            "sparse_values": [random.random() for _ in range(num_elements)],
            "metadata": {"doc_id": str(uuid.uuid4()), "chunk_id": random_string(10)},
        }
    return df


def build_sparse_fixture(num_rows):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    fixture_dir = os.path.join(current_dir, "..", "fixtures")
    filename = os.path.join(fixture_dir, f"sparse_{num_rows}.parquet")
    df = build_random_df(num_rows)
    df.to_parquet(filename, index=False)
    return df


if __name__ == "__main__":
    build_sparse_fixture(num_rows=100)
