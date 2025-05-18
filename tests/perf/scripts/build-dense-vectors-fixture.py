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


def build_random_df(num_rows=1000, dimension=1536):
    df = pd.DataFrame(columns=["id", "values", "metadata"])
    for i in range(num_rows):
        df.loc[i] = {
            "id": random_string(10),
            "values": random_embedding_values(dimension),
            "metadata": {"doc_id": str(uuid.uuid4()), "chunk_id": random_string(10)},
        }
    return df


def build_dense_fixture(num_rows, dimension):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    fixture_dir = os.path.join(current_dir, "..", "fixtures")
    filename = os.path.join(fixture_dir, f"dense_{num_rows}_{dimension}.parquet")
    df = build_random_df(num_rows, dimension)
    df.to_parquet(filename, index=False)
    return df


if __name__ == "__main__":
    build_dense_fixture(num_rows=100, dimension=768)
