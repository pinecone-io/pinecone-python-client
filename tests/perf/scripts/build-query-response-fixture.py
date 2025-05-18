import random
import string
import pandas as pd
import os
import uuid


def random_string(length):
    return "".join(random.choice(string.ascii_lowercase) for i in range(length))


def build_random_df(num_rows, dimension):
    matches = []
    for i in range(num_rows):
        matches.append({"id": f"id{i}", "score": random.random(), "values": []})
    matches.sort(key=lambda x: x["score"], reverse=True)

    df = pd.DataFrame(columns=["id", "score", "values", "metadata"])
    for i in range(num_rows):
        df.loc[i] = {
            "id": matches[i]["id"],
            "score": matches[i]["score"],
            "values": matches[i]["values"],
            "metadata": {"doc_id": str(uuid.uuid4()), "chunk_id": random_string(10)},
        }
    return df


def build_query_matches_fixture(num_rows, dimension, filename):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    fixture_dir = os.path.join(current_dir, "..", "fixtures")
    filename = os.path.join(fixture_dir, filename)
    df = build_random_df(num_rows, dimension)
    df.to_parquet(filename, index=False)
    return df


if __name__ == "__main__":
    for ns in range(10):
        build_query_matches_fixture(
            num_rows=100, dimension=768, filename=f"query_matches_{ns}_100_768.parquet"
        )
