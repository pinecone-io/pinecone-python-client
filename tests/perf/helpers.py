import os
import pandas as pd


def load_fixture(fixture_name):
    full_path = os.path.join(os.path.dirname(__file__), "fixtures", fixture_name)
    df = pd.read_parquet(full_path)
    vectors = df.to_dict(orient="records")
    return vectors
