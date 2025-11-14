import os
import random
import string
import uuid
from pinecone import Pinecone
from datetime import datetime


def read_env_var(name):
    value = os.environ.get(name)
    if value is None:
        raise "Environment variable {} is not set".format(name)
    return value


def random_string(length):
    return "".join(random.choice(string.ascii_lowercase) for i in range(length))


def write_gh_output(name, value):
    with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
        print(f"{name}={value}", file=fh)


def generate_index_name(name_prefix: str) -> str:
    name = name_prefix.lower() + "-" + str(uuid.uuid4())
    return name[:45]


def get_tags():
    github_actor = os.getenv("GITHUB_ACTOR", None)
    user = os.getenv("USER", None)
    index_owner = github_actor or user or "unknown"

    github_job = os.getenv("GITHUB_JOB", "")
    tags = {
        "owner": index_owner,
        "test-suite": "pinecone-python-client",
        "created-at": datetime.now().strftime("%Y-%m-%d"),
        "test-job": github_job,
    }
    return tags


def main():
    pc = Pinecone(api_key=read_env_var("PINECONE_API_KEY"))
    index_name = generate_index_name(read_env_var("NAME_PREFIX"))
    dimension_var = read_env_var("DIMENSION")
    if dimension_var is not None and dimension_var != "":
        dimension = int(dimension_var)
    else:
        dimension = None

    vector_type_var = read_env_var("VECTOR_TYPE")
    if vector_type_var is not None and vector_type_var != "":
        vector_type = vector_type_var
    else:
        vector_type = None

    metric = read_env_var("METRIC")
    cloud = read_env_var("CLOUD")
    region = read_env_var("REGION")
    tags = get_tags()

    pc.create_index(
        name=index_name,
        metric=metric,
        dimension=dimension,
        vector_type=vector_type,
        tags=tags,
        spec={"serverless": {"cloud": cloud, "region": region}},
    )
    description = pc.describe_index(name=index_name)
    write_gh_output("index_name", index_name)
    write_gh_output("index_host", description.host)


if __name__ == "__main__":
    main()
