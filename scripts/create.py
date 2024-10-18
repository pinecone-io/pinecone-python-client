import os
import re
import random
import string
from datetime import datetime
from pinecone import Pinecone


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


def generate_index_name(test_name: str) -> str:
    github_actor = os.getenv("GITHUB_ACTOR", None)
    user = os.getenv("USER", None)
    index_owner = github_actor or user

    formatted_date = datetime.now().strftime("%Y%m%d-%H%M%S%f")[:-3]

    github_job = os.getenv("GITHUB_JOB", None)

    if test_name.startswith("test_"):
        test_name = test_name[5:]

    # Remove trailing underscore, if any
    if test_name.endswith("_"):
        test_name = test_name[:-1]

    name_parts = [index_owner, formatted_date, github_job, test_name]
    index_name = "-".join([x for x in name_parts if x is not None])

    # Remove invalid characters
    replace_with_hyphen = re.compile(r"[\[\(_,\s]")
    index_name = re.sub(replace_with_hyphen, "-", index_name)
    replace_with_empty = re.compile(r"[\]\)\.]")
    index_name = re.sub(replace_with_empty, "", index_name)

    max_length = 45
    index_name = index_name[:max_length]

    # Trim final character if it is not alphanumeric
    if index_name.endswith("_") or index_name.endswith("-"):
        index_name = index_name[:-1]

    return index_name.lower()


def main():
    pc = Pinecone(api_key=read_env_var("PINECONE_API_KEY"))

    index_name = generate_index_name(read_env_var("NAME_PREFIX") + random_string(20))
    dimension = int(read_env_var("DIMENSION"))
    metric = read_env_var("METRIC")

    pc.create_index(
        name=index_name,
        metric=metric,
        dimension=dimension,
        spec={"serverless": {"cloud": read_env_var("CLOUD"), "region": read_env_var("REGION")}},
    )
    desc = pc.describe_index(index_name)
    write_gh_output("index_name", index_name)
    write_gh_output("index_host", desc.host)
    write_gh_output("index_metric", metric)
    write_gh_output("index_dimension", dimension)


if __name__ == "__main__":
    main()
