import os
import logging
from datetime import datetime
from pinecone import Admin

logging.basicConfig(
    level=logging.DEBUG, format="%(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
)
logger = logging.getLogger(__name__)


def generate_project_name():
    github_actor = os.getenv("GITHUB_ACTOR", None)
    user = os.getenv("USER", None)
    owner = github_actor or user

    name_prefix = os.getenv("NAME_PREFIX", None)
    formatted_date = datetime.now().strftime("%Y%m%d-%H%M%S-%f")

    name_parts = [name_prefix, owner, formatted_date]
    return "-".join([x for x in name_parts if x is not None])


def mask(value):
    """Mask the value in Github Actions logs"""
    print(f"::add-mask::{value}")


def main():
    client_id = os.getenv("PINECONE_SERVICE_ACCOUNT_CLIENT_ID")
    client_secret = os.getenv("PINECONE_SERVICE_ACCOUNT_CLIENT_SECRET")

    if client_id is None:
        raise Exception("PINECONE_SERVICE_ACCOUNT_CLIENT_ID must be set")
    if client_secret is None:
        raise Exception("PINECONE_SERVICE_ACCOUNT_CLIENT_SECRET must be set")

    admin_api = Admin(client_id, client_secret)

    project_name = generate_project_name()
    max_pods = int(os.getenv("MAX_PODS", 1))
    project = admin_api.project.create(name=project_name, max_pods=max_pods)
    project_api_key = admin_api.api_key.create(project_id=project.id).value
    mask(project_api_key)

    output_file = os.environ.get("GITHUB_OUTPUT", None)
    if output_file is None:
        logger.error("GITHUB_OUTPUT is not set, cannot write to output file")
    else:
        with open(output_file, "a") as f:
            f.write(f"project_name={project_name}\n")
            f.write(f"project_id={project.id}\n")
            f.write(f"project_api_key={project_api_key}\n")


if __name__ == "__main__":
    main()
