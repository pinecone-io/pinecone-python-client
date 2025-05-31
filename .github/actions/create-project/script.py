import json
import os
import urllib3
import logging
from datetime import datetime

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


class AdminAPI:
    def __init__(self, client_id, client_secret):
        self.http = urllib3.PoolManager()
        self.token = None
        self.token = self._get_token(client_id, client_secret)

    def _request(self, method, url, headers={}, body_dict=None):
        logger.info(f"Requesting {method} {url} with body {body_dict}")
        api_version = os.environ.get("API_VERSION", "2025-04")
        default_headers = {
            "X-Pinecone-Api-Version": api_version,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.token is not None:
            default_headers["Authorization"] = f"Bearer {self.token}"
        headers = {**default_headers, **headers}

        args = {"method": method, "url": url, "headers": headers}
        if body_dict is not None:
            args["body"] = json.dumps(body_dict)
        response = self.http.request(**args)

        logger.info(f"Response Status: {response.status}")
        if response.status >= 400:
            raise Exception(
                f"Request failed with status {response.status}: {response.data.decode('utf-8')}"
            )

        if response is None or response.data is None or response.data == b"":
            return None
        return json.loads(response.data.decode("utf-8"))

    def _get_token(self, client_id, client_secret):
        response = self._request(
            "POST",
            "https://login.pinecone.io/oauth/token",
            body_dict={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
                "audience": "https://api.pinecone.io/",
            },
        )

        token = response["access_token"]
        mask(token)
        return token

    def create_project(self, project_name, max_pods):
        response = self._request(
            "POST",
            "https://api.pinecone.io/admin/projects",
            body_dict={"name": project_name, "max_pods": max_pods},
        )
        return response

    def create_api_key(self, project_id):
        response = self._request(
            "POST",
            f"https://api.pinecone.io/admin/projects/{project_id}/api-keys",
            body_dict={"name": "ci-key"},
        )
        return response

    def list_projects(self):
        response = self._request("GET", "https://api.pinecone.io/admin/projects")
        return response

    def describe_project(self, project_id):
        response = self._request("GET", f"https://api.pinecone.io/admin/projects/{project_id}")
        return response

    def delete_project(self, project_id):
        response = self._request("DELETE", f"https://api.pinecone.io/admin/projects/{project_id}")
        return response

    def get_project_id(self, project_name):
        project_list = self.list_projects()["data"]
        for project in project_list:
            if project["name"] == project_name:
                return project["id"]
        return None


def main():
    client_id = os.getenv("PINECONE_SERVICE_ACCOUNT_CLIENT_ID")
    client_secret = os.getenv("PINECONE_SERVICE_ACCOUNT_CLIENT_SECRET")

    if client_id is None:
        raise Exception("PINECONE_SERVICE_ACCOUNT_CLIENT_ID must be set")
    if client_secret is None:
        raise Exception("PINECONE_SERVICE_ACCOUNT_CLIENT_SECRET must be set")

    admin_api = AdminAPI(client_id, client_secret)

    project_name = generate_project_name()
    max_pods = int(os.getenv("MAX_PODS", 1))
    project_id = admin_api.create_project(project_name, max_pods)["id"]
    project_api_key = admin_api.create_api_key(project_id)["value"]
    mask(project_api_key)

    output_file = os.environ.get("GITHUB_OUTPUT", None)
    if output_file is None:
        logger.error("GITHUB_OUTPUT is not set, cannot write to output file")
    else:
        with open(output_file, "a") as f:
            f.write(f"project_name={project_name}\n")
            f.write(f"project_id={project_id}\n")
            f.write(f"project_api_key={project_api_key}\n")


if __name__ == "__main__":
    main()
