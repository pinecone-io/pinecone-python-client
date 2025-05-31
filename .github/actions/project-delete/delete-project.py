import json
import os
import urllib3
import logging

logging.basicConfig(
    level=logging.DEBUG, format="%(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
)
logger = logging.getLogger(__name__)


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

    def delete_project(self, project_id):
        response = self._request("DELETE", f"https://api.pinecone.io/admin/projects/{project_id}")
        return response


client_id = os.getenv("PINECONE_SERVICE_ACCOUNT_CLIENT_ID")
client_secret = os.getenv("PINECONE_SERVICE_ACCOUNT_CLIENT_SECRET")

if client_id is None:
    raise Exception("PINECONE_SERVICE_ACCOUNT_CLIENT_ID must be set")
if client_secret is None:
    raise Exception("PINECONE_SERVICE_ACCOUNT_CLIENT_SECRET must be set")

admin_api = AdminAPI(client_id, client_secret)

project_id = os.getenv("PROJECT_ID")
if project_id is None:
    raise Exception("PROJECT_ID must be set")

admin_api.delete_project(project_id)
