import time
import os
import pytest
import subprocess
from ..helpers import get_environment_var

PROXIES = {
    "proxy1": {
        "name": "proxy1",
        "port": 8080,
        "ssl_ca_certs": os.path.abspath("./tests/integration/proxy_config/.mitm/proxy1"),
        "auth": None,
    },
    "proxy2": {
        "name": "proxy2",
        "port": 8081,
        "ssl_ca_certs": os.path.abspath("./tests/integration/proxy_config/.mitm/proxy2"),
        "auth": ("testuser", "testpassword"),
    },
}


def docker_command(proxy):
    cmd = [
        "docker",
        "run",
        "-d",  # detach to run in background
        "--rm",  # remove container when stopped
        "--name",
        proxy["name"],  # name the container
        "-p",
        f"{proxy['port']}:8080",  # map the port
        "-v",
        f"{proxy['ssl_ca_certs']}:/home/mitmproxy/.mitmproxy",  # mount config as volume
        "mitmproxy/mitmproxy",  # docker image name
        "mitmdump",  # command to run
    ]
    if proxy["auth"]:
        cmd.append(f"--set proxyauth={proxy['auth'][0]}:{proxy['auth'][1]}")
    print(" ".join(cmd))
    return " ".join(cmd)


def run_cmd(cmd, output):
    output.write("Going to run: " + cmd + "\n")
    exit_code = subprocess.call(cmd, shell=True, stdout=output, stderr=output)
    if exit_code != 0:
        raise Exception(f"Failed to run command: {cmd}")


def use_grpc():
    return os.environ.get("USE_GRPC", "false") == "true"


@pytest.fixture(scope="session", autouse=True)
def start_docker():
    with open("tests/integration/proxy_config/logs/proxyconfig-docker-start.log", "a") as output:
        run_cmd(docker_command(PROXIES["proxy1"]), output)
        run_cmd(docker_command(PROXIES["proxy2"]), output)

    time.sleep(5)
    with open("tests/integration/proxy_config/logs/proxyconfig-docker-ps.log", "a") as output:
        run_cmd("docker ps --all", output)

    yield
    with open("tests/integration/proxy_config/logs/proxyconfig-docker-stop.log", "a") as output:
        run_cmd("docker stop proxy1", output)
        run_cmd("docker stop proxy2", output)


@pytest.fixture()
def proxy1():
    return PROXIES["proxy1"]


@pytest.fixture()
def proxy2():
    return PROXIES["proxy2"]


@pytest.fixture()
def client_cls():
    if use_grpc():
        from pinecone.grpc import PineconeGRPC

        return PineconeGRPC
    else:
        from pinecone import Pinecone

        return Pinecone


@pytest.fixture()
def api_key():
    return get_environment_var("PINECONE_API_KEY")


@pytest.fixture()
def index_name():
    return get_environment_var("PINECONE_INDEX_NAME")
