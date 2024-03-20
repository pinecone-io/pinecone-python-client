import time
import os
import pytest
import subprocess
from ..helpers import get_environment_var

PROXIES = {
    'proxy1': {
        'name': 'proxy1',
        'port': 8080,
        'ssl_ca_certs': os.path.abspath('./tests/integration/proxy_config/.mitm/proxy1'),
        'auth': None
    },
    'proxy2': {
        'name': 'proxy2',
        'port': 8081, 
        'ssl_ca_certs': os.path.abspath('./tests/integration/proxy_config/.mitm/proxy2'),
        'auth': ('testuser', 'testpassword')
    }
}

def docker_command(proxy):
    cmd = [
        "docker", "run", "-d", # detach to run in background
        "--rm", # remove container when stopped
        "--name", proxy['name'],  # name the container
        "-p", f"{proxy['port']}:8080", # map the port
        "-v", f"{proxy['ssl_ca_certs']}:/home/mitmproxy/.mitmproxy", # mount config as volume 
        "mitmproxy/mitmproxy", # docker image name
        "mitmdump" # command to run
    ]
    if proxy['auth']:
        cmd.append(f"--set proxyauth={proxy['auth'][0]}:{proxy['auth'][1]}")
    print(" ".join(cmd))
    return " ".join(cmd)
    
@pytest.fixture(scope='session', autouse=True)
def start_docker():
    with open("/tmp/proxyconfig-docker-start.log", "a") as output:
        subprocess.call(docker_command(PROXIES['proxy1']), shell=True, stdout=output, stderr=output)
        subprocess.call(docker_command(PROXIES['proxy2']), shell=True, stdout=output, stderr=output)
    time.sleep(2)
    yield
    with open("/tmp/proxyconfig-docker-stop.log", "a") as output:
        subprocess.call("docker stop proxy1", shell=True, stdout=output, stderr=output)
        subprocess.call("docker stop proxy2", shell=True, stdout=output, stderr=output)

@pytest.fixture()
def proxy1():
    return PROXIES['proxy1']

@pytest.fixture()
def proxy2():
    return PROXIES['proxy2']

@pytest.fixture()
def api_key():
    return get_environment_var('PINECONE_API_KEY')

@pytest.fixture()
def index_name():
    return get_environment_var('PINECONE_INDEX_NAME')