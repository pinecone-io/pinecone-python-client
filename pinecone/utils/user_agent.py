import requests
import urllib3

from pinecone.utils import get_version

def get_user_agent():
    client_id = f"python-client-{get_version()}"
    user_agent_details = {"requests": requests.__version__, "urllib3": urllib3.__version__}
    user_agent = "{} ({})".format(client_id, ", ".join([f"{k}:{v}" for k, v in user_agent_details.items()]))
    return user_agent