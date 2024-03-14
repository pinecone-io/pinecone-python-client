import urllib3

from .version import __version__

def get_user_agent(config):
    client_id = f"python-client-{__version__}"
    user_agent_details = {"urllib3": urllib3.__version__, "source_partner": config.source_partner}
    user_agent = "{} ({})".format(client_id, ", ".join([f"{k}:{v}" for k, v in user_agent_details.items()]))
    return user_agent