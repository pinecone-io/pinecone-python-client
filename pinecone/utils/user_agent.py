import urllib3

from .version import __version__
from .constants import SOURCE_INTEGRATION

def _build_source_integration_identifier(config):
    identifier = f"{SOURCE_INTEGRATION}={config.source_integration_name}"
    identifier += f":{config.source_integration_version}" if config.source_integration_version else ""
    return identifier

def get_user_agent(config):
    client_id = f"python-client-{__version__}"
    user_agent_details = {"urllib3": urllib3.__version__}
    user_agent = "{} ({})".format(client_id, ", ".join([f"{k}:{v}" for k, v in user_agent_details.items()]))
    user_agent += f"; {_build_source_integration_identifier(config)}" if config.source_integration_name else ""
    return user_agent