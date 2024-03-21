import urllib3

from .version import __version__
from .constants import SOURCE_TAG
import re

def _build_source_tag_field(source_tag):
    # normalize source tag
    # 1. Lowercase
    # 2. Trim left/right whitespace
    # 3. Condense multiple spaces to one, and replace with underscore
    # 4. Limit charset to [a-z0-9_]
    tag = source_tag.lower().strip()
    tag = "_".join(tag.split())
    tag = re.sub(r'[^a-z0-9_]', '', tag)
    return f"{SOURCE_TAG}={tag}"

def get_user_agent(config):
    client_id = f"python-client-{__version__}"
    user_agent_details = {"urllib3": urllib3.__version__}
    user_agent = "{} ({})".format(client_id, ", ".join([f"{k}:{v}" for k, v in user_agent_details.items()]))
    user_agent += f"; {_build_source_tag_field(config.source_tag)}" if config.source_tag else ""
    return user_agent