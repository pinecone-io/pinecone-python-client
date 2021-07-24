#
# Copyright (c) 2020-2021 Pinecone Systems Inc. All right reserved.
#

from pinecone.utils.sentry import sentry_decorator as sentry
from .api_action import ActionAPI
from .constants import Config

import time
import requests

__all__ = ["version"]


def _get_action_api():
    return ActionAPI(host=Config.CONTROLLER_HOST, api_key=Config.API_KEY)


@sentry
def version():
    """Returns version information (client and server)."""
    api = _get_action_api()
    return api.version()


@sentry
def wait_controller_ready(timeout: int = 30):
    connection = False
    max_time = time.time() + timeout
    while (not connection) and time.time() < max_time:
        try:
            version()
            time.sleep(3)
            connection = True
        except requests.exceptions.ConnectionError:
            time.sleep(1)