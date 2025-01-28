import io
import json
import logging
import re
import ssl
import os
from urllib.parse import urlencode, quote
from abc import ABC, abstractmethod
import httpx
import weakref
import asyncio

import urllib3


from .exceptions import (
    PineconeApiException,
    UnauthorizedException,
    ForbiddenException,
    NotFoundException,
    ServiceException,
    PineconeApiValueError,
)

logging.basicConfig(
    format="%(levelname)s [%(asctime)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
)

logger = logging.getLogger(__name__)





# class HttpxRestClient(RestClientInterface):
#     def __init__(self, configuration, pools_size=4, maxsize=None, http2=False):
#         self.client = httpx.Client(http2=http2)

#     def close(self):
#         self.client.close()

#     def __del__(self):
#         self.close()

#     def request(
#         self,
#         method,
#         url,
#         query_params=None,
#         headers=None,
#         body=None,
#         post_params=None,
#         _preload_content=True,
#         _request_timeout=None,
#     ):
#         # print(f"Requesting {method} {url}")
#         # print(f"Query params: {query_params}")
#         # print(f"Headers: {headers}")
#         # print(f"Post params: {post_params}")
#         # print(f"Preload content: {_preload_content}")

#         if method in ["POST", "PUT", "PATCH", "OPTIONS"] and ("Content-Type" not in headers):
#             headers["Content-Type"] = "application/json"

#         r = httpx.Request(method=method, url=url, params=query_params, headers=headers, json=body)
#         resp = self.client.send(r)

#         # print(resp)
#         # print(resp.text)
#         # print(resp.content)
#         # print(resp.headers)
#         # print(resp.reason_phrase)

#         return raise_exceptions_or_return(
#             RESTResponse(resp.status_code, resp.content, resp.headers, resp.reason_phrase)
#         )


