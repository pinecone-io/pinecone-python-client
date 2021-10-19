#
# Copyright (c) 2020-2021 Pinecone Systems Inc. All right reserved.
#
import logging

import dns.resolver
import sentry_sdk
import os
import json
import inspect
from functools import wraps

from pinecone.core.utils.constants import PACKAGE_ENVIRONMENT, SENTRY_DSN_TXT_RECORD


_logger = logging.getLogger(__name__)


def sentry_decorator(func):
    @wraps(func)
    def inner_func(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            init_sentry()
            sentry_sdk.capture_exception(e)
            raise

    # Override signature
    sig = inspect.signature(func)
    inner_func.__signature__ = sig
    return inner_func


def init_sentry():
    """Init Sentry if necessary.

    The Sentry DSN is stored as a txt record.
    """
    if not sentry_sdk.Hub.current.client:
        _logger.info("Sentry is not initialized.")
        # sentry is not initialized
        sentry_dsn = None
        try:
            dns_result = dns.resolver.resolve(SENTRY_DSN_TXT_RECORD, "TXT")
            for res in dns_result:
                sentry_dsn = json.loads(res.to_text())
                break
        except Exception:
            _logger.warning("Unable to resolve Sentry DSN.")
        if sentry_dsn:
            debug = os.getenv("SENTRY_DEBUG") == "True"
            sentry_sdk.init(dsn=sentry_dsn, debug=debug, environment=PACKAGE_ENVIRONMENT, traces_sample_rate=0.1)
    return None
