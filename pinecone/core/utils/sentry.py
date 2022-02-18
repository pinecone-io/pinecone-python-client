#
# Copyright (c) 2020-2021 Pinecone Systems Inc. All right reserved.
#
import logging

import sentry_sdk
import inspect
from functools import wraps



_logger = logging.getLogger(__name__)


def sentry_decorator(func):
    @wraps(func)
    def inner_func(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise

    # Override signature
    sig = inspect.signature(func)
    inner_func.__signature__ = sig
    return inner_func
