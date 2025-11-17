from __future__ import annotations

import inspect
import logging
from typing import Callable, Any


def check_kwargs(caller: Callable[..., Any], given: set[str]) -> None:
    argspec = inspect.getfullargspec(caller)
    diff = set(given).difference(argspec.args)
    if diff:
        logging.exception(
            caller.__name__ + " had unexpected keyword argument(s): " + ", ".join(diff),
            exc_info=False,
        )
