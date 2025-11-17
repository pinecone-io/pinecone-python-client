import functools
import inspect
from typing import TypeVar, Callable
from typing_extensions import ParamSpec

P = ParamSpec("P")
R = TypeVar("R")


def require_kwargs(func: Callable[P, R]) -> Callable[P, R]:
    """
    Decorator that requires all arguments (except self) to be passed as keyword arguments.

    :param func: The function to wrap
    :return: The wrapped function with the same signature
    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        if len(args) > 1:  # First arg is self
            param_names = list(inspect.signature(func).parameters.keys())[1:]  # Skip self
            raise TypeError(
                f"{func.__name__}() requires keyword arguments. "
                f"Please use {func.__name__}({', '.join(f'{name}=value' for name in param_names)})"
            )
        return func(*args, **kwargs)

    return wrapper
