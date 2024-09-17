import warnings
from functools import wraps
from typing import Optional, Callable


class PineconePrereleaseWarning(UserWarning):
    pass


warnings.simplefilter("once", PineconePrereleaseWarning)


def prerelease_feature(
    message: str = "This is a pre-release feature and may change in the future.", api_version: Optional[str] = None
):
    # The indirection of the function called decorator is needed for the
    # @prerelease_feature decorator to work correctly with or without arguments.
    def decorator(func: Callable):

        # The @wraps decorator is used to preserve the metadata of the original function when calling
        # .__name__ or .__docs__ on the decorated function.
        @wraps(func)
        def wrapper(*args, **kwargs):
            if api_version is None:
                combined_message = f"{message}"
            else:
                combined_message = f"{message} It is implemented against the {api_version} version of Pinecone's API."

            warnings.warn(combined_message, category=PineconePrereleaseWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper

    return decorator
