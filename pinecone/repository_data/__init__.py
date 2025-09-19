from .repository import Repository


_Repository = Repository  # alias for backwards compatibility


__all__ = ["_Repository"]


def __getattr__(name):
    if name in locals():
        return locals()[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
