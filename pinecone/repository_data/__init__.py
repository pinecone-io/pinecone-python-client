from .repository import Repository
from .models.document import Document


_Repository = Repository  # alias for backwards compatibility


__all__ = ["_Repository", "Document"]


def __getattr__(name):
    if name in locals():
        return locals()[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
