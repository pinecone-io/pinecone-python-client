"""
Lazy import handler for Pinecone.

This module provides a way to lazily load imports that were previously
available via star imports but are no longer imported at the top level.
"""

import importlib
import sys
from types import ModuleType
from typing import cast

# Dictionary mapping import names to their actual module paths
# Format: 'name': ('module_path', 'actual_name')
LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Example: 'Vector': ('pinecone.db_data.models', 'Vector')
    # Add all your lazy imports here
}


class LazyModule:
    def __init__(self, original_module, lazy_imports):
        self._original_module = original_module
        self._lazy_imports = lazy_imports
        self._loaded_attrs = {}

    def __getattribute__(self, name):
        if name == "__doc__":
            return object.__getattribute__(self, "_original_module").__doc__
        if name == "__dict__":
            # Get the base dictionary from the original module
            base_dict = object.__getattribute__(self, "_original_module").__dict__.copy()
            # Add lazy-loaded items
            loaded_attrs = object.__getattribute__(self, "_loaded_attrs")
            for name, value in loaded_attrs.items():
                base_dict[name] = value
            return base_dict
        return object.__getattribute__(self, name)

    def __dir__(self):
        # Get the base directory listing from the original module
        base_dir = dir(self._original_module)

        # Add lazy-loaded items
        lazy_dir = list(self._lazy_imports.keys())

        # Return combined list
        return sorted(set(base_dir + lazy_dir))

    def __getattr__(self, name):
        # First try the original module
        try:
            return getattr(self._original_module, name)
        except AttributeError:
            pass

        # Then try lazy imports
        if name in self._lazy_imports:
            if name not in self._loaded_attrs:
                module_path, item_name = self._lazy_imports[name]
                module = importlib.import_module(module_path)
                self._loaded_attrs[name] = getattr(module, item_name)
            return self._loaded_attrs[name]

        raise AttributeError(f"module '{self._original_module.__name__}' has no attribute '{name}'")


def setup_lazy_imports(lazy_imports: dict[str, tuple[str, str]] | None = None) -> None:
    """
    Set up the lazy import handler.

    Args:
        lazy_imports: Optional dictionary of imports to handle lazily.
                     If None, uses the default LAZY_IMPORTS dictionary.
    """
    if lazy_imports is None:
        lazy_imports = LAZY_IMPORTS

    # Only proceed if the pinecone module is already loaded
    if "pinecone" not in sys.modules:
        return

    # Create a proxy for the pinecone module
    original_module = sys.modules["pinecone"]
    proxy = LazyModule(original_module, lazy_imports)

    # Replace the pinecone module with our proxy
    # Use a type cast to satisfy the type checker
    sys.modules["pinecone"] = cast(ModuleType, proxy)
