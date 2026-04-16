"""Docstring admonition lint tests for the preview namespace.

Every public class and method in ``pinecone.preview`` must include the
Preview admonition marker in its docstring, as required by
docs/conventions/preview-channel.md § User-facing signals.

These tests pass vacuously when no preview areas have been added yet —
they become load-bearing as new areas are introduced.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil

ADMONITION_MARKER = "Preview surface is not covered by SemVer"

# Namespace router classes — they have no area methods yet and are excluded
# from the method check (but ARE included in the class check).
_NAMESPACE_ROUTER_CLASSES = {"Preview", "AsyncPreview"}


def _discover_preview_public_classes() -> list[tuple[str, type]]:
    """Find all public classes in pinecone.preview (excluding _internal)."""
    import pinecone.preview as root

    classes: list[tuple[str, type]] = []

    for info in pkgutil.walk_packages(root.__path__, prefix="pinecone.preview."):
        if "._internal" in info.name:
            continue
        try:
            mod = importlib.import_module(info.name)
        except ImportError:
            continue
        for attr_name in dir(mod):
            if attr_name.startswith("_"):
                continue
            obj = getattr(mod, attr_name)
            if inspect.isclass(obj) and obj.__module__ == info.name:
                classes.append((f"{info.name}.{attr_name}", obj))
    return classes


def test_all_preview_public_classes_have_admonition() -> None:
    """Every public class in pinecone.preview must have the preview admonition."""
    missing: list[str] = []
    for qualname, cls in _discover_preview_public_classes():
        doc = cls.__doc__ or ""
        if ADMONITION_MARKER not in doc:
            missing.append(qualname)

    assert missing == [], (
        f"Preview classes missing the docstring admonition: {missing}. "
        f"Every public preview class must include '{ADMONITION_MARKER}' in its docstring. "
        "See docs/conventions/preview-channel.md § User-facing signals."
    )


def test_all_preview_public_methods_have_admonition() -> None:
    """Every public method on preview area classes must have the preview admonition.

    The ``Preview`` and ``AsyncPreview`` namespace router classes are excluded
    because they act as routers with no area methods.
    """
    # Skip namespace router classes — they have no area methods yet.
    area_classes = [
        (qualname, cls)
        for qualname, cls in _discover_preview_public_classes()
        if cls.__name__ not in _NAMESPACE_ROUTER_CLASSES
    ]

    missing: list[str] = []
    for _qualname, cls in area_classes:
        for method_name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            if method_name.startswith("_"):
                continue
            doc = method.__doc__ or ""
            if ADMONITION_MARKER not in doc:
                missing.append(f"{cls.__name__}.{method_name}")

    assert missing == [], (
        f"Preview methods missing the docstring admonition: {missing}. "
        f"Every public preview method must include '{ADMONITION_MARKER}' in its docstring. "
        "See docs/conventions/preview-channel.md § User-facing signals."
    )
