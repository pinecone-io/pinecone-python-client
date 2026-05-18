#!/usr/bin/env python3
"""Generate ``pinecone/__init__.pyi`` from ``_LAZY_IMPORTS`` in ``pinecone/__init__.py``.

Keeps the stub provably in sync with the runtime lazy-loader: every symbol in
``_LAZY_IMPORTS`` that is also listed in ``__all__`` gets an explicit re-export line
in the stub, which lets Jedi (marimo, PyCharm CE, older ``jedi-language-server``)
resolve types instead of falling back to ``Any``.

Usage examples:

    # Write (or overwrite) the stub
    uv run python scripts/generate_init_stub.py

    # Check that the on-disk stub is up to date; exit 1 with a diff if not
    uv run python scripts/generate_init_stub.py --check

Run after any change to ``_LAZY_IMPORTS`` or ``__all__`` in ``pinecone/__init__.py``.
"""

from __future__ import annotations

import argparse
import difflib
import pathlib
import sys

STUB_PATH = pathlib.Path(__file__).resolve().parent.parent / "pinecone" / "__init__.pyi"

_STUB_HEADER = '''\
"""Auto-generated stub for pinecone/__init__.py — do not edit manually.

Regenerate after changes to _LAZY_IMPORTS or __all__ in pinecone/__init__.py:
    uv run python scripts/generate_init_stub.py
"""
'''


def _generate() -> str:
    import pinecone

    lazy_imports: dict[str, tuple[str, str]] = pinecone._LAZY_IMPORTS  # type: ignore[attr-defined]
    all_names: list[str] = list(pinecone.__all__)

    all_set = set(all_names)

    # Group by source module: module -> [(source_attr, exported_name)]
    by_module: dict[str, list[tuple[str, str]]] = {}
    for exported_name, (module_path, source_attr) in lazy_imports.items():
        if exported_name not in all_set:
            continue  # defensive: skip anything not in __all__
        by_module.setdefault(module_path, []).append((source_attr, exported_name))

    parts: list[str] = [_STUB_HEADER]

    for module in sorted(by_module):
        pairs = sorted(by_module[module], key=lambda p: p[1])  # sort by exported name
        imports = ", ".join(f"{src} as {exp}" for src, exp in pairs)
        parts.append(f"from {module} import {imports}\n")

    parts.append("\n__version__: str\n")

    parts.append("\n__all__ = [\n")
    parts.extend(f'    "{name}",\n' for name in sorted(all_names))
    parts.append("]\n")

    return "".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate pinecone/__init__.pyi from _LAZY_IMPORTS."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 (with diff to stderr) if the on-disk stub differs from the generated output.",
    )
    args = parser.parse_args(argv)

    content = _generate()

    if args.check:
        if not STUB_PATH.exists():
            print(f"error: stub not found at {STUB_PATH}", file=sys.stderr)
            return 1
        existing = STUB_PATH.read_text()
        if existing == content:
            return 0
        diff = difflib.unified_diff(
            existing.splitlines(keepends=True),
            content.splitlines(keepends=True),
            fromfile=str(STUB_PATH),
            tofile="<generated>",
        )
        sys.stderr.writelines(diff)
        return 1

    STUB_PATH.write_text(content)
    print(f"Wrote {STUB_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
