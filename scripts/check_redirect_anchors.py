#!/usr/bin/env python3
"""Guard the redirect stub's anchor tables against pointing at nothing.

``docs/_templates/redirect.html`` rewrites a deep link's fragment as it
forwards, naming a ``reference/`` page and usually an id on it. A stale entry
fails silently: the browser loads the page and simply does not scroll, so
neither the build nor anyone but the reader who followed that link finds out.

This validates the entries that are there. It cannot know an anchor is missing
one it ought to have.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = REPO_ROOT / "docs" / "_templates" / "redirect.html"
DEFAULT_BUILD_DIR = REPO_ROOT / "docs" / "_build" / "html"
REFERENCE = "reference/"

# The tables hold plain double-quoted literals with no escapes, so reading them
# does not need a JS parser.
_STR = r'"([^"]*)"'


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _strip_comments(js: str) -> str:
    return re.sub(r"^\s*//.*$", "", js, flags=re.MULTILINE)


def _table(js: str, name: str) -> str:
    """Return the literal assigned to ``var <name>``, brackets included."""
    match = re.search(rf"var\s+{name}\s*=\s*([\[{{])", js)
    if match is None:
        raise ValueError(f"{name} table not found in {TEMPLATE.name}")
    opener = match.group(1)
    closer = {"[": "]", "{": "}"}[opener]
    depth, start = 0, match.end() - 1
    for i in range(start, len(js)):
        if js[i] == opener:
            depth += 1
        elif js[i] == closer:
            depth -= 1
            if depth == 0:
                return js[start : i + 1]
    raise ValueError(f"{name} table is not closed in {TEMPLATE.name}")


def parse_destinations(template_text: str) -> list[tuple[str, str, str | None]]:
    """Extract (source, page, fragment) for every rewrite the template can do.

    ``page`` is relative to ``reference/``. ``fragment`` is None when the entry
    names only a page, landing the reader at the top of it.
    """
    js = _strip_comments(template_text)
    out: list[tuple[str, str, str | None]] = []

    for old, page, new in re.findall(
        rf"\[\s*{_STR}\s*,\s*{_STR}\s*,\s*{_STR}\s*\]", _table(js, "classes")
    ):
        out.append((f"classes[{old}]", page, new))

    for _, dest in re.findall(rf"{_STR}\s*:\s*{_STR}", _table(js, "sections")):
        page, _, fragment = dest.partition("#")
        out.append((f"sections[{dest}]", page, fragment or None))

    # The method ids are assembled at runtime from three tables, so rebuild
    # every id that can be produced rather than trusting the parts separately.
    verbs = re.findall(rf"{_STR}\s*:\s*\[\s*{_STR}\s*,\s*{_STR}\s*\]", _table(js, "verbs"))
    resources = dict(re.findall(rf"{_STR}\s*:\s*{_STR}", _table(js, "resources")))
    clients = re.findall(
        rf"\[\s*{_STR}\s*,\s*{_STR}\s*,\s*{_STR}\s*,\s*{_STR}\s*\]", _table(js, "clients")
    )
    for old_class, page, module, class_prefix in clients:
        for verb, resource, member in verbs:
            if resource not in resources:
                raise ValueError(f"verbs names resource {resource!r}, absent from resources")
            new_id = f"{module}.{resource}.{class_prefix}{resources[resource]}.{member}"
            out.append((f"verbs[{old_class}.{verb}]", page, new_id))

    return out


def ids_on_page(path: Path) -> set[str]:
    return set(re.findall(r'id="([^"]+)"', path.read_text(encoding="utf-8")))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "build_dir",
        nargs="?",
        default=DEFAULT_BUILD_DIR,
        type=Path,
        help="built HTML to check against (default: docs/_build/html)",
    )
    args = parser.parse_args(argv)

    if not args.build_dir.is_dir():
        print(
            f"error: no build at {args.build_dir}. Run `make -C docs html` first.",
            file=sys.stderr,
        )
        return 2

    destinations = parse_destinations(TEMPLATE.read_text(encoding="utf-8"))
    cache: dict[str, set[str]] = {}
    problems: list[str] = []

    for source, page, fragment in destinations:
        path = args.build_dir / REFERENCE / page
        if not path.is_file():
            problems.append(f"{source} -> {REFERENCE}{page} (no such page)")
            continue
        if fragment is None:
            continue
        if page not in cache:
            cache[page] = ids_on_page(path)
        if fragment not in cache[page]:
            problems.append(f"{source} -> {REFERENCE}{page}#{fragment} (no such id on that page)")

    if problems:
        print(f"error: {len(problems)} redirect anchor(s) point at nothing:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        print(f"\nFix the table in {_display(TEMPLATE)}, or the page it names.", file=sys.stderr)
        return 1

    print(f"ok: all {len(destinations)} redirect anchor destinations exist in this build.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
