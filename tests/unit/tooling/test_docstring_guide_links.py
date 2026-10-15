"""Guide references in docstrings must be followable outside a Sphinx build.

``help()``, IDE hovers and ``pydoc`` print a Sphinx ``:doc:`` role as literal
text, naming no host. A reference to a published guide therefore has to be an
absolute URL, and that URL has to be one the docs site has promised to keep.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE = REPO_ROOT / "pinecone"
PUBLISHED_URLS = REPO_ROOT / "docs" / "published-urls.txt"
SITE = "https://sdk.pinecone.io/python/"

DOC_ROLE = re.compile(r":doc:`")
SITE_URL = re.compile(re.escape(SITE) + r"([^\s<>`'\"),;]+)")


def _sources() -> list[Path]:
    return sorted(PACKAGE.rglob("*.py"))


def _published() -> set[str]:
    lines = PUBLISHED_URLS.read_text().splitlines()
    return {line.strip() for line in lines if line.strip() and not line.startswith("#")}


def test_no_doc_role_in_package_source() -> None:
    offenders = [
        f"{path.relative_to(REPO_ROOT)}:{n}"
        for path in _sources()
        for n, line in enumerate(path.read_text().splitlines(), 1)
        if DOC_ROLE.search(line)
    ]
    assert not offenders, (
        f"{len(offenders)} :doc: role(s) print as literal text in help(): "
        + ", ".join(offenders[:10])
    )


def _guide_targets(text: str) -> list[str]:
    return [m.group(1).rstrip(".").split("#", 1)[0] for m in SITE_URL.finditer(text)]


def test_every_guide_url_in_package_is_published() -> None:
    published = _published()
    unknown: list[str] = []
    found = 0
    for path in _sources():
        for target in _guide_targets(path.read_text()):
            found += 1
            if target and target not in published:
                unknown.append(f"{path.relative_to(REPO_ROOT)}: {target}")
    assert found > 100, "the scan found almost no guide URLs; the pattern has stopped matching"
    assert not unknown, "guide URLs missing from docs/published-urls.txt: " + "; ".join(unknown)


def test_target_scan_handles_each_url_shape() -> None:
    base = SITE + "guides/retries.html"
    for text in (
        f"`Retries <{base}>`__",
        f'URL = "{base}"',
        f"see {base}. More text",
        f"see {base}, then",
        f"({base})",
    ):
        assert _guide_targets(text) == ["guides/retries.html"], text


def test_target_scan_flags_unpublished_page() -> None:
    assert _guide_targets(f"`Gone <{SITE}guides/not-a-page.html>`__") == ["guides/not-a-page.html"]
    assert "guides/not-a-page.html" not in _published()
