#!/usr/bin/env python3
"""Guard published documentation URLs against silent removal.

Every URL listed in ``docs/published-urls.txt`` has to still resolve in a
fresh build. Deleting a page is fine; deleting it without a redirect is not.
sphinx-reredirects writes a real file at the old path, which is why one
existence check covers both a page that is still there and one that moved.

``--update`` only ever adds. Run it after deleting a page and the URL stays
in the inventory and the check still fails; retiring a URL for good means
removing its line by hand.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urljoin

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUILD_DIR = REPO_ROOT / "docs" / "_build" / "html"
DEFAULT_BASELINE = REPO_ROOT / "docs" / "published-urls.txt"
DEFAULT_SITEMAP_URL = "https://sdk.pinecone.io/python/sitemap.xml"

SITEMAP_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

# _modules/ is viewcode's listings, named after Python module paths; promising
# those URLs would mean never renaming an internal module.
EXCLUDED_PREFIXES = ("_modules/",)
EXCLUDED_NAMES = frozenset({"genindex.html", "py-modindex.html", "search.html"})

BASELINE_HEADER = """\
# Documentation URLs that sdk.pinecone.io/python has published.
#
# Checked by scripts/check_docs_urls.py against the pages `make -C docs html`
# produces, so a page cannot be removed without a redirect taking its place.
# Paths are relative to https://sdk.pinecone.io/python/.
#
# To add newly published pages:  uv run python scripts/check_docs_urls.py --update
# Removing a line retires a URL for good — only do that for one that was never
# publicly reachable.
"""


def _is_tracked(relative_path: str) -> bool:
    if relative_path.startswith(EXCLUDED_PREFIXES):
        return False
    return relative_path.rsplit("/", 1)[-1] not in EXCLUDED_NAMES


def built_urls(build_dir: Path) -> set[str]:
    """URL paths the Sphinx build emits, including sphinx-reredirects stubs."""
    if not build_dir.is_dir():
        raise SystemExit(
            f"error: no docs build at {build_dir}\n"
            "       run `make -C docs html` first, or pass --build-dir."
        )
    urls = {path.relative_to(build_dir).as_posix() for path in build_dir.rglob("*.html")}
    tracked = {url for url in urls if _is_tracked(url)}
    if not tracked:
        raise SystemExit(f"error: {build_dir} contains no documentation pages.")
    return tracked


def baseline_urls(baseline: Path) -> set[str]:
    if not baseline.is_file():
        raise SystemExit(
            f"error: no URL inventory at {baseline}\n"
            "       create it with `--update` after building the docs."
        )
    lines = baseline.read_text(encoding="utf-8").splitlines()
    return {line.strip() for line in lines if line.strip() and not line.startswith("#")}


def published_urls(sitemap_url: str) -> set[str]:
    """URL paths the live sitemap advertises.

    Redirect stubs are not Sphinx documents and never appear here, so --update
    unions this with the build rather than trusting it alone.
    """
    import httpx  # deferred so the check CI runs needs only the standard library

    site_root = urljoin(sitemap_url, ".")
    try:
        response = httpx.get(sitemap_url, timeout=30, follow_redirects=True)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise SystemExit(
            f"error: could not fetch {sitemap_url}: {exc}\n"
            "       pass --no-fetch to update from the local build alone."
        ) from exc

    document = response.text
    # Refusing these is what keeps expat away from an entity-expansion bomb.
    if "<!DOCTYPE" in document or "<!ENTITY" in document:
        raise SystemExit(f"error: {sitemap_url} declares entities; refusing to parse it.")

    root = ET.fromstring(document)  # noqa: S314 - entity declarations rejected above
    urls = set()
    for location in root.findall(".//sm:url/sm:loc", SITEMAP_NS):
        url = (location.text or "").strip()
        if not url.startswith(site_root):
            raise SystemExit(f"error: sitemap entry outside {site_root}: {url}")
        urls.add(url[len(site_root) :])
    if not urls:
        raise SystemExit(f"error: {sitemap_url} advertised no URLs.")
    return {url for url in urls if _is_tracked(url)}


def check(build_dir: Path, baseline: Path) -> int:
    built = built_urls(build_dir)
    expected = baseline_urls(baseline)

    removed = sorted(expected - built)
    added = sorted(built - expected)

    if removed:
        print(
            f"error: {len(removed)} published URL(s) are no longer produced by this build:",
            file=sys.stderr,
        )
        for url in removed:
            print(f"  {url}", file=sys.stderr)
        print(
            "\nEvery published URL has to keep resolving. Either restore the page, or\n"
            "add an entry to `redirects` in docs/conf.py pointing at its replacement —\n"
            "sphinx-reredirects writes a forwarding stub at the old path, which\n"
            "satisfies this check.",
            file=sys.stderr,
        )
    if added:
        print(
            f"error: {len(added)} page(s) are built but absent from {baseline.name}:",
            file=sys.stderr,
        )
        for url in added:
            print(f"  {url}", file=sys.stderr)
        print(
            "\nRun `uv run python scripts/check_docs_urls.py --update` and commit\n"
            f"docs/{baseline.name} so these URLs are protected from now on.",
            file=sys.stderr,
        )
    if removed or added:
        return 1

    print(f"ok: all {len(expected)} published URLs still resolve in this build.")
    return 0


def update(build_dir: Path, baseline: Path, sitemap_url: str, fetch: bool) -> int:
    known = baseline_urls(baseline) if baseline.is_file() else set()
    discovered = built_urls(build_dir)
    if fetch:
        discovered |= published_urls(sitemap_url)

    merged = known | discovered
    added = sorted(merged - known)

    baseline.write_text(BASELINE_HEADER + "\n".join(sorted(merged)) + "\n", encoding="utf-8")

    if added:
        print(f"added {len(added)} URL(s) to {baseline}:")
        for url in added:
            print(f"  {url}")
    else:
        print(f"no new URLs; {baseline} already lists all {len(merged)}.")

    stale = sorted(known - discovered)
    if stale:
        print(
            f"\nnote: {len(stale)} inventoried URL(s) are not in this build and were kept.\n"
            "      The check will fail until each is restored or redirected:",
        )
        for url in stale:
            print(f"  {url}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--update",
        action="store_true",
        help="add newly built and newly published URLs to the inventory",
    )
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="with --update, skip the live sitemap and use the local build alone",
    )
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--sitemap-url", default=DEFAULT_SITEMAP_URL)
    args = parser.parse_args(argv)

    if args.no_fetch and not args.update:
        parser.error("--no-fetch only applies to --update")

    if args.update:
        return update(args.build_dir, args.baseline, args.sitemap_url, fetch=not args.no_fetch)
    return check(args.build_dir, args.baseline)


if __name__ == "__main__":
    sys.exit(main())
