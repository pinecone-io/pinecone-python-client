"""Tests for scripts/check_docs_urls.py, the published-URL regression gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_docs_urls.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("check_docs_urls", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


urls = _load_script()


def _build(tmp_path: Path, pages: list[str]) -> Path:
    build_dir = tmp_path / "html"
    for page in pages:
        target = build_dir / page
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("<html></html>", encoding="utf-8")
    return build_dir


def _baseline(tmp_path: Path, pages: list[str]) -> Path:
    path = tmp_path / "published-urls.txt"
    path.write_text(urls.BASELINE_HEADER + "\n".join(pages) + "\n", encoding="utf-8")
    return path


def _run(build_dir: Path, baseline: Path, *extra: str) -> int:
    return urls.main(["--build-dir", str(build_dir), "--baseline", str(baseline), *extra])


class TestBuiltUrls:
    def test_collects_nested_pages(self, tmp_path):
        build_dir = _build(tmp_path, ["index.html", "how-to/vectors/upsert.html"])
        assert urls.built_urls(build_dir) == {"index.html", "how-to/vectors/upsert.html"}

    def test_excludes_viewcode_and_generated_pages(self, tmp_path):
        build_dir = _build(
            tmp_path,
            [
                "index.html",
                "_modules/pinecone/index.html",
                "genindex.html",
                "py-modindex.html",
                "search.html",
            ],
        )
        assert urls.built_urls(build_dir) == {"index.html"}

    def test_missing_build_dir_is_an_error(self, tmp_path):
        with pytest.raises(SystemExit, match="no docs build"):
            urls.built_urls(tmp_path / "absent")

    def test_empty_build_is_an_error(self, tmp_path):
        with pytest.raises(SystemExit, match="no documentation pages"):
            urls.built_urls(_build(tmp_path, ["search.html"]))


class TestCheck:
    def test_passes_when_build_matches_inventory(self, tmp_path):
        build_dir = _build(tmp_path, ["index.html", "faq.html"])
        baseline = _baseline(tmp_path, ["index.html", "faq.html"])
        assert _run(build_dir, baseline) == 0

    def test_fails_when_a_published_page_disappears(self, tmp_path, capsys):
        build_dir = _build(tmp_path, ["index.html"])
        baseline = _baseline(tmp_path, ["index.html", "guides/retries.html"])
        assert _run(build_dir, baseline) == 1
        assert "guides/retries.html" in capsys.readouterr().err

    def test_a_redirect_stub_satisfies_a_removed_page(self, tmp_path):
        # sphinx-reredirects writes a real file at the old path, so the same
        # existence check covers "still here" and "moved, with a stub".
        build_dir = _build(tmp_path, ["index.html", "rest.html"])
        baseline = _baseline(tmp_path, ["index.html", "rest.html"])
        assert _run(build_dir, baseline) == 0

    def test_fails_when_a_new_page_is_not_inventoried(self, tmp_path, capsys):
        build_dir = _build(tmp_path, ["index.html", "guides/new.html"])
        baseline = _baseline(tmp_path, ["index.html"])
        assert _run(build_dir, baseline) == 1
        assert "guides/new.html" in capsys.readouterr().err

    def test_missing_inventory_is_an_error(self, tmp_path):
        build_dir = _build(tmp_path, ["index.html"])
        with pytest.raises(SystemExit, match="no URL inventory"):
            _run(build_dir, tmp_path / "absent.txt")


class TestUpdate:
    def test_adds_new_pages(self, tmp_path):
        build_dir = _build(tmp_path, ["index.html", "guides/new.html"])
        baseline = _baseline(tmp_path, ["index.html"])
        assert _run(build_dir, baseline, "--update", "--no-fetch") == 0
        assert urls.baseline_urls(baseline) == {"index.html", "guides/new.html"}
        assert _run(build_dir, baseline) == 0

    def test_never_drops_a_url_the_build_no_longer_produces(self, tmp_path):
        # The whole point: --update must not be a way to launder a deletion.
        build_dir = _build(tmp_path, ["index.html"])
        baseline = _baseline(tmp_path, ["index.html", "guides/retries.html"])
        assert _run(build_dir, baseline, "--update", "--no-fetch") == 0
        assert "guides/retries.html" in urls.baseline_urls(baseline)
        assert _run(build_dir, baseline) == 1

    def test_creates_the_inventory_when_absent(self, tmp_path):
        build_dir = _build(tmp_path, ["index.html"])
        baseline = tmp_path / "published-urls.txt"
        assert _run(build_dir, baseline, "--update", "--no-fetch") == 0
        assert urls.baseline_urls(baseline) == {"index.html"}

    def test_no_fetch_requires_update(self, tmp_path):
        build_dir = _build(tmp_path, ["index.html"])
        baseline = _baseline(tmp_path, ["index.html"])
        with pytest.raises(SystemExit):
            _run(build_dir, baseline, "--no-fetch")


class TestPublishedUrls:
    SITEMAP = """<?xml version='1.0' encoding='utf-8'?>
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    <url><loc>https://sdk.pinecone.io/python/index.html</loc></url>
    <url><loc>https://sdk.pinecone.io/python/guides/retries.html</loc></url>
    <url><loc>https://sdk.pinecone.io/python/search.html</loc></url>
    </urlset>"""

    @staticmethod
    def _patch_fetch(monkeypatch, body: str):
        def _get(url, **kwargs):
            return httpx.Response(200, text=body, request=httpx.Request("GET", url))

        monkeypatch.setattr(httpx, "get", _get)

    def test_parses_and_relativises_locations(self, monkeypatch):
        self._patch_fetch(monkeypatch, self.SITEMAP)
        assert urls.published_urls(urls.DEFAULT_SITEMAP_URL) == {
            "index.html",
            "guides/retries.html",
        }

    def test_update_unions_the_live_sitemap(self, tmp_path, monkeypatch):
        self._patch_fetch(monkeypatch, self.SITEMAP)
        build_dir = _build(tmp_path, ["index.html"])
        baseline = _baseline(tmp_path, ["index.html"])
        assert _run(build_dir, baseline, "--update") == 0
        assert "guides/retries.html" in urls.baseline_urls(baseline)

    def test_rejects_a_location_outside_the_site_root(self, monkeypatch):
        self._patch_fetch(
            monkeypatch,
            "<urlset xmlns='http://www.sitemaps.org/schemas/sitemap/0.9'>"
            "<url><loc>https://example.com/evil.html</loc></url></urlset>",
        )
        with pytest.raises(SystemExit, match="outside"):
            urls.published_urls(urls.DEFAULT_SITEMAP_URL)

    def test_rejects_entity_declarations(self, monkeypatch):
        self._patch_fetch(
            monkeypatch,
            "<!DOCTYPE urlset [<!ENTITY a 'aaaa'>]>"
            "<urlset xmlns='http://www.sitemaps.org/schemas/sitemap/0.9'></urlset>",
        )
        with pytest.raises(SystemExit, match="declares entities"):
            urls.published_urls(urls.DEFAULT_SITEMAP_URL)

    def test_http_failure_is_an_error(self, monkeypatch):
        def _boom(*a, **kw):
            raise httpx.ConnectError("unreachable")

        monkeypatch.setattr(httpx, "get", _boom)
        with pytest.raises(SystemExit, match="could not fetch"):
            urls.published_urls(urls.DEFAULT_SITEMAP_URL)

    def test_error_status_is_an_error(self, monkeypatch):
        def _get(url, **kwargs):
            return httpx.Response(503, text="", request=httpx.Request("GET", url))

        monkeypatch.setattr(httpx, "get", _get)
        with pytest.raises(SystemExit, match="could not fetch"):
            urls.published_urls(urls.DEFAULT_SITEMAP_URL)
