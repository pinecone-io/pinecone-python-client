from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "check_redirect_anchors.py"

_spec = importlib.util.spec_from_file_location("check_redirect_anchors", SCRIPT)
assert _spec is not None and _spec.loader is not None
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


TEMPLATE = """
<script>
(function () {
  var classes = [
    ["old.Klass", "sync-index.html", "new.Klass"]
  ];
  var verbs = {
    "create_index": ["indexes", "create"]
  };
  var resources = {
    "indexes": "Indexes"
  };
  var clients = [
    ["pinecone.Pinecone", "pinecone.html", "pinecone.client", ""],
    ["pinecone.PineconeAsyncio", "async-pinecone.html", "pinecone.async_client", "Async"]
  ];
  var sections = {
    "rest.html#db-data-plane": "sync-index.html",
    "admin.html#admin-api": "admin.html#admin"
  };
})();
</script>
"""

FULL_PAGES = {
    "sync-index.html": {"new.Klass"},
    "admin.html": {"admin"},
    "pinecone.html": {"pinecone.client.indexes.Indexes.create"},
    "async-pinecone.html": {"pinecone.async_client.indexes.AsyncIndexes.create"},
}


@pytest.fixture
def template(tmp_path, monkeypatch):
    path = tmp_path / "redirect.html"
    path.write_text(TEMPLATE, encoding="utf-8")
    monkeypatch.setattr(mod, "TEMPLATE", path)
    return path


def build_with(tmp_path: Path, pages: dict[str, set[str]]) -> Path:
    reference = tmp_path / "build" / "reference"
    reference.mkdir(parents=True)
    for name, ids in pages.items():
        body = "".join(f'<h1 id="{i}">x</h1>' for i in ids)
        (reference / name).write_text(f"<html>{body}</html>", encoding="utf-8")
    return reference.parent


def test_parses_every_table():
    found = mod.parse_destinations(TEMPLATE)
    assert ("classes[old.Klass]", "sync-index.html", "new.Klass") in found
    assert ("sections[sync-index.html]", "sync-index.html", None) in found
    assert ("sections[admin.html#admin]", "admin.html", "admin") in found


def test_expands_method_ids_for_every_client():
    found = mod.parse_destinations(TEMPLATE)
    assert (
        "verbs[pinecone.Pinecone.create_index]",
        "pinecone.html",
        "pinecone.client.indexes.Indexes.create",
    ) in found
    assert (
        "verbs[pinecone.PineconeAsyncio.create_index]",
        "async-pinecone.html",
        "pinecone.async_client.indexes.AsyncIndexes.create",
    ) in found


def test_commented_out_rows_are_not_parsed_as_entries():
    ghost = "    " + chr(47) * 2 + ' ["ghost.Klass", "gone.html", "x"]\n'
    commented = TEMPLATE.replace('    ["old.Klass"', ghost + '    ["old.Klass"')
    assert all("ghost" not in source for source, _, _ in mod.parse_destinations(commented))


def test_passes_when_every_destination_exists(tmp_path, template, capsys):
    assert mod.main([str(build_with(tmp_path, FULL_PAGES))]) == 0
    assert "ok:" in capsys.readouterr().out


def test_fails_on_a_missing_page(tmp_path, template, capsys):
    pages = {k: v for k, v in FULL_PAGES.items() if k != "async-pinecone.html"}
    assert mod.main([str(build_with(tmp_path, pages))]) == 1
    assert "no such page" in capsys.readouterr().err


def test_fails_on_a_fragment_the_page_does_not_define(tmp_path, template, capsys):
    pages = dict(FULL_PAGES, **{"admin.html": {"something-else"}})
    assert mod.main([str(build_with(tmp_path, pages))]) == 1
    assert "no such id on that page" in capsys.readouterr().err


def test_missing_build_directory_is_a_usage_error(tmp_path, capsys):
    assert mod.main([str(tmp_path / "absent")]) == 2
    assert "no build at" in capsys.readouterr().err


def test_verbs_naming_an_unknown_resource_is_rejected():
    broken = TEMPLATE.replace('"create_index": ["indexes"', '"create_index": ["nope"')
    with pytest.raises(ValueError, match="absent from resources"):
        mod.parse_destinations(broken)


def test_unclosed_table_is_rejected():
    with pytest.raises(ValueError, match="not closed"):
        mod.parse_destinations('<script>var classes = [\n  ["a", "b", "c"]\n</script>')


def test_the_real_template_parses_and_names_only_html_pages():
    found = mod.parse_destinations(mod.TEMPLATE.read_text(encoding="utf-8"))
    assert len(found) > 50
    assert all(page.endswith(".html") for _, page, _ in found)
