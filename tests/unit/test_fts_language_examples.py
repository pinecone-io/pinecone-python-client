"""``{"language": ...}`` on its own enables no stemming; examples must not imply it does."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from pinecone import SchemaBuilder
from pinecone.models.indexes.schema import FullTextSearchConfig

ROOT = Path(__file__).resolve().parents[2]

SOURCES = [
    "pinecone/client/indexes.py",
    "pinecone/async_client/indexes.py",
    "pinecone/schema_builder.py",
    "docs/guides/concepts.md",
    "docs/migration/v10-migration.md",
]

_FTS_LITERAL = re.compile(r"full_text_search\"?\s*[:=]\s*(\{[^{}]*\})", re.DOTALL)
_NOTE = re.compile(r"exact words? only", re.IGNORECASE)


def _language_only_examples(text: str) -> list[tuple[int, str]]:
    found = []
    for m in _FTS_LITERAL.finditer(text):
        body = m.group(1)
        if "language" in body and not re.search(r"\"stemming\"\s*:\s*True", body):
            window = text[max(0, m.start() - 250) : m.end() + 250]
            if not _NOTE.search(window):
                found.append((text.count("\n", 0, m.start()) + 1, " ".join(body.split())))
    return found


@pytest.mark.parametrize("rel", SOURCES)
def test_language_examples_set_stemming_or_say_exact_match(rel: str) -> None:
    text = (ROOT / rel).read_text()
    assert _FTS_LITERAL.search(text), f"{rel} no longer has a full_text_search example"
    assert _language_only_examples(text) == []


def test_guard_flags_language_only_example() -> None:
    bad = 'schema={"body": {"type": "string", "full_text_search": {"language": "en"}}}'
    assert _language_only_examples(bad) != []


def test_guard_accepts_stemming_or_note() -> None:
    ok = '"full_text_search": {"language": "en", "stemming": True}'
    noted = '# language alone matches exact words only\n"full_text_search": {"language": "en"}'
    assert _language_only_examples(ok) == []
    assert _language_only_examples(noted) == []


def test_full_text_search_config_documents_language_alone() -> None:
    doc = FullTextSearchConfig.__doc__ or ""
    language = doc.split("language:", 1)[1].split("stemming:", 1)[0]
    flat = " ".join(language.split())
    assert "does not turn on" in flat
    assert "stemming" in flat and "stop_words" in flat
    assert "exact words only" in flat


def test_corrected_example_schema_carries_stemming() -> None:
    schema = (
        SchemaBuilder()
        .add_string_field("body", full_text_search={"language": "en", "stemming": True})
        .build()
    )
    fts = schema["fields"]["body"]["full_text_search"]
    assert fts["language"] == "en"
    assert fts["stemming"] is True
