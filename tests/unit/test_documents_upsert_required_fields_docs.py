"""Upsert docs state the schema's required fields, and their examples carry them.

The server requires every ``dense_vector`` field and every ``string`` field declared
with ``embed``; ``sparse_vector`` fields, ``sparse_embed``-only or full-text-only
``string`` fields, and metadata are optional (pinecone-db ``pc-types`` ``DataSchema``
conversion and ``pc-docs`` document parsing).
"""

from __future__ import annotations

import ast
import inspect
import re
import textwrap
from typing import Any

import pytest

from pinecone.async_client.documents import AsyncDocuments
from pinecone.client.documents import Documents
from pinecone.index import Index
from pinecone.models.batch import BatchResult
from pinecone.models.documents.document import DocumentRecord
from pinecone.models.response_info import BatchResponseInfo, ResponseInfo


def _doc(obj: Any) -> str:
    doc = inspect.getdoc(obj)
    assert doc is not None
    return doc


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text)


RULE_DOCSTRINGS = {
    "Documents.upsert": Documents.upsert,
    "AsyncDocuments.upsert": AsyncDocuments.upsert,
    "Documents.batch_upsert": Documents.batch_upsert,
    "AsyncDocuments.batch_upsert": AsyncDocuments.batch_upsert,
    "DocumentRecord": DocumentRecord,
}


@pytest.mark.parametrize("name", sorted(RULE_DOCSTRINGS))
def test_docstring_states_required_field_rule(name: str) -> None:
    text = _flat(_doc(RULE_DOCSTRINGS[name]))

    assert re.search(r"(every|each) ``dense_vector`` field", text, re.IGNORECASE)
    assert re.search(r"``string`` field declared with ``embed``", text)
    optional_clause = re.search(r"``sparse_vector`` fields, (?P<clause>[^.]*?) are optional", text)
    assert optional_clause is not None, "sparse and full-text fields must be stated as optional"
    clause = optional_clause.group("clause")
    assert "``sparse_embed``" in clause
    assert re.search(r"full-text(-only| search)", clause)
    assert clause.endswith("metadata")


@pytest.mark.parametrize("name", sorted(RULE_DOCSTRINGS))
def test_docstring_does_not_call_id_the_only_requirement(name: str) -> None:
    text = _flat(_doc(RULE_DOCSTRINGS[name]))

    assert "only reserved key" not in text
    assert "one required reserved key" not in text
    assert "every document may carry a different set" not in text


UPSERT_EXAMPLE_DOCSTRINGS = {
    "Documents": Documents,
    "AsyncDocuments": AsyncDocuments,
    "Documents.upsert": Documents.upsert,
    "AsyncDocuments.upsert": AsyncDocuments.upsert,
    "Documents.batch_upsert": Documents.batch_upsert,
    "AsyncDocuments.batch_upsert": AsyncDocuments.batch_upsert,
    "Index.documents": Index.documents,
    "ResponseInfo": ResponseInfo,
    "BatchResponseInfo": BatchResponseInfo,
    "BatchResult": BatchResult,
}

_DENSE = re.compile(r"declares ``(\w+)`` as a (?:\d+-dimension )?``dense_vector``")
_EMBED = re.compile(r"declares ``(\w+)`` as a ``string`` field with ``embed``")


def _required_fields(doc: str) -> set[str]:
    text = _flat(doc)
    return set(_DENSE.findall(text)) | set(_EMBED.findall(text))


def _example_sources(doc: str) -> list[str]:
    lines = doc.splitlines()
    sources: list[str] = []

    doctest_lines = [
        line.strip()[4:] for line in lines if line.strip().startswith((">>> ", "... "))
    ]
    if doctest_lines:
        sources.append("\n".join(doctest_lines))

    i = 0
    while i < len(lines):
        if lines[i].strip() == ".. code-block:: python":
            indent = len(lines[i]) - len(lines[i].lstrip())
            body: list[str] = []
            i += 1
            while i < len(lines) and (
                not lines[i].strip() or len(lines[i]) - len(lines[i].lstrip()) > indent
            ):
                body.append(lines[i])
                i += 1
            sources.append(textwrap.dedent("\n".join(body)))
        else:
            i += 1
    return sources


def _upserted_documents(source: str) -> list[ast.Dict]:
    flags = ast.PyCF_ONLY_AST | ast.PyCF_ALLOW_TOP_LEVEL_AWAIT
    tree = compile(source, "<docstring>", "exec", flags)
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        and any(isinstance(k, ast.Constant) and k.value == "_id" for k in node.keys)
    ]


@pytest.mark.parametrize("name", sorted(UPSERT_EXAMPLE_DOCSTRINGS))
def test_upsert_example_names_its_schema(name: str) -> None:
    assert _required_fields(_doc(UPSERT_EXAMPLE_DOCSTRINGS[name])), (
        f"{name}: the example does not say which schema field it assumes"
    )


@pytest.mark.parametrize("name", sorted(UPSERT_EXAMPLE_DOCSTRINGS))
def test_upsert_example_documents_carry_required_fields(name: str) -> None:
    doc = _doc(UPSERT_EXAMPLE_DOCSTRINGS[name])
    required = _required_fields(doc)
    documents = [d for src in _example_sources(doc) for d in _upserted_documents(src)]

    assert documents, f"{name}: found no upserted document in the example"
    for document in documents:
        keys = {k.value for k in document.keys if isinstance(k, ast.Constant)}
        missing = required - keys
        assert not missing, f"{name}: example document lacks required {sorted(missing)}"


def test_extractor_finds_the_documents_the_old_examples_sent() -> None:
    old = """Examples:
    >>> idx.documents.upsert(
    ...     namespace="published",
    ...     documents=[{"_id": "article-101", "title": "Intro to vectors"}],
    ... )

    .. code-block:: python

        documents = [{"_id": f"a-{i}", "title": "t"} for i in range(3)]
        await idx.documents.batch_upsert(namespace="n", documents=documents)
"""
    documents = [d for src in _example_sources(old) for d in _upserted_documents(src)]

    assert len(documents) == 2
    assert all(
        {k.value for k in d.keys if isinstance(k, ast.Constant)} == {"_id", "title"}
        for d in documents
    )
