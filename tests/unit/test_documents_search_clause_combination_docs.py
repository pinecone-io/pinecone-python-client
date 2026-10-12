"""Search docs state how fields and clauses combine, and the examples run.

The rules come from pinecone-db: ``pc-tantivy-utils`` ``MergedTantivyQueryBuilder``
joins every ``text`` and ``query_string`` clause into one Tantivy query with ``" OR "``
(one ``field:(terms)`` clause per field a ``text`` clause names), and the query router
adds one such clause per entry in ``fields``. Scores summing across matched clauses,
identical clauses collapsing to one, and ``^N`` multiplying were observed against
production on 2026-10-09 (``pinecone==10.1.0rc1``, API 2026-07), not read in source.
"""

from __future__ import annotations

import inspect
import re
from typing import Any

import pytest

from pinecone.async_client.documents import AsyncDocuments
from pinecone.client.documents import Documents
from pinecone.models.documents import score_by
from pinecone.models.documents.score_by import QueryStringQuery, TextQuery


def _flat(obj: Any) -> str:
    doc = inspect.getdoc(obj)
    assert doc is not None
    return re.sub(r"\s+", " ", doc)


SEARCH = {"Documents.search": Documents.search, "AsyncDocuments.search": AsyncDocuments.search}


def test_text_query_states_any_field_match_and_summed_scores() -> None:
    text = _flat(TextQuery)

    assert "matches if the query matches in any one of the named fields" in text
    assert "sum of its BM25 scores in the fields that matched" in text
    assert "a ``^N`` inside ``query`` is not a boost" in text


@pytest.mark.parametrize("name", sorted(SEARCH))
def test_search_score_by_states_or_sum_collapse_and_boost(name: str) -> None:
    text = _flat(SEARCH[name])

    assert "Combined clauses are OR'd, not AND'd" in text
    assert "scores the sum of the scores from each clause it matches" in text
    assert "naming several ``fields`` works the same way across those fields" in text
    assert "naming a field twice" in text
    assert "both a ``text`` and a ``query_string`` clause" in text
    assert "repeating a clause does not up-weight it" in text
    assert "a ``text`` clause has no boost" in text
    assert "``^N`` boost" in text
    assert "``title:entropy^5``" in text
    assert "use ``AND`` inside one ``QueryStringQuery``" in text


def test_search_twins_state_the_same_rule() -> None:
    def rule(obj: Any) -> str:
        text = _flat(obj)
        start = text.index("Combined clauses are OR'd")
        return text[start : text.index("``QueryStringQuery``.", start)]

    assert rule(Documents.search) == rule(AsyncDocuments.search)


def test_query_string_documents_the_boost_form() -> None:
    text = _flat(QueryStringQuery)

    assert "``^N`` boost" in text
    assert "``title:entropy^5``" in text
    assert "multiplied by ``N``" in text


def test_module_docstring_states_the_combination_rule() -> None:
    text = _flat(score_by)

    assert "The clauses are OR'd" in text
    assert "scores the sum of the scores from each clause it matches" in text
    assert "naming a field twice" in text
    assert "repeating a clause does not up-weight it" in text
    assert "a ``^N`` boost in a ``query_string`` clause does up-weight a term" in text


def _doctest_source(obj: Any) -> str:
    lines = [line.strip() for line in (inspect.getdoc(obj) or "").splitlines()]
    return "\n".join(line[4:] for line in lines if line.startswith((">>> ", "... ")))


def test_query_string_examples_execute_and_show_the_boost() -> None:
    namespace: dict[str, Any] = {}
    source = _doctest_source(QueryStringQuery)

    assert "title:entropy^5 abstract:entropy" in source
    exec(source, namespace)  # noqa: S102

    assert namespace["QueryStringQuery"] is QueryStringQuery


def test_text_query_example_executes() -> None:
    namespace: dict[str, Any] = {}

    source = _doctest_source(TextQuery)
    assert source
    exec(source, namespace)  # noqa: S102
