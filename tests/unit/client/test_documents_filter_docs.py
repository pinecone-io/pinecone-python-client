"""The ``filter`` docs on document search and fetch state the field-kind rules.

The server splits filter predicates by the kind of field they name: a data field
(searched by the index) takes no comparison operators, a string whose text is stored and
searchable takes the text-match operators (declared with ``full_text_search``, or with
``embed`` or ``sparse_embed`` and the default ``store_text``), and metadata fields take
the comparison operators. These tests pin that the reference text a caller reads says so on every
surface that documents ``filter``.
"""

from __future__ import annotations

import inspect

import pytest

from pinecone.async_client.documents import AsyncDocuments
from pinecone.client.documents import Documents
from pinecone.models.documents.requests import FetchDocumentsRequest, SearchDocumentsRequest

MATCH_OPERATORS = ("$match_phrase", "$match_all", "$match_any")

SURFACES = [
    Documents.search,
    Documents.fetch,
    AsyncDocuments.search,
    AsyncDocuments.fetch,
    SearchDocumentsRequest,
    FetchDocumentsRequest,
]


def _doc(obj: object) -> str:
    return " ".join((inspect.getdoc(obj) or "").split())


@pytest.mark.parametrize("surface", SURFACES, ids=lambda s: s.__qualname__)
class TestFilterDocsStateFieldKindRules:
    def test_names_every_text_match_operator(self, surface: object) -> None:
        text = _doc(surface)
        for operator in MATCH_OPERATORS:
            assert operator in text

    def test_text_match_rule_covers_every_string_that_keeps_its_text(self, surface: object) -> None:
        text = _doc(surface)
        assert "only a string declared with ``full_text_search``" not in text
        assert "stored and searchable takes the text-match operators" in text
        assert (
            "one declared with ``full_text_search``, or with ``embed`` or ``sparse_embed``" in text
        )
        assert "default ``store_text``" in text
        assert "``store_text=False`` string" in text

    def test_lists_every_data_field_declaration(self, surface: object) -> None:
        assert "``full_text_search``, ``embed`` or ``sparse_embed``" in _doc(surface)

    def test_says_the_wrong_kind_of_operator_is_a_400(self, surface: object) -> None:
        text = _doc(surface)
        assert "400" in text
        assert "metadata field" in text


@pytest.mark.parametrize("method", [Documents.search, Documents.fetch], ids=lambda m: m.__name__)
@pytest.mark.parametrize("twin", ["sync", "async"])
def test_method_examples_show_a_text_match_filter(method: object, twin: str) -> None:
    target = method if twin == "sync" else getattr(AsyncDocuments, method.__name__)  # type: ignore[attr-defined]
    doc = inspect.getdoc(target) or ""
    examples = doc[doc.index("Examples:") :]
    assert "$match_" in examples
    assert "$eq" in examples
