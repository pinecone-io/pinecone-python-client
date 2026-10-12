"""Docstrings must match what the server does on create (recorded in #730)."""

from __future__ import annotations

import inspect
import re

from pinecone.models.indexes.schema import FieldEmbedConfig, IntegerField


def _flat(obj: object) -> str:
    return re.sub(r"\s+", " ", inspect.getdoc(obj) or "")


def _metric_entry() -> str:
    doc = _flat(FieldEmbedConfig)
    start = doc.index("metric: ")
    return doc[start : doc.index("write_parameters:", start)]


def test_embed_metric_doc_says_support_depends_on_the_model() -> None:
    entry = _metric_entry()
    assert "depends on the model" in entry
    assert "llama-text-embed-v2" in entry
    assert "multilingual-e5-large" in entry


def test_embed_metric_doc_names_the_supported_metrics_per_model() -> None:
    entry = _metric_entry()
    split = entry.index("multilingual-e5-large")
    llama = entry[entry.index("llama-text-embed-v2") : split]
    e5 = entry[split:].split(".")[0]
    assert "cosine" in llama and "dotproduct" in llama and "euclidean" not in llama
    assert "cosine" in e5 and "euclidean" in e5 and "dotproduct" not in e5


def test_embed_metric_doc_states_server_default_is_cosine() -> None:
    assert '(``"cosine"`` for those two)' in _metric_entry()


def test_integer_field_doc_says_422_names_the_field_path() -> None:
    doc = _flat(IntegerField)
    assert "422" in doc
    assert "schema.fields." in doc


def test_integer_field_doc_no_longer_claims_the_field_is_unidentified() -> None:
    doc = _flat(IntegerField)
    assert "will not tell you which field" not in doc
