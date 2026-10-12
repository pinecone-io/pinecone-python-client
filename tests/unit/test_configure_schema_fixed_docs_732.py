"""``configure()`` documents that a ``create(schema=...)`` index's schema is fixed (#732).

The server answers a ``schema=`` change on such an index with a raw 422/400, so
the docstring is where a caller learns why. Each test pins one claim, on the
sync and async clients and on ``ConfigureIndexRequest``.
"""

from __future__ import annotations

import inspect
import re

import pytest

from pinecone.async_client.indexes import AsyncIndexes
from pinecone.client.indexes import Indexes
from pinecone.models.indexes.requests import ConfigureIndexRequest


def _flat(text: str | None) -> str:
    assert text is not None
    return re.sub(r"\s+", " ", text)


def _section(doc: str, start: str, end: str) -> str:
    flat = _flat(doc)
    head = flat.index(start)
    return flat[head : flat.index(end, head)]


CLIENTS = [pytest.param(Indexes, id="sync"), pytest.param(AsyncIndexes, id="async")]


@pytest.mark.parametrize("cls", CLIENTS)
def test_schema_arg_says_create_schema_index_is_fixed(cls: type) -> None:
    arg = _section(inspect.getdoc(cls.configure) or "", "schema: ", "read_capacity: ")
    assert "create(schema=...)" in arg
    assert "nothing in" in arg and "can be changed" in arg
    assert "embed" in arg and "sparse_embed" in arg


@pytest.mark.parametrize("cls", CLIENTS)
def test_schema_note_names_create_for_model_as_only_updatable_case(cls: type) -> None:
    note = _section(inspect.getdoc(cls.configure) or "", ".. note::", ".. versionchanged::")
    assert "create_for_model" in note
    assert "create(schema=...)" in note
    assert "fixed" in note


@pytest.mark.parametrize("cls", CLIENTS)
def test_raises_lists_api_error_for_schema_and_deployment(cls: type) -> None:
    raises = _section(inspect.getdoc(cls.configure) or "", "Raises:", "Examples:")
    assert "ApiError" in raises
    assert "422" in raises and "400" in raises
    assert "schema" in raises
    assert "replicas" in raises and "pod_type" in raises


@pytest.mark.parametrize("cls", CLIENTS)
def test_deployment_arg_says_only_replicas_and_pod_type(cls: type) -> None:
    arg = _section(inspect.getdoc(cls.configure) or "", "deployment: ", "schema: ")
    assert "Only those two keys are accepted" in arg
    assert "rejected by the server" in arg


def test_request_model_schema_attribute_matches() -> None:
    attr = _section(ConfigureIndexRequest.__doc__ or "", "schema: ", "deployment: ")
    assert "create(schema=...)" in attr
    assert "fixed" in attr
    deployment = _section(ConfigureIndexRequest.__doc__ or "", "deployment: ", "read_capacity: ")
    assert "Only ``replicas`` and ``pod_type`` are accepted" in deployment
    assert "rejected by the server" in deployment


def test_sync_and_async_docs_agree_on_new_text() -> None:
    sync = _flat(inspect.getdoc(Indexes.configure))
    asyn = _flat(inspect.getdoc(AsyncIndexes.configure))
    for start, end in [("schema: ", "read_capacity: "), ("deployment: ", "schema: ")]:
        assert _section(sync, start, end) == _section(asyn, start, end)
