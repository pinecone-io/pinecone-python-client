"""The bulk-ingest guide describes documents ingest with the documents API's own names.

``Documents.batch_upsert`` returns a ``BatchResult`` (``successful_item_count``), not the
vectors API's ``UpsertResponse`` (``upserted_count``), and ``Documents.upsert`` is a single
request. The guide is linked from ``batch_upsert``'s docstring.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import msgspec

from pinecone._internal.bulk.core import GATE_CEILING
from pinecone.async_client.documents import AsyncDocuments
from pinecone.client.documents import Documents
from pinecone.models.batch import BatchResult
from pinecone.models.vectors.responses import UpsertResponse

GUIDE = Path(__file__).resolve().parents[2] / "docs" / "guides" / "bulk-ingest.md"

GATE_COUNTERS = ("throttle_event_count", "peak_inflight", "final_limit", "stalled")


def _guide() -> str:
    return GUIDE.read_text(encoding="utf-8")


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _section(heading: str, text: str | None = None) -> str:
    text = _guide() if text is None else text
    text = re.sub(r"^```.*?^```", "", text, flags=re.DOTALL | re.MULTILINE)
    match = re.search(
        rf"^#+ {re.escape(heading)}\n(?P<body>.*?)(?=^#+ |\Z)", text, re.MULTILINE | re.DOTALL
    )
    assert match is not None, f"guide has no section {heading!r}"
    return match.group("body")


def _table_rows(text: str) -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    for line in text.splitlines():
        if not line.startswith("|") or re.match(r"^\|[-| ]+\|$", line):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        rows[re.sub(r"^`(\w+)`$", r"\1", cells[0])] = cells[1:]
    return rows


def _intro_table(text: str | None = None) -> dict[str, list[str]]:
    text = _guide() if text is None else text
    return _table_rows(text.split("\n## ", 1)[0])


def _gate_table(text: str | None = None) -> dict[str, list[str]]:
    return _table_rows(_section("Reading the gate counters on the result", text))


def _fields(cls: type) -> set[str]:
    return {f.name for f in msgspec.structs.fields(cls)}


def _result_attributes() -> set[str]:
    return {
        name
        for cls in (BatchResult, UpsertResponse)
        for name in dir(cls)
        if not name.startswith("_")
    }


def test_intro_table_names_the_batched_call_for_each_data_model() -> None:
    vectors, documents = _intro_table()["Batched call"]

    assert "`upsert(vectors=" in vectors
    assert "`documents.batch_upsert()`" in documents
    assert "`documents.upsert()` is a single request" in _flat(_guide())


def test_intro_table_gives_each_data_model_its_own_count_field() -> None:
    returns = _intro_table()["Returns"]
    vectors, documents = _intro_table()["Accepted-item count"]

    assert returns == ["`UpsertResponse`", "`BatchResult`"]
    assert re.findall(r"`(\w+)`", vectors) == ["upserted_count"]
    assert "upserted_count" in _fields(UpsertResponse)
    assert re.findall(r"`(\w+)`", documents) == ["successful_item_count", "success_count"]
    assert "successful_item_count" in _fields(BatchResult)


def test_documents_batch_size_default_matches_the_signature() -> None:
    documents = _intro_table()["Default `batch_size`"][1]

    for cls in (Documents, AsyncDocuments):
        default = inspect.signature(cls.batch_upsert).parameters["batch_size"].default
        assert documents.startswith(f"{default};")


def test_batch_size_ceiling_is_attributed_to_the_server() -> None:
    documents = _intro_table()["Default `batch_size`"][1]

    assert "server accepts at most 1000" in documents
    assert "The SDK only requires `batch_size >= 1`" not in _section("`batch_size`")


def test_partial_failure_contract_names_both_count_fields() -> None:
    contract = _flat(_section("Partial failures: the reporting contract"))

    assert "`upserted_count` (vectors)" in contract
    assert "`successful_item_count` (documents)" in contract


def test_gate_table_explains_exactly_the_gate_counters() -> None:
    table = _gate_table()

    assert set(table) - {"Field"} == set(GATE_COUNTERS)
    for counter in GATE_COUNTERS:
        assert counter in _fields(BatchResult)
        assert all(table[counter]), f"{counter} has an empty cell"


def test_final_limit_is_described_as_the_host_gate_limit() -> None:
    meaning, reading = _gate_table()["final_limit"]

    assert "host gate" in meaning
    assert f"from 1 to {GATE_CEILING}" in meaning
    assert f"Compare with {GATE_CEILING}, not `max_concurrency`" in reading


def _guide_named_attributes(text: str) -> set[str]:
    named = set(re.findall(r"\b(?:result|response)\.(\w+)", text))
    contract = _section("Partial failures: the reporting contract", text)
    for line in re.findall(r"^- (.+?) —", contract, re.MULTILINE):
        named |= set(re.findall(r"`(\w+)`", line))
    named |= set(_gate_table(text)) - {"Field"}
    for cell in _intro_table(text)["Accepted-item count"]:
        named |= set(re.findall(r"`(\w+)`", cell))
    return named


def test_every_result_attribute_the_guide_names_exists() -> None:
    unknown = _guide_named_attributes(_guide()) - _result_attributes()

    assert not unknown, f"the guide names result attributes that do not exist: {sorted(unknown)}"


def test_attribute_check_flags_a_name_neither_result_type_has() -> None:
    edited = _guide().replace("`failed_item_count` and", "`made_up_count` and", 1)
    edited = edited.replace("| `peak_inflight` |", "| `made_up_peak` |", 1)
    edited += "\nprint(result.also_made_up)\n"

    assert _guide_named_attributes(edited) - _result_attributes() == {
        "made_up_count",
        "made_up_peak",
        "also_made_up",
    }
