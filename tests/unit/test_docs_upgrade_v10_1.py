"""Executes the code blocks in ``docs/migration/v10-1-upgrade.md``.

The blocks are read out of the page and run in order in one namespace, the way
the page tells a reader to run them, against a stubbed control and data plane.
A top-level statement followed by a ``# SomeError: message`` comment must raise
that exception (or emit that warning) with a message starting with the comment's
text; every other statement must run cleanly. The block labelled as the 10.0.0
recipe is not run in sequence: it is the loop the page says never exits, and a
separate test shows that it doesn't.
"""

from __future__ import annotations

import ast
import builtins
import json
import re
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

import pinecone
from tests.factories import make_index_response

GUIDE = Path(__file__).resolve().parents[2] / "docs/migration/v10-1-upgrade.md"
OLD_RECIPE = "# 10.0.0 recipe"
EXPECTED = re.compile(r"# (\w+(?:Error|Warning)): (.*)")

CONTROL = "https://api.pinecone.io"
INDEX = make_index_response(name="movies")
DATA = f"https://{INDEX['host']}"
COMMITTED = "28"


def _blocks() -> list[str]:
    blocks = re.findall(r"```python\n(.*?)```", GUIDE.read_text(), re.DOTALL)
    assert blocks, f"no python blocks found in {GUIDE}"
    return blocks


BLOCKS = _blocks()


def _data_plane(request: httpx.Request) -> httpx.Response:
    op = request.url.path.rsplit("/", 1)[-1]
    if op == "upsert":
        count = len(json.loads(request.content)["documents"])
        return httpx.Response(
            202, json={"upserted_count": count}, headers={"x-pinecone-request-lsn": COMMITTED}
        )
    if op == "search":
        return httpx.Response(
            200,
            json={"matches": [], "namespace": "movies-en", "usage": {"read_units": 1}},
            headers={"x-pinecone-max-indexed-lsn": COMMITTED},
        )
    if op == "fetch":
        return httpx.Response(
            200, json={"documents": {}, "namespace": "movies-en", "usage": {"read_units": 1}}
        )
    raise AssertionError(f"the page reached an unstubbed data-plane route: {request.url}")


@pytest.fixture
def stubs() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        router.get(f"{CONTROL}/indexes/movies").mock(return_value=httpx.Response(200, json=INDEX))
        router.route(url__startswith=DATA).mock(side_effect=_data_plane)
        yield router


def _expectation(lines: list[str], stmt: ast.stmt) -> tuple[type[BaseException], str] | None:
    end = stmt.end_lineno or stmt.lineno
    following = lines[end] if end < len(lines) else ""
    match = EXPECTED.fullmatch(following.strip())
    if match is None:
        return None
    name, message = match.groups()
    cls = getattr(pinecone, name, None) or getattr(builtins, name)
    return cls, message


def _run_block(source: str, namespace: dict[str, Any]) -> int:
    lines = source.splitlines()
    checked = 0
    for stmt in ast.parse(source).body:
        code = compile(ast.Module(body=[stmt], type_ignores=[]), str(GUIDE), "exec")
        expected = _expectation(lines, stmt)
        if expected is None:
            exec(code, namespace)  # noqa: S102
            continue
        cls, message = expected
        if issubclass(cls, Warning):
            with pytest.warns(cls) as record:
                exec(code, namespace)  # noqa: S102
            assert any(str(w.message).startswith(message) for w in record), (
                f"no {cls.__name__} starting {message!r}"
            )
        else:
            with pytest.raises(cls) as excinfo:
                exec(code, namespace)  # noqa: S102
            assert str(excinfo.value).startswith(message), str(excinfo.value)
        checked += 1
    return checked


def _run_page_through(marker: str) -> dict[str, Any]:
    namespace: dict[str, Any] = {}
    for source in BLOCKS:
        if not source.startswith(OLD_RECIPE):
            _run_block(source, namespace)
        if marker in source:
            return namespace
    raise AssertionError(f"no block contains {marker!r}")


def test_every_block_runs_in_order_as_the_page_says(stubs: respx.MockRouter) -> None:
    namespace: dict[str, Any] = {}
    checked = 0
    for source in BLOCKS:
        if source.startswith(OLD_RECIPE):
            continue
        checked += _run_block(source, namespace)
    assert checked == sum(
        1 for source in BLOCKS for line in source.splitlines() if EXPECTED.fullmatch(line.strip())
    ), "an expected-error comment was not attached to the statement before it"


def test_the_bounded_recipe_polls_search_until_it_reaches_the_write(
    stubs: respx.MockRouter,
) -> None:
    namespace = _run_page_through("deadline = time.monotonic()")
    assert namespace["target"] == int(COMMITTED)
    assert namespace["found"].response_info.is_reconciled(namespace["target"])


def test_the_10_0_recipe_never_exits_against_a_10_1_fetch(
    stubs: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    class StillPollingError(Exception):
        pass

    polls = iter(range(20))

    def sleep(_seconds: float) -> None:
        if next(polls) == 19:
            raise StillPollingError

    monkeypatch.setattr(time, "sleep", sleep)
    namespace = _run_page_through("pc = Pinecone(")
    old = next(s for s in BLOCKS if s.startswith(OLD_RECIPE))
    with pytest.raises(StillPollingError):
        _run_block(old, namespace)
    assert namespace["target"] == int(COMMITTED)


def test_additional_headers_in_any_case_send_one_api_version(stubs: respx.MockRouter) -> None:
    namespace = _run_page_through("pinned = ")
    namespace["pinned"].indexes.describe("movies")
    sent = stubs.calls.last.request.headers.get_list("x-pinecone-api-version")
    assert sent == ["2025-10"]


def test_documents_bare_str_was_never_sent(stubs: respx.MockRouter) -> None:
    _run_page_through('ids="movie-42"')
    assert not [c for c in stubs.calls if c.request.url.path.endswith("/delete")]
