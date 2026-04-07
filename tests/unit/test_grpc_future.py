"""Tests for PineconeFuture wrapper."""

from __future__ import annotations

import time
from concurrent.futures import Future, as_completed, wait, FIRST_COMPLETED, ALL_COMPLETED, FIRST_EXCEPTION

import pytest

from pinecone.errors.exceptions import PineconeTimeoutError
from pinecone.grpc.future import PineconeFuture


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _completed_future(value: object) -> Future[object]:
    """Return a stdlib Future that is already resolved with *value*."""
    f: Future[object] = Future()
    f.set_result(value)
    return f


def _failed_future(exc: BaseException) -> Future[object]:
    """Return a stdlib Future that is already failed with *exc*."""
    f: Future[object] = Future()
    f.set_exception(exc)
    return f


# ---------------------------------------------------------------------------
# (a) Completed future returns result
# ---------------------------------------------------------------------------


class TestCompletedFutureReturnsResult:
    def test_result_returns_value(self) -> None:
        underlying = _completed_future(42)
        pf: PineconeFuture[object] = PineconeFuture(underlying)
        assert pf.result() == 42

    def test_done_is_true(self) -> None:
        underlying = _completed_future("hello")
        pf: PineconeFuture[object] = PineconeFuture(underlying)
        assert pf.done() is True

    def test_cancelled_is_false(self) -> None:
        underlying = _completed_future("hello")
        pf: PineconeFuture[object] = PineconeFuture(underlying)
        assert pf.cancelled() is False

    def test_exception_is_none(self) -> None:
        underlying = _completed_future("ok")
        pf: PineconeFuture[object] = PineconeFuture(underlying)
        assert pf.exception() is None


# ---------------------------------------------------------------------------
# (b) Failed future re-raises the original exception
# ---------------------------------------------------------------------------


class TestFailedFutureReRaises:
    def test_result_raises_original(self) -> None:
        err = ValueError("boom")
        underlying = _failed_future(err)
        pf: PineconeFuture[object] = PineconeFuture(underlying)
        with pytest.raises(ValueError, match="boom"):
            pf.result()

    def test_exception_returns_original(self) -> None:
        err = RuntimeError("oops")
        underlying = _failed_future(err)
        pf: PineconeFuture[object] = PineconeFuture(underlying)
        assert pf.exception() is err


# ---------------------------------------------------------------------------
# (c) Cancel propagation
# ---------------------------------------------------------------------------


class TestCancelPropagation:
    def test_cancel_pending_future(self) -> None:
        underlying: Future[object] = Future()
        pf: PineconeFuture[object] = PineconeFuture(underlying)
        # A bare (not-yet-running) Future can be cancelled.
        assert pf.cancel() is True
        assert pf.cancelled() is True
        assert underlying.cancelled() is True

    def test_cancel_completed_future_returns_false(self) -> None:
        underlying = _completed_future("done")
        pf: PineconeFuture[object] = PineconeFuture(underlying)
        assert pf.cancel() is False


# ---------------------------------------------------------------------------
# (d) Done-callback invoked
# ---------------------------------------------------------------------------


class TestDoneCallback:
    def test_callback_called_on_completion(self) -> None:
        underlying: Future[object] = Future()
        pf: PineconeFuture[object] = PineconeFuture(underlying)

        results: list[object] = []
        pf.add_done_callback(lambda f: results.append(f))

        # Resolve the underlying future — callback should fire.
        underlying.set_result("value")

        # Give a tiny window for the callback to execute.
        time.sleep(0.05)
        assert len(results) == 1
        # The callback receives the PineconeFuture, not the underlying.
        assert results[0] is pf

    def test_callback_called_immediately_if_already_done(self) -> None:
        underlying = _completed_future(99)
        pf: PineconeFuture[object] = PineconeFuture(underlying)

        results: list[object] = []
        pf.add_done_callback(lambda f: results.append(f.result()))

        time.sleep(0.05)
        assert results == [99]


# ---------------------------------------------------------------------------
# (e) Timeout raises PineconeTimeoutError
# ---------------------------------------------------------------------------


class TestTimeoutRaisesPineconeTimeoutError:
    def test_result_default_timeout(self) -> None:
        underlying: Future[object] = Future()
        pf: PineconeFuture[object] = PineconeFuture(underlying)
        with pytest.raises(PineconeTimeoutError, match="deadline exceeded"):
            pf.result(timeout=0.01)

    def test_exception_timeout(self) -> None:
        underlying: Future[object] = Future()
        pf: PineconeFuture[object] = PineconeFuture(underlying)
        with pytest.raises(PineconeTimeoutError, match="deadline exceeded"):
            pf.exception(timeout=0.01)

    def test_default_timeout_is_five_seconds(self) -> None:
        """Verify the default timeout is 5s (not infinite)."""
        from pinecone.grpc.future import _DEFAULT_TIMEOUT

        assert _DEFAULT_TIMEOUT == 5.0


# ---------------------------------------------------------------------------
# (f) Compatible with as_completed / wait
# ---------------------------------------------------------------------------


class TestConcurrentFuturesCompatibility:
    def test_as_completed(self) -> None:
        f1: Future[int] = Future()
        f2: Future[int] = Future()
        pf1: PineconeFuture[int] = PineconeFuture(f1)
        pf2: PineconeFuture[int] = PineconeFuture(f2)

        f1.set_result(1)
        f2.set_result(2)

        completed = set(as_completed([pf1, pf2], timeout=2))
        assert pf1 in completed
        assert pf2 in completed

    def test_wait_all_completed(self) -> None:
        f1: Future[str] = Future()
        f2: Future[str] = Future()
        pf1: PineconeFuture[str] = PineconeFuture(f1)
        pf2: PineconeFuture[str] = PineconeFuture(f2)

        f1.set_result("a")
        f2.set_result("b")

        done, not_done = wait([pf1, pf2], timeout=2, return_when=ALL_COMPLETED)
        assert len(done) == 2
        assert len(not_done) == 0

    def test_wait_first_completed(self) -> None:
        f1: Future[int] = Future()
        f2: Future[int] = Future()
        pf1: PineconeFuture[int] = PineconeFuture(f1)
        pf2: PineconeFuture[int] = PineconeFuture(f2)

        # Only resolve the first one.
        f1.set_result(10)

        done, not_done = wait([pf1, pf2], timeout=0.1, return_when=FIRST_COMPLETED)
        assert pf1 in done
        assert pf2 in not_done

    def test_wait_first_exception(self) -> None:
        f1: Future[int] = Future()
        f2: Future[int] = Future()
        pf1: PineconeFuture[int] = PineconeFuture(f1)
        pf2: PineconeFuture[int] = PineconeFuture(f2)

        f1.set_exception(RuntimeError("fail"))

        done, not_done = wait([pf1, pf2], timeout=0.1, return_when=FIRST_EXCEPTION)
        assert pf1 in done
