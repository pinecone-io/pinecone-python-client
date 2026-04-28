"""PineconeFuture — a thin wrapper around concurrent.futures.Future.

Provides SDK-specific timeout defaults and exception translation so that
callers get :class:`~pinecone.errors.PineconeTimeoutError` instead of the
stdlib ``TimeoutError`` when a result is not ready in time.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future
from typing import Any, TypeVar

from pinecone.errors.exceptions import PineconeTimeoutError

_T = TypeVar("_T")

_DEFAULT_TIMEOUT: float = 5.0


class PineconeFuture(Future["_T"]):
    """Future returned by ``GrpcIndex.*_async()`` methods.

    Wraps a :class:`concurrent.futures.Future` and is fully compatible with
    :func:`concurrent.futures.as_completed` and
    :func:`concurrent.futures.wait`.

    The default :meth:`result` timeout is **5 seconds**.  When the timeout
    elapses, :class:`~pinecone.errors.PineconeTimeoutError` is raised with
    the message ``"deadline exceeded"``.

    Examples:
        Upsert vectors asynchronously and wait for the result:

        .. code-block:: python

            from pinecone.grpc import GrpcIndex
            idx = GrpcIndex(host="article-search-abc123.svc.pinecone.io", api_key="your-api-key")
            future = idx.upsert_async(vectors=[("article-101", [0.012, -0.087, 0.153, ...])])
            result = future.result()  # blocks up to 5 seconds
            result.upserted_count
            # 1

        Fire multiple upserts concurrently and collect results:

        .. code-block:: python

            from concurrent.futures import as_completed
            futures = [
                idx.upsert_async(vectors=[("article-101", [0.012, -0.087, 0.153, ...])]),
                idx.upsert_async(vectors=[("article-102", [0.045, 0.021, -0.064, ...])]),
            ]
            for future in as_completed(futures):
                print(future.result().upserted_count)
    """

    def __init__(self, underlying: Future[_T]) -> None:
        # Do NOT call super().__init__() — we delegate everything to the
        # underlying future.  We *do* need the internal state that Future
        # expects however, so we initialise ourselves as a bare Future and
        # then wire up callbacks so our own state mirrors the underlying one.
        super().__init__()
        self._underlying = underlying

        # Mirror terminal state from the underlying future into *self* so
        # that concurrent.futures infrastructure (as_completed / wait) which
        # inspects our internal condition/state sees the correct values.
        self._underlying.add_done_callback(self._propagate_state)

    # ------------------------------------------------------------------
    # State propagation
    # ------------------------------------------------------------------

    def _propagate_state(self, _fut: Future[_T]) -> None:
        """Copy the terminal state of the underlying future into *self*."""
        if self._underlying.cancelled():
            # Mark ourselves cancelled so wait/as_completed see it.
            super().cancel()
            super().set_running_or_notify_cancel()
        elif self._underlying.exception() is not None:
            try:
                super().set_exception(self._underlying.exception())
            except Exception:
                pass  # already in terminal state
        else:
            try:
                super().set_result(self._underlying.result(timeout=0))
            except Exception:
                pass  # already in terminal state

    # ------------------------------------------------------------------
    # Public interface — delegates to the underlying future
    # ------------------------------------------------------------------

    def result(self, timeout: float | None = _DEFAULT_TIMEOUT) -> _T:
        """Return the result of the call that the future represents.

        Args:
            timeout: Maximum seconds to wait.  Defaults to 5.0.
                Pass ``None`` to block indefinitely.

        Returns:
            The result value set by the underlying future.

        Raises:
            PineconeTimeoutError: If *timeout* seconds elapse before the
                result is available.

        Examples:
            Wait for the result with the default 5-second timeout:

            .. code-block:: python

                future = idx.upsert_async(vectors=[("article-101", [0.012, -0.087, 0.153, ...])])
                result = future.result()
                result.upserted_count  # 1

            Wait up to 30 seconds for a large batch to complete:

            .. code-block:: python

                future = idx.upsert_async(vectors=large_batch)
                result = future.result(timeout=30.0)

            Block indefinitely until the operation finishes:

            .. code-block:: python

                result = future.result(timeout=None)
        """
        try:
            return self._underlying.result(timeout=timeout)
        except TimeoutError:
            raise PineconeTimeoutError("deadline exceeded") from None

    def exception(self, timeout: float | None = _DEFAULT_TIMEOUT) -> BaseException | None:
        """Return the exception raised by the call, or ``None``.

        Args:
            timeout: Maximum seconds to wait.  Defaults to 5.0.

        Raises:
            PineconeTimeoutError: If *timeout* seconds elapse.
        """
        try:
            return self._underlying.exception(timeout=timeout)
        except TimeoutError:
            raise PineconeTimeoutError("deadline exceeded") from None

    def cancel(self) -> bool:
        """Attempt to cancel the underlying call.

        Returns ``True`` if the call was successfully cancelled, ``False``
        if the call has already completed or is running.
        """
        return self._underlying.cancel()

    def cancelled(self) -> bool:
        """Return ``True`` if the call was successfully cancelled."""
        return self._underlying.cancelled()

    def done(self) -> bool:
        """Return ``True`` if the call has completed or was cancelled."""
        return self._underlying.done()

    def running(self) -> bool:
        """Return ``True`` if the call is currently being executed."""
        return self._underlying.running()

    def add_done_callback(self, fn: Callable[..., Any]) -> None:
        """Attach a callable to be called when the future finishes.

        The callable will be called with the future as its only argument.
        """
        self._underlying.add_done_callback(lambda _underlying: fn(self))
