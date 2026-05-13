from __future__ import annotations

from pinecone.errors.exceptions import ApiError, PineconeError, PineconeTimeoutError, RateLimitError


def test_pinecone_timeout_error_extends_builtin_timeout_error() -> None:
    err = PineconeTimeoutError("timed out")
    assert isinstance(err, TimeoutError)
    assert isinstance(err, PineconeTimeoutError)


def test_pinecone_timeout_error_caught_by_builtin_timeout_error() -> None:
    caught = False
    try:
        raise PineconeTimeoutError("timed out")
    except TimeoutError:
        caught = True
    assert caught


def test_rate_limit_error_is_api_error_subclass() -> None:
    err = RateLimitError()
    assert isinstance(err, ApiError)
    assert isinstance(err, PineconeError)


def test_rate_limit_error_default_status_code_is_429() -> None:
    err = RateLimitError()
    assert err.status_code == 429


def test_rate_limit_error_default_message() -> None:
    err = RateLimitError()
    assert err.message == "Rate limit exceeded"


def test_rate_limit_error_retry_after_defaults_to_none() -> None:
    err = RateLimitError()
    assert err.retry_after is None


def test_rate_limit_error_retry_after_stored() -> None:
    err = RateLimitError(retry_after=30)
    assert err.retry_after == 30


def test_rate_limit_error_caught_by_api_error_block() -> None:
    caught = False
    try:
        raise RateLimitError(retry_after=5)
    except ApiError as exc:
        caught = True
        assert exc.status_code == 429
    assert caught


def test_rate_limit_error_str_includes_status_and_error_code() -> None:
    err = RateLimitError(
        message="too many requests",
        error_code="RESOURCE_EXHAUSTED",
        request_id="req-xyz",
        retry_after=10,
    )
    assert str(err) == "[429 RESOURCE_EXHAUSTED] too many requests (request_id: req-xyz)"


def test_rate_limit_error_importable_from_pinecone_errors() -> None:
    from pinecone.errors import RateLimitError as _RLE  # noqa: F401


def test_rate_limit_error_importable_from_top_level() -> None:
    import pinecone

    assert pinecone.RateLimitError is RateLimitError
