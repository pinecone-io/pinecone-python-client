from pinecone.errors.exceptions import PineconeTimeoutError


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
