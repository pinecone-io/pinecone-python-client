import pytest
from pinecone.grpc.future import PineconeGrpcFuture
from pinecone.exceptions import PineconeException

import grpc
from concurrent.futures import CancelledError, TimeoutError


def mock_grpc_future(
    mocker, done=False, cancelled=False, exception=None, running=False, result=None
):
    grpc_future = mocker.MagicMock()
    grpc_future.cancelled.return_value = cancelled
    grpc_future.done.return_value = done
    grpc_future.exception.return_value = exception
    grpc_future.running.return_value = running
    grpc_future.result.return_value = result
    return grpc_future


class FakeGrpcError(grpc.RpcError):
    def __init__(self, mocker):
        self._state = mocker.Mock()
        self._state.debug_error_string = "Test gRPC error"


class TestPineconeGrpcFuture:
    def test_wraps_grpc_future_already_done(self, mocker):
        grpc_future = mock_grpc_future(mocker, done=True, result="final result")

        future = PineconeGrpcFuture(grpc_future)

        assert not future.cancelled()
        assert future._state == "FINISHED"
        assert future.done()
        assert future.result() == "final result"

    def test_wraps_grpc_already_failed(self, mocker):
        grpc_future = mock_grpc_future(
            mocker, done=True, exception=Exception("Simulated gRPC error")
        )

        future = PineconeGrpcFuture(grpc_future)

        assert future._state == "FINISHED"
        assert future.done()
        with pytest.raises(Exception, match="Simulated gRPC error"):
            future.result()

    def test_wraps_grpc_future_already_running(self, mocker):
        grpc_future = mock_grpc_future(mocker, running=True)
        future = PineconeGrpcFuture(grpc_future)

        assert not future.cancelled()
        assert not future.done()
        assert future.running()
        assert future._state == "RUNNING"

    def test_wraps_grpc_future_already_cancelled(self, mocker):
        grpc_future = mock_grpc_future(mocker, cancelled=True)
        future = PineconeGrpcFuture(grpc_future)

        assert future.cancelled()
        assert future._state == "CANCELLED"
        assert future.done()
        with pytest.raises(CancelledError):
            future.result()

    def test_wraps_grpc_future_cancel_pending(self, mocker):
        grpc_future = mock_grpc_future(mocker)
        future = PineconeGrpcFuture(grpc_future)

        assert not future.cancelled()
        assert future._state == "PENDING"
        assert future.cancel()
        assert future._state == "CANCELLED"

        assert future.cancelled()
        assert not future.running()

        # Also cancel the grpc future
        grpc_future.cancel.assert_called_once()

        with pytest.raises(CancelledError):
            future.result()

    def test_cancel_already_cancelled(self, mocker):
        grpc_future = mock_grpc_future(mocker, cancelled=True, done=True)
        future = PineconeGrpcFuture(grpc_future)

        assert future.cancelled()
        assert future._state == "CANCELLED"

        # Cancel returns True even if the future is already cancelled
        assert future.cancel()

    def test_cancel_already_done(self, mocker):
        grpc_future = mock_grpc_future(mocker, done=True, result="final result")
        future = PineconeGrpcFuture(grpc_future)

        assert not future.cancelled()
        assert future._state == "FINISHED"

        # Can't cancel a future that is already done
        assert future.cancel() is False

        assert future.result() == "final result"

    def test_cancel_already_failed(self, mocker):
        grpc_future = mock_grpc_future(
            mocker, done=True, exception=Exception("Simulated gRPC error")
        )
        future = PineconeGrpcFuture(grpc_future)

        assert future._state == "FINISHED"
        assert future.done()

        # Can't cancel a future that is already done
        assert future.cancel() is False

        with pytest.raises(Exception, match="Simulated gRPC error"):
            future.result()

    def test_cancel_already_running(self, mocker):
        grpc_future = mock_grpc_future(mocker, running=True)
        future = PineconeGrpcFuture(grpc_future)

        assert not future.cancelled()
        assert not future.done()
        assert future.running()
        assert future._state == "RUNNING"

        # Can't cancel a future that is already running
        assert future.cancel() is False

        assert future.running()
        assert not future.done()
        assert not future.cancelled()
        assert future._state == "RUNNING"

    def test_cancel_pending(self, mocker):
        grpc_future = mock_grpc_future(mocker)
        future = PineconeGrpcFuture(grpc_future)

        assert not future.cancelled()
        assert not future.done()
        assert not future.running()
        assert future._state == "PENDING"

        # Cancel the future
        assert future.cancel()

        assert future.cancelled()
        assert future.done()
        assert not future.running()
        assert future._state == "CANCELLED"

        # Marks underlying grpc future as cancelled
        grpc_future.cancel.assert_called_once()

    def test_result_success(self, mocker):
        grpc_future = mock_grpc_future(mocker, running=True)
        future = PineconeGrpcFuture(grpc_future)

        assert not future.cancelled()
        assert not future.done()
        assert future.running()
        assert future._state == "RUNNING"

        # Update the state of the grpc future
        grpc_future.done.return_value = True
        grpc_future.running.return_value = False
        grpc_future.result.return_value = "final result"

        # Trigger the done callback to update the state of the wrapper future
        future._sync_state(grpc_future)

        assert future.result() == "final result"
        assert future.done()
        assert not future.cancelled()
        assert not future.running()

    def test_result_exception(self, mocker):
        grpc_future = mock_grpc_future(mocker, running=True)
        future = PineconeGrpcFuture(grpc_future)

        assert not future.cancelled()
        assert not future.done()
        assert future.running()
        assert future._state == "RUNNING"

        # Update the state of the grpc future
        grpc_future.done.return_value = True
        grpc_future.running.return_value = False
        grpc_future.result.side_effect = Exception("Simulated gRPC error")

        # Trigger the done callback to update the state of the wrapper future
        future._sync_state(grpc_future)

        with pytest.raises(Exception, match="Simulated gRPC error"):
            future.result()

        assert future.done()
        assert not future.cancelled()
        assert not future.running()

    def test_result_already_successful(self, mocker):
        grpc_future = mock_grpc_future(mocker, done=True, result="final result")
        future = PineconeGrpcFuture(grpc_future)

        assert not future.cancelled()
        assert future.done()
        assert not future.running()
        assert future._state == "FINISHED"

        assert future.result() == "final result"

    def test_result_already_failed(self, mocker):
        grpc_future = mock_grpc_future(
            mocker, done=True, exception=Exception("Simulated gRPC error")
        )
        future = PineconeGrpcFuture(grpc_future)

        assert not future.cancelled()
        assert future.done()
        assert not future.running()
        assert future._state == "FINISHED"

        with pytest.raises(Exception, match="Simulated gRPC error"):
            future.result()

    def test_result_already_cancelled(self, mocker):
        grpc_future = mock_grpc_future(mocker, cancelled=True, done=True)
        future = PineconeGrpcFuture(grpc_future)

        assert future.cancelled()
        assert future.done()
        assert not future.running()
        assert future._state == "CANCELLED"

        with pytest.raises(CancelledError):
            future.result()

    def test_result_timeout_running(self, mocker):
        grpc_future = mock_grpc_future(mocker, running=True)
        future = PineconeGrpcFuture(grpc_future)

        assert not future.cancelled()
        assert not future.done()
        assert future.running()
        assert future._state == "RUNNING"

        with pytest.raises(TimeoutError):
            future.result(timeout=1)

    def test_result_timeout_pending(self, mocker):
        grpc_future = mock_grpc_future(mocker)
        future = PineconeGrpcFuture(grpc_future)

        assert not future.cancelled()
        assert not future.done()
        assert not future.running()
        assert future._state == "PENDING"

        with pytest.raises(TimeoutError):
            future.result(timeout=1)

    def test_result_default_timeout(self, mocker):
        grpc_future = mock_grpc_future(mocker)
        future = PineconeGrpcFuture(grpc_future, timeout=1)

        assert not future.cancelled()
        assert not future.done()
        assert not future.running()
        assert future._state == "PENDING"

        with pytest.raises(TimeoutError):
            future.result()

        assert not future.cancelled()
        assert not future.done()
        assert not future.running()
        assert future._state == "PENDING"

    def test_result_catch_grpc_exceptions(self, mocker):
        grpc_future = mock_grpc_future(mocker)
        grpc_future.result.side_effect = FakeGrpcError(mocker)

        future = PineconeGrpcFuture(grpc_future)

        grpc_future.done.return_value = True
        future._sync_state(grpc_future)

        assert not future.cancelled()
        assert not future.running()
        assert future.done()
        assert future._state == "FINISHED"

        with pytest.raises(PineconeException, match="Test gRPC error"):
            future.result()

        assert isinstance(future.exception(), PineconeException)

        assert not future.cancelled()
        assert not future.running()
        assert future.done()
        assert future._state == "FINISHED"

    def test_exception_when_done_maps_grpc_exception(self, mocker):
        grpc_future = mock_grpc_future(mocker, done=True)
        grpc_future.exception.return_value = FakeGrpcError(mocker)

        future = PineconeGrpcFuture(grpc_future)

        assert not future.cancelled()
        assert not future.running()
        assert future.done()
        assert future._state == "FINISHED"

        assert isinstance(future.exception(), PineconeException)

    def test_exception_when_done_no_exceptions(self, mocker):
        grpc_future = mock_grpc_future(mocker, done=True)
        future = PineconeGrpcFuture(grpc_future)

        assert not future.cancelled()
        assert not future.running()
        assert future.done()
        assert future._state == "FINISHED"

        assert future.exception() is None

    def test_exception_when_running_default_timeout(self, mocker):
        grpc_future = mock_grpc_future(mocker, running=True)
        future = PineconeGrpcFuture(grpc_future, timeout=1)

        assert not future.cancelled()
        assert future.running()
        assert not future.done()
        assert future._state == "RUNNING"

        with pytest.raises(TimeoutError):
            future.exception()

    def test_exception_when_running_timeout(self, mocker):
        grpc_future = mock_grpc_future(mocker, running=True)
        future = PineconeGrpcFuture(grpc_future)

        assert not future.cancelled()
        assert future.running()
        assert not future.done()
        assert future._state == "RUNNING"

        with pytest.raises(TimeoutError):
            future.exception(timeout=1)

    def test_exception_when_pending_default_timeout(self, mocker):
        grpc_future = mock_grpc_future(mocker)
        future = PineconeGrpcFuture(grpc_future, timeout=1)

        assert not future.cancelled()
        assert not future.running()
        assert not future.done()
        assert future._state == "PENDING"

        with pytest.raises(TimeoutError):
            future.exception()

    def test_exception_when_pending_timeout(self, mocker):
        grpc_future = mock_grpc_future(mocker)

        future = PineconeGrpcFuture(grpc_future)

        assert not future.cancelled()
        assert not future.running()
        assert not future.done()
        assert future._state == "PENDING"

        with pytest.raises(TimeoutError):
            future.exception(timeout=1)

    def test_concurrent_futures_as_completed(self, mocker):
        grpc_future = mock_grpc_future(mocker, running=True)

        future = PineconeGrpcFuture(grpc_future, timeout=1)

        # Trigger the done callback
        grpc_future.done.return_value = True
        grpc_future.result.return_value = "success"
        future._sync_state(grpc_future)

        from concurrent.futures import as_completed

        for future in as_completed([future], timeout=1):
            assert future.result() == "success"
            assert future.done()
            assert not future.cancelled()

    def test_concurrent_futures_as_completed_timeout(self, mocker):
        grpc_future = mock_grpc_future(mocker, running=True)
        future1 = PineconeGrpcFuture(grpc_future, timeout=3)

        grpc_future2 = mock_grpc_future(mocker, done=True, result="success")
        future2 = PineconeGrpcFuture(grpc_future2, timeout=3)

        grpc_future3 = mock_grpc_future(mocker, done=True, cancelled=True)
        future3 = PineconeGrpcFuture(grpc_future3, timeout=3)

        from concurrent.futures import as_completed

        completed_count = 0
        with pytest.raises(TimeoutError):
            for f in as_completed([future1, future2, future3], timeout=1):
                completed_count += 1

        assert completed_count == 1

    def test_concurrent_futures_wait_first_completed(self, mocker):
        grpc_future1 = mock_grpc_future(mocker, done=True, result="success")
        future1 = PineconeGrpcFuture(grpc_future1)

        grpc_future2 = mock_grpc_future(mocker, running=True)
        future2 = PineconeGrpcFuture(grpc_future2)

        from concurrent.futures import wait, FIRST_COMPLETED

        done, not_done = wait([future1, future2], timeout=1, return_when=FIRST_COMPLETED)
        assert len(done) == 1
        assert len(not_done) == 1
        assert done.pop().result() == "success"

        # order should not matter
        done, not_done = wait([future2, future1], timeout=1, return_when=FIRST_COMPLETED)
        assert len(done) == 1
        assert len(not_done) == 1
        assert done.pop().result() == "success"

    def test_concurrent_futures_wait_all_completed(self, mocker):
        grpc_future1 = mock_grpc_future(mocker, done=True, result="success")
        future1 = PineconeGrpcFuture(grpc_future1)

        grpc_future2 = mock_grpc_future(mocker, done=True, result="success")
        future2 = PineconeGrpcFuture(grpc_future2)

        from concurrent.futures import wait, ALL_COMPLETED

        done, not_done = wait([future1, future2], timeout=3, return_when=ALL_COMPLETED)
        assert len(done) == 2
        assert len(not_done) == 0
        assert all(f.result() == "success" for f in done)

    def test_concurrent_futures_wait_first_exception(self, mocker):
        grpc_future1 = mock_grpc_future(mocker)
        future1 = PineconeGrpcFuture(grpc_future1)

        grpc_future2 = mock_grpc_future(mocker, done=True)
        grpc_future2.exception.return_value = Exception("Simulated gRPC error")
        future2 = PineconeGrpcFuture(grpc_future2)

        from concurrent.futures import wait, FIRST_EXCEPTION

        done, not_done = wait([future1, future2], return_when=FIRST_EXCEPTION)
        assert len(done) == 1
        assert len(not_done) == 1

        failed_future = done.pop()
        assert isinstance(failed_future.exception(), Exception)
        assert failed_future.exception().args == ("Simulated gRPC error",)

    def test_concurrent_futures_wait_timeout(self, mocker):
        grpc_future1 = mock_grpc_future(mocker, running=True)
        future1 = PineconeGrpcFuture(grpc_future1)

        grpc_future2 = mock_grpc_future(mocker, running=True)
        future2 = PineconeGrpcFuture(grpc_future2)

        from concurrent.futures import wait, FIRST_COMPLETED

        done, not_done = wait([future1, future2], timeout=1, return_when=FIRST_COMPLETED)
        assert len(done) == 0
        assert len(not_done) == 2

    def test_concurrent_futures_wait_all_timeout(self, mocker):
        grpc_future1 = mock_grpc_future(mocker, running=True)
        future1 = PineconeGrpcFuture(grpc_future1)

        grpc_future2 = mock_grpc_future(mocker, running=True)
        future2 = PineconeGrpcFuture(grpc_future2)

        from concurrent.futures import wait, ALL_COMPLETED

        done, not_done = wait([future1, future2], timeout=1, return_when=ALL_COMPLETED)
        assert len(done) == 0
        assert len(not_done) == 2
