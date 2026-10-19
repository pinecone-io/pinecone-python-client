"""``PineconeTypeError`` for an unexpected keyword at client setup and in ``create_index``.

The ``Pinecone`` and ``AsyncPinecone`` constructors and the legacy ``create_index``
shim raise ``PineconeTypeError`` for an unexpected keyword, so a single
``except PineconeError`` handler around client setup and legacy index creation
catches all of them. ``PineconeTypeError`` derives from ``TypeError``, so the older
``except TypeError`` handler keeps catching them as well; the assertions below pin
both directions.

The assistants methods reject an unexpected keyword through their 9.x alias layer
with ``PineconeValueError`` instead. This module also pins that, the "Did you mean"
hint on both constructors, and the error-handling guide naming each exception class
the condition produces.
"""

from __future__ import annotations

import pathlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from pinecone import Pinecone
from pinecone.async_client.pinecone import AsyncPinecone
from pinecone.errors.exceptions import PineconeError, PineconeTypeError, PineconeValueError


class TestPineconeConstructor:
    def test_raises_pinecone_type_error(self) -> None:
        with pytest.raises(PineconeTypeError) as excinfo:
            Pinecone(api_key="test-key", bogus_kwarg=True)
        assert type(excinfo.value) is PineconeTypeError

    def test_caught_by_pinecone_error_handler(self) -> None:
        try:
            Pinecone(api_key="test-key", bogus_kwarg=True)
        except PineconeError as exc:
            assert isinstance(exc, PineconeTypeError)
        else:
            pytest.fail("Pinecone() accepted an unexpected keyword argument")

    def test_still_caught_by_type_error_handler(self) -> None:
        try:
            Pinecone(api_key="test-key", bogus_kwarg=True)
        except TypeError as exc:
            assert isinstance(exc, PineconeTypeError)
        else:
            pytest.fail("Pinecone() accepted an unexpected keyword argument")

    def test_message_and_args_unchanged(self) -> None:
        expected = "Pinecone() got unexpected keyword arguments: ['bogus_kwarg']"
        with pytest.raises(PineconeTypeError) as excinfo:
            Pinecone(api_key="test-key", bogus_kwarg=True)
        assert str(excinfo.value) == expected
        assert excinfo.value.args == (expected,)

    def test_pool_threads_is_still_accepted(self) -> None:
        pc = Pinecone(api_key="test-key", pool_threads=4)
        assert pc._legacy_pool_threads == 4

    def test_documented_in_the_raises_block(self) -> None:
        assert Pinecone.__doc__ is not None
        raises_block = Pinecone.__doc__.split("Raises:", 1)[1].split("Examples:", 1)[0]
        assert "PineconeTypeError" in raises_block


class TestAsyncPineconeConstructor:
    def test_raises_pinecone_type_error(self) -> None:
        with pytest.raises(PineconeTypeError) as excinfo:
            AsyncPinecone(api_key="test-key", bogus_kwarg=True)  # type: ignore[arg-type]
        assert type(excinfo.value) is PineconeTypeError
        assert (
            str(excinfo.value)
            == "AsyncPinecone() got unexpected keyword arguments: ['bogus_kwarg']"
        )

    def test_caught_by_both_handlers(self) -> None:
        for handler in (PineconeError, TypeError):
            with pytest.raises(handler):
                AsyncPinecone(api_key="test-key", bogus_kwarg=True)  # type: ignore[arg-type]

    def test_documented_in_the_raises_block(self) -> None:
        assert AsyncPinecone.__doc__ is not None
        raises_block = AsyncPinecone.__doc__.split("Raises:", 1)[1].split("Examples:", 1)[0]
        assert ":exc:`PineconeTypeError`" in raises_block


class TestLegacyCreateIndexShim:
    @staticmethod
    def _pc_with_mock_indexes() -> Pinecone:
        pc = Pinecone(api_key="test-key")
        pc._indexes = MagicMock()
        return pc

    def test_raises_pinecone_type_error(self) -> None:
        with pytest.raises(PineconeTypeError) as excinfo:
            self._pc_with_mock_indexes().create_index(name="x", schema={"fields": {}})
        assert type(excinfo.value) is PineconeTypeError

    def test_caught_by_pinecone_error_handler(self) -> None:
        try:
            self._pc_with_mock_indexes().create_index(name="x", schema={"fields": {}})
        except PineconeError as exc:
            assert isinstance(exc, PineconeTypeError)
        else:
            pytest.fail("create_index() accepted schema=")

    def test_still_caught_by_type_error_handler(self) -> None:
        try:
            self._pc_with_mock_indexes().create_index(name="x", schema={"fields": {}})
        except TypeError as exc:
            assert isinstance(exc, PineconeTypeError)
        else:
            pytest.fail("create_index() accepted schema=")

    async def test_async_twin_raises_the_same_type(self) -> None:
        pc = AsyncPinecone(api_key="test-key")
        mock_indexes = MagicMock()
        mock_indexes.create = AsyncMock(return_value=MagicMock())
        pc._indexes = mock_indexes
        try:
            await pc.create_index(name="x", schema={"fields": {}})
        except PineconeError as exc:
            assert type(exc) is PineconeTypeError
            assert isinstance(exc, TypeError)
        else:
            pytest.fail("create_index() accepted schema=")
        mock_indexes.create.assert_not_called()


class TestAssistantsLegacyAliasPaths:
    def test_reject_with_pinecone_value_error_not_a_type_error(self) -> None:
        pc = Pinecone(api_key="test-key")
        with pytest.raises(PineconeValueError) as excinfo:
            pc.assistant.create(name="a", bogus_kwarg=True)
        assert not isinstance(excinfo.value, TypeError)


class TestErrorHandlingGuide:
    @staticmethod
    def _misspelled_keyword_section() -> str:
        guide = pathlib.Path(__file__).parents[2] / "docs" / "guides" / "error-handling.md"
        return guide.read_text(encoding="utf-8").split("Every class below", 1)[0]

    def test_names_every_class_the_condition_produces(self) -> None:
        section = self._misspelled_keyword_section()
        assert "`PineconeTypeError`" in section
        assert "`PineconeValueError`" in section
        assert "`TypeError`" in section

    def test_no_longer_claims_the_condition_is_never_a_pinecone_error(self) -> None:
        section = self._misspelled_keyword_section()
        assert "not a\n`PineconeTypeError`" not in section
        assert "Neither is a\n`PineconeError`" not in section


class TestBothSitesAgree:
    def test_one_pinecone_error_handler_covers_both_sites(self) -> None:
        caught: list[type[BaseException]] = []
        for call in (
            lambda: Pinecone(api_key="test-key", bogus_kwarg=True),
            lambda: Pinecone(api_key="test-key").create_index(name="x", schema={"fields": {}}),
        ):
            try:
                call()
            except PineconeError as exc:
                caught.append(type(exc))
        assert caught == [PineconeTypeError, PineconeTypeError]


class TestMisspelledKeywordHint:
    @pytest.mark.parametrize("cls", [Pinecone, AsyncPinecone])
    def test_names_the_closest_accepted_keyword(self, cls: type) -> None:
        with pytest.raises(PineconeTypeError) as excinfo:
            cls(api_key="test-key", timout=5)
        assert str(excinfo.value) == (
            f"{cls.__name__}() got unexpected keyword arguments: ['timout']. "
            "Did you mean 'timeout'?"
        )

    @pytest.mark.parametrize("cls", [Pinecone, AsyncPinecone])
    def test_hint_follows_the_first_keyword_with_a_match(self, cls: type) -> None:
        with pytest.raises(PineconeTypeError) as excinfo:
            cls(api_key="test-key", zzz=1, ssl_verfy=False)
        assert str(excinfo.value).endswith("['ssl_verfy', 'zzz']. Did you mean 'ssl_verify'?")
