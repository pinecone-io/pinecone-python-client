"""Unit tests for Index list and list_paginated methods.

These tests replace integration tests that were making real API calls to test
keyword argument translation to API calls. They test the argument translation
logic directly without requiring API access.
"""

import pytest

from pinecone.db_data import _Index, _IndexAsyncio

from tests.fixtures import make_list_response, make_list_item, make_pagination


class TestIndexListPaginated:
    """Test parameter translation in Index.list_paginated()"""

    def setup_method(self):
        self.index = _Index(api_key="test-key", host="https://test.pinecone.io")

    def test_list_paginated_with_all_params(self, mocker):
        """Test list_paginated with all parameters"""
        mocker.patch.object(self.index._vector_api, "list_vectors", autospec=True)

        mock_response = make_list_response(
            vectors=[make_list_item(id="vec1"), make_list_item(id="vec2")],
            namespace="test-ns",
            pagination=None,
        )
        self.index._vector_api.list_vectors.return_value = mock_response

        result = self.index.list_paginated(
            prefix="pref", limit=10, pagination_token="token123", namespace="test-ns"
        )

        # Verify API was called with correct arguments (None values filtered out)
        self.index._vector_api.list_vectors.assert_called_once_with(
            prefix="pref", limit=10, pagination_token="token123", namespace="test-ns"
        )
        assert result == mock_response

    def test_list_paginated_with_partial_params(self, mocker):
        """Test list_paginated with only prefix and namespace"""
        mocker.patch.object(self.index._vector_api, "list_vectors", autospec=True)

        mock_response = make_list_response(
            vectors=[make_list_item(id="vec1")], namespace="test-ns", pagination=None
        )
        self.index._vector_api.list_vectors.return_value = mock_response

        result = self.index.list_paginated(prefix="pref", namespace="test-ns")

        # Verify only non-None params are passed
        self.index._vector_api.list_vectors.assert_called_once_with(
            prefix="pref", namespace="test-ns"
        )
        assert result == mock_response

    def test_list_paginated_with_no_params(self, mocker):
        """Test list_paginated with no parameters"""
        mocker.patch.object(self.index._vector_api, "list_vectors", autospec=True)

        mock_response = make_list_response(
            vectors=[make_list_item(id="vec1")], namespace="", pagination=None
        )
        self.index._vector_api.list_vectors.return_value = mock_response

        result = self.index.list_paginated()

        # Verify empty dict is passed (all None values filtered out)
        self.index._vector_api.list_vectors.assert_called_once_with()
        assert result == mock_response

    def test_list_paginated_filters_none_values(self, mocker):
        """Test that None values are filtered out by parse_non_empty_args"""
        mocker.patch.object(self.index._vector_api, "list_vectors", autospec=True)

        mock_response = make_list_response(vectors=[], namespace="test-ns", pagination=None)
        self.index._vector_api.list_vectors.return_value = mock_response

        self.index.list_paginated(
            prefix=None, limit=None, pagination_token=None, namespace="test-ns"
        )

        # Verify None values are not passed to API
        self.index._vector_api.list_vectors.assert_called_once_with(namespace="test-ns")

    def test_list_paginated_with_pagination_response(self, mocker):
        """Test list_paginated returns response with pagination"""
        mocker.patch.object(self.index._vector_api, "list_vectors", autospec=True)

        mock_pagination = make_pagination(next="next-token-123")
        mock_response = make_list_response(
            vectors=[make_list_item(id="vec1"), make_list_item(id="vec2")],
            namespace="test-ns",
            pagination=mock_pagination,
        )
        self.index._vector_api.list_vectors.return_value = mock_response

        result = self.index.list_paginated(prefix="pref", limit=2, namespace="test-ns")

        assert result.pagination is not None
        assert result.pagination.next == "next-token-123"
        assert len(result.vectors) == 2


class TestIndexList:
    """Test generator behavior in Index.list()"""

    def setup_method(self):
        self.index = _Index(api_key="test-key", host="https://test.pinecone.io")

    def test_list_single_page(self, mocker):
        """Test list with single page (no pagination)"""
        mocker.patch.object(self.index._vector_api, "list_vectors", autospec=True)

        mock_response = make_list_response(
            vectors=[
                make_list_item(id="vec1"),
                make_list_item(id="vec2"),
                make_list_item(id="vec3"),
            ],
            namespace="test-ns",
            pagination=None,
        )
        self.index._vector_api.list_vectors.return_value = mock_response

        results = list(self.index.list(prefix="pref", namespace="test-ns"))

        # Should yield one page with all IDs
        assert len(results) == 1
        assert results[0] == ["vec1", "vec2", "vec3"]
        self.index._vector_api.list_vectors.assert_called_once_with(
            prefix="pref", namespace="test-ns"
        )

    def test_list_multiple_pages(self, mocker):
        """Test list with multiple pages (pagination)"""
        mocker.patch.object(self.index._vector_api, "list_vectors", autospec=True)

        # First page response
        mock_pagination1 = make_pagination(next="token-page2")
        mock_response1 = make_list_response(
            vectors=[make_list_item(id="vec1"), make_list_item(id="vec2")],
            namespace="test-ns",
            pagination=mock_pagination1,
        )

        # Second page response
        mock_pagination2 = make_pagination(next="token-page3")
        mock_response2 = make_list_response(
            vectors=[make_list_item(id="vec3"), make_list_item(id="vec4")],
            namespace="test-ns",
            pagination=mock_pagination2,
        )

        # Third page response (no pagination - last page)
        mock_response3 = make_list_response(
            vectors=[make_list_item(id="vec5")], namespace="test-ns", pagination=None
        )

        self.index._vector_api.list_vectors.side_effect = [
            mock_response1,
            mock_response2,
            mock_response3,
        ]

        results = list(self.index.list(prefix="pref", limit=2, namespace="test-ns"))

        # Should yield three pages
        assert len(results) == 3
        assert results[0] == ["vec1", "vec2"]
        assert results[1] == ["vec3", "vec4"]
        assert results[2] == ["vec5"]

        # Verify API was called three times with correct pagination tokens
        assert self.index._vector_api.list_vectors.call_count == 3
        self.index._vector_api.list_vectors.assert_any_call(
            prefix="pref", limit=2, namespace="test-ns"
        )
        self.index._vector_api.list_vectors.assert_any_call(
            prefix="pref", limit=2, namespace="test-ns", pagination_token="token-page2"
        )
        self.index._vector_api.list_vectors.assert_any_call(
            prefix="pref", limit=2, namespace="test-ns", pagination_token="token-page3"
        )

    def test_list_empty_results(self, mocker):
        """Test list with empty results"""
        mocker.patch.object(self.index._vector_api, "list_vectors", autospec=True)

        mock_response = make_list_response(vectors=[], namespace="test-ns", pagination=None)
        self.index._vector_api.list_vectors.return_value = mock_response

        results = list(self.index.list(prefix="pref", namespace="test-ns"))

        # Should yield no pages (empty generator)
        assert len(results) == 0
        self.index._vector_api.list_vectors.assert_called_once_with(
            prefix="pref", namespace="test-ns"
        )

    def test_list_empty_page_with_pagination(self, mocker):
        """Test list with empty page but pagination token (edge case)"""
        mocker.patch.object(self.index._vector_api, "list_vectors", autospec=True)

        # First page: empty but has pagination
        mock_pagination1 = make_pagination(next="token-page2")
        mock_response1 = make_list_response(
            vectors=[], namespace="test-ns", pagination=mock_pagination1
        )

        # Second page: has results
        mock_response2 = make_list_response(
            vectors=[make_list_item(id="vec1")], namespace="test-ns", pagination=None
        )

        self.index._vector_api.list_vectors.side_effect = [mock_response1, mock_response2]

        results = list(self.index.list(prefix="pref", namespace="test-ns"))

        # Should yield one page (first was empty, second has results)
        assert len(results) == 1
        assert results[0] == ["vec1"]
        assert self.index._vector_api.list_vectors.call_count == 2


@pytest.mark.asyncio
class TestIndexAsyncioListPaginated:
    """Test parameter translation in _IndexAsyncio.list_paginated()"""

    def setup_method(self):
        # Note: We'll mock setup_async_openapi_client in each test to avoid event loop issues
        pass

    def _create_index(self, mocker):
        """Helper to create async index with mocked setup"""
        mock_vector_api = mocker.Mock()
        # Make list_vectors an async mock
        mock_vector_api.list_vectors = mocker.AsyncMock()
        mocker.patch(
            "pinecone.db_data.index_asyncio.setup_async_openapi_client",
            return_value=mock_vector_api,
        )
        return _IndexAsyncio(api_key="test-key", host="https://test.pinecone.io")

    async def test_list_paginated_with_all_params(self, mocker):
        """Test list_paginated with all parameters"""
        index = self._create_index(mocker)

        mock_response = make_list_response(
            vectors=[make_list_item(id="vec1"), make_list_item(id="vec2")],
            namespace="test-ns",
            pagination=None,
        )
        index._vector_api.list_vectors.return_value = mock_response

        result = await index.list_paginated(
            prefix="pref", limit=10, pagination_token="token123", namespace="test-ns"
        )

        # Verify API was called with correct arguments
        index._vector_api.list_vectors.assert_called_once_with(
            prefix="pref", limit=10, pagination_token="token123", namespace="test-ns"
        )
        assert result == mock_response

    async def test_list_paginated_with_partial_params(self, mocker):
        """Test list_paginated with only prefix and namespace"""
        index = self._create_index(mocker)

        mock_response = make_list_response(
            vectors=[make_list_item(id="vec1")], namespace="test-ns", pagination=None
        )
        index._vector_api.list_vectors.return_value = mock_response

        result = await index.list_paginated(prefix="pref", namespace="test-ns")

        # Verify only non-None params are passed
        index._vector_api.list_vectors.assert_called_once_with(prefix="pref", namespace="test-ns")
        assert result == mock_response

    async def test_list_paginated_with_no_params(self, mocker):
        """Test list_paginated with no parameters"""
        index = self._create_index(mocker)

        mock_response = make_list_response(
            vectors=[make_list_item(id="vec1")], namespace="", pagination=None
        )
        index._vector_api.list_vectors.return_value = mock_response

        result = await index.list_paginated()

        # Verify empty dict is passed
        index._vector_api.list_vectors.assert_called_once_with()
        assert result == mock_response

    async def test_list_paginated_filters_none_values(self, mocker):
        """Test that None values are filtered out"""
        index = self._create_index(mocker)

        mock_response = make_list_response(vectors=[], namespace="test-ns", pagination=None)
        index._vector_api.list_vectors.return_value = mock_response

        await index.list_paginated(
            prefix=None, limit=None, pagination_token=None, namespace="test-ns"
        )

        # Verify None values are not passed to API
        index._vector_api.list_vectors.assert_called_once_with(namespace="test-ns")


@pytest.mark.asyncio
class TestIndexAsyncioList:
    """Test async generator behavior in _IndexAsyncio.list()"""

    def setup_method(self):
        # Note: We'll mock setup_async_openapi_client in each test to avoid event loop issues
        pass

    def _create_index(self, mocker):
        """Helper to create async index with mocked setup"""
        mock_vector_api = mocker.Mock()
        # Make list_vectors an async mock
        mock_vector_api.list_vectors = mocker.AsyncMock()
        mocker.patch(
            "pinecone.db_data.index_asyncio.setup_async_openapi_client",
            return_value=mock_vector_api,
        )
        return _IndexAsyncio(api_key="test-key", host="https://test.pinecone.io")

    async def test_list_single_page(self, mocker):
        """Test list with single page (no pagination)"""
        index = self._create_index(mocker)

        mock_response = make_list_response(
            vectors=[
                make_list_item(id="vec1"),
                make_list_item(id="vec2"),
                make_list_item(id="vec3"),
            ],
            namespace="test-ns",
            pagination=None,
        )
        index._vector_api.list_vectors.return_value = mock_response

        results = [page async for page in index.list(prefix="pref", namespace="test-ns")]

        # Should yield one page with all IDs
        assert len(results) == 1
        assert results[0] == ["vec1", "vec2", "vec3"]
        index._vector_api.list_vectors.assert_called_once_with(prefix="pref", namespace="test-ns")

    async def test_list_multiple_pages(self, mocker):
        """Test list with multiple pages (pagination)"""
        index = self._create_index(mocker)

        # First page response
        mock_pagination1 = make_pagination(next="token-page2")
        mock_response1 = make_list_response(
            vectors=[make_list_item(id="vec1"), make_list_item(id="vec2")],
            namespace="test-ns",
            pagination=mock_pagination1,
        )

        # Second page response
        mock_pagination2 = make_pagination(next="token-page3")
        mock_response2 = make_list_response(
            vectors=[make_list_item(id="vec3"), make_list_item(id="vec4")],
            namespace="test-ns",
            pagination=mock_pagination2,
        )

        # Third page response (no pagination - last page)
        mock_response3 = make_list_response(
            vectors=[make_list_item(id="vec5")], namespace="test-ns", pagination=None
        )

        index._vector_api.list_vectors.side_effect = [
            mock_response1,
            mock_response2,
            mock_response3,
        ]

        results = [page async for page in index.list(prefix="pref", limit=2, namespace="test-ns")]

        # Should yield three pages
        assert len(results) == 3
        assert results[0] == ["vec1", "vec2"]
        assert results[1] == ["vec3", "vec4"]
        assert results[2] == ["vec5"]

        # Verify API was called three times with correct pagination tokens
        assert index._vector_api.list_vectors.call_count == 3
        index._vector_api.list_vectors.assert_any_call(prefix="pref", limit=2, namespace="test-ns")
        index._vector_api.list_vectors.assert_any_call(
            prefix="pref", limit=2, namespace="test-ns", pagination_token="token-page2"
        )
        index._vector_api.list_vectors.assert_any_call(
            prefix="pref", limit=2, namespace="test-ns", pagination_token="token-page3"
        )

    async def test_list_empty_results(self, mocker):
        """Test list with empty results"""
        index = self._create_index(mocker)

        mock_response = make_list_response(vectors=[], namespace="test-ns", pagination=None)
        index._vector_api.list_vectors.return_value = mock_response

        results = [page async for page in index.list(prefix="pref", namespace="test-ns")]

        # Should yield no pages (empty generator)
        assert len(results) == 0
        index._vector_api.list_vectors.assert_called_once_with(prefix="pref", namespace="test-ns")
