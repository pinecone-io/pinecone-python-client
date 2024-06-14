from pinecone import PineconeException
import pytest


class TestListErrors:
    def test_list_change_prefix_while_fetching_next_page(self, idx, list_namespace):
        results = idx.list_paginated(prefix="99", limit=5, namespace=list_namespace)
        with pytest.raises(PineconeException) as e:
            idx.list_paginated(prefix="98", limit=5, pagination_token=results.pagination.next)
        assert "prefix" in str(e.value)

    @pytest.mark.skip(reason="Bug filed")
    def test_list_change_namespace_while_fetching_next_page(self, idx, namespace):
        results = idx.list_paginated(limit=5, namespace=namespace)
        with pytest.raises(PineconeException) as e:
            idx.list_paginated(limit=5, namespace="new-namespace", pagination_token=results.pagination.next)
        assert "namespace" in str(e.value)
