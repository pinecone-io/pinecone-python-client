import logging
from typing import Any

from pinecone.db_control.models import CollectionList

from pinecone.db_control.request_factory import PineconeDBControlRequestFactory
from pinecone.utils import require_kwargs

logger = logging.getLogger(__name__)
""" :meta private: """


class CollectionResourceAsyncio:
    def __init__(self, index_api):
        self.index_api = index_api

    @require_kwargs
    async def create(self, *, name: str, source: str) -> None:
        req = PineconeDBControlRequestFactory.create_collection_request(name=name, source=source)
        await self.index_api.create_collection(create_collection_request=req)

    @require_kwargs
    async def list(self) -> CollectionList:
        response = await self.index_api.list_collections()
        return CollectionList(response)

    @require_kwargs
    async def delete(self, *, name: str) -> None:
        await self.index_api.delete_collection(name)

    @require_kwargs
    async def describe(self, *, name: str) -> dict[str, Any]:
        from typing import cast

        result = await self.index_api.describe_collection(name)
        return cast(dict[str, Any], result.to_dict())
