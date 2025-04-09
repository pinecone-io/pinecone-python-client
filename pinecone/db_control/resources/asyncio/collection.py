import logging
from typing import TYPE_CHECKING


from pinecone.db_control.models import CollectionList

from pinecone.db_control.request_factory import PineconeDBControlRequestFactory

logger = logging.getLogger(__name__)
""" @private """

if TYPE_CHECKING:
    pass


class CollectionResourceAsyncio:
    def __init__(self, index_api):
        self.index_api = index_api

    async def create(self, name: str, source: str):
        req = PineconeDBControlRequestFactory.create_collection_request(name=name, source=source)
        await self.index_api.create_collection(create_collection_request=req)

    async def list(self) -> CollectionList:
        response = await self.index_api.list_collections()
        return CollectionList(response)

    async def delete(self, name: str):
        await self.index_api.delete_collection(name)

    async def describe(self, name: str):
        return await self.index_api.describe_collection(name).to_dict()
