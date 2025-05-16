import dotenv
import asyncio
import logging
from pinecone import PineconeAsyncio

dotenv.load_dotenv()

logging.basicConfig(level=logging.DEBUG)


async def main():
    async with PineconeAsyncio(host="http://localhost:8000") as pc:
        await pc.db.index.list()


asyncio.run(main())
