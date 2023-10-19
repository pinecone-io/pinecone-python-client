import os
from .config import Config

DEFAULT_CONTROLLER_HOST = "https://api.pinecone.io"


class PineconeConfig(Config):
    def __init__(self, api_key: str = None, host: str = None, **kwargs):
        host = host or kwargs.get("host") or os.getenv("PINECONE_CONTROLLER_HOST") or DEFAULT_CONTROLLER_HOST
        super().__init__(api_key=api_key, host=host, **kwargs)
