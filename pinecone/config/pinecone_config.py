from typing import Optional
import os
from .config import ConfigBuilder, Config

DEFAULT_CONTROLLER_HOST = "https://api.pinecone.io"


class PineconeConfig():
    @staticmethod
    def build(api_key: Optional[str] = None, host: Optional[str] = None, **kwargs) -> Config:
        host = host or kwargs.get("host") or os.getenv("PINECONE_CONTROLLER_HOST") or DEFAULT_CONTROLLER_HOST
        return ConfigBuilder.build(api_key=api_key, host=host, **kwargs)
