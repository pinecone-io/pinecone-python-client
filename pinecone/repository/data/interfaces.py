from abc import ABC, abstractmethod
from typing import Union, Dict, Any

from pinecone.core.openapi.ckb_knowledge_data.models import (
    DocumentForUpsert,
    UpsertDocumentResponse,
)


class RepositoryInterface(ABC):
    @abstractmethod
    def upsert(
        self, namespace: str, document: Union[Dict[str, Any], DocumentForUpsert], **kwargs
    ) -> UpsertDocumentResponse:
        """
        Upserts a document into a Pinecone Repository.

        Returns:
            `UpsertDocumentResponse`, includes the number of vectors upserted.
        """
        pass
