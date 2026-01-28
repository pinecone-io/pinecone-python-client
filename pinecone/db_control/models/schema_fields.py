"""Schema field type classes for defining index schemas.

These classes provide a user-friendly API for defining index schemas with typed fields.
Each field class serializes to the format expected by the Pinecone API.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TextField:
    """A text field for storing string values.

    :param filterable: Whether the field can be used in query filters.
    :param full_text_searchable: Whether the field is indexed for full-text search.
    :param description: Optional description of the field.

    Example usage::

        from pinecone import TextField

        schema = {
            "title": TextField(full_text_searchable=True),
            "category": TextField(filterable=True),
        }
    """

    filterable: bool = False
    full_text_searchable: bool = False
    description: str | None = None

    def to_dict(self) -> dict:
        """Serialize to API format.

        :returns: Dictionary representation for the API.
        """
        result: dict = {"type": "string"}
        if self.filterable:
            result["filterable"] = True
        if self.full_text_searchable:
            result["full_text_searchable"] = True
        if self.description is not None:
            result["description"] = self.description
        return result


@dataclass
class IntegerField:
    """An integer field for storing numeric values.

    :param filterable: Whether the field can be used in query filters.
    :param description: Optional description of the field.

    Example usage::

        from pinecone import IntegerField

        schema = {
            "year": IntegerField(filterable=True),
            "count": IntegerField(),
        }
    """

    filterable: bool = False
    description: str | None = None

    def to_dict(self) -> dict:
        """Serialize to API format.

        :returns: Dictionary representation for the API.
        """
        result: dict = {"type": "integer"}
        if self.filterable:
            result["filterable"] = True
        if self.description is not None:
            result["description"] = self.description
        return result


@dataclass
class FloatField:
    """A floating-point field for storing decimal values.

    :param filterable: Whether the field can be used in query filters.
    :param description: Optional description of the field.

    Example usage::

        from pinecone import FloatField

        schema = {
            "price": FloatField(filterable=True),
            "score": FloatField(),
        }
    """

    filterable: bool = False
    description: str | None = None

    def to_dict(self) -> dict:
        """Serialize to API format.

        :returns: Dictionary representation for the API.
        """
        result: dict = {"type": "float"}
        if self.filterable:
            result["filterable"] = True
        if self.description is not None:
            result["description"] = self.description
        return result


@dataclass
class DenseVectorField:
    """A dense vector field for storing vector embeddings.

    :param dimension: The dimension of the vectors (1 to 20000).
    :param metric: The distance metric for similarity search.
        Must be one of: "cosine", "euclidean", "dotproduct".
    :param description: Optional description of the field.

    Example usage::

        from pinecone import DenseVectorField

        schema = {
            "embedding": DenseVectorField(dimension=1536, metric="cosine"),
        }
    """

    dimension: int
    metric: str
    description: str | None = None

    def to_dict(self) -> dict:
        """Serialize to API format.

        :returns: Dictionary representation for the API.
        """
        result: dict = {"type": "dense_vector", "dimension": self.dimension, "metric": self.metric}
        if self.description is not None:
            result["description"] = self.description
        return result


@dataclass
class SparseVectorField:
    """A sparse vector field for storing sparse embeddings.

    :param metric: The distance metric for similarity search.
        Must be "dotproduct" for sparse vectors.
    :param description: Optional description of the field.

    Example usage::

        from pinecone import SparseVectorField

        schema = {
            "sparse_embedding": SparseVectorField(metric="dotproduct"),
        }
    """

    metric: str = "dotproduct"
    description: str | None = None

    def to_dict(self) -> dict:
        """Serialize to API format.

        :returns: Dictionary representation for the API.
        """
        result: dict = {"type": "sparse_vector", "metric": self.metric}
        if self.description is not None:
            result["description"] = self.description
        return result


@dataclass
class SemanticTextField:
    """A semantic text field with integrated inference embedding.

    This field type enables automatic embedding generation using a specified model.
    When documents are upserted, the text in the mapped field is automatically
    converted to vectors.

    :param model: The name of the embedding model to use.
    :param field_map: Maps field names in documents to the field used for embedding.
    :param read_parameters: Optional parameters for the model during queries.
    :param write_parameters: Optional parameters for the model during indexing.
    :param description: Optional description of the field.

    Example usage::

        from pinecone import SemanticTextField

        schema = {
            "content": SemanticTextField(
                model="multilingual-e5-large",
                field_map={"text": "content"},
            ),
        }
    """

    model: str
    field_map: dict[str, str]
    read_parameters: dict[str, object] | None = None
    write_parameters: dict[str, object] | None = None
    description: str | None = None

    def to_dict(self) -> dict:
        """Serialize to API format.

        :returns: Dictionary representation for the API.
        """
        result: dict = {"type": "semantic_text", "model": self.model, "field_map": self.field_map}
        if self.read_parameters is not None:
            result["read_parameters"] = self.read_parameters
        if self.write_parameters is not None:
            result["write_parameters"] = self.write_parameters
        if self.description is not None:
            result["description"] = self.description
        return result


# Type alias for any schema field
SchemaField = (
    TextField | IntegerField | FloatField | DenseVectorField | SparseVectorField | SemanticTextField
)
