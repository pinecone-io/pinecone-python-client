"""SchemaBuilder fluent API for building index schemas.

This module provides a builder pattern for constructing index schemas
with a fluent, chainable API.
"""

from __future__ import annotations

from .schema_fields import (
    TextField,
    IntegerField,
    FloatField,
    DenseVectorField,
    SparseVectorField,
    SemanticTextField,
    SchemaField,
)


class SchemaBuilder:
    """A fluent builder for constructing index schemas.

    The SchemaBuilder provides a chainable API for defining index schemas
    with typed fields. Each method returns the builder instance, allowing
    for method chaining.

    Example usage::

        from pinecone import SchemaBuilder

        schema = (SchemaBuilder()
            .text("title", full_text_searchable=True)
            .integer("year", filterable=True)
            .dense_vector("embedding", dimension=1536, metric="cosine")
            .build())

        pc.create_index(name="my-index", schema=schema, ...)
    """

    def __init__(self) -> None:
        """Initialize an empty schema builder."""
        self._fields: dict[str, SchemaField] = {}

    def text(
        self,
        name: str,
        *,
        filterable: bool = False,
        full_text_searchable: bool = False,
        description: str | None = None,
    ) -> SchemaBuilder:
        """Add a text field to the schema.

        :param name: The field name.
        :param filterable: Whether the field can be used in query filters.
        :param full_text_searchable: Whether the field is indexed for full-text search.
        :param description: Optional description of the field.
        :returns: The builder instance for chaining.

        Example::

            schema = (SchemaBuilder()
                .text("title", full_text_searchable=True)
                .text("category", filterable=True)
                .build())
        """
        self._fields[name] = TextField(
            filterable=filterable,
            full_text_searchable=full_text_searchable,
            description=description,
        )
        return self

    def integer(
        self, name: str, *, filterable: bool = False, description: str | None = None
    ) -> SchemaBuilder:
        """Add an integer field to the schema.

        :param name: The field name.
        :param filterable: Whether the field can be used in query filters.
        :param description: Optional description of the field.
        :returns: The builder instance for chaining.

        Example::

            schema = (SchemaBuilder()
                .integer("year", filterable=True)
                .integer("count")
                .build())
        """
        self._fields[name] = IntegerField(filterable=filterable, description=description)
        return self

    def float(
        self, name: str, *, filterable: bool = False, description: str | None = None
    ) -> SchemaBuilder:
        """Add a float field to the schema.

        :param name: The field name.
        :param filterable: Whether the field can be used in query filters.
        :param description: Optional description of the field.
        :returns: The builder instance for chaining.

        Example::

            schema = (SchemaBuilder()
                .float("price", filterable=True)
                .float("score")
                .build())
        """
        self._fields[name] = FloatField(filterable=filterable, description=description)
        return self

    def dense_vector(
        self, name: str, *, dimension: int, metric: str, description: str | None = None
    ) -> SchemaBuilder:
        """Add a dense vector field to the schema.

        :param name: The field name.
        :param dimension: The dimension of the vectors (1 to 20000).
        :param metric: The distance metric ("cosine", "euclidean", or "dotproduct").
        :param description: Optional description of the field.
        :returns: The builder instance for chaining.

        Example::

            schema = (SchemaBuilder()
                .dense_vector("embedding", dimension=1536, metric="cosine")
                .build())
        """
        self._fields[name] = DenseVectorField(
            dimension=dimension, metric=metric, description=description
        )
        return self

    def sparse_vector(
        self, name: str, *, metric: str = "dotproduct", description: str | None = None
    ) -> SchemaBuilder:
        """Add a sparse vector field to the schema.

        :param name: The field name.
        :param metric: The distance metric (must be "dotproduct" for sparse vectors).
        :param description: Optional description of the field.
        :returns: The builder instance for chaining.

        Example::

            schema = (SchemaBuilder()
                .sparse_vector("sparse_embedding")
                .build())
        """
        self._fields[name] = SparseVectorField(metric=metric, description=description)
        return self

    def semantic_text(
        self,
        name: str,
        *,
        model: str,
        field_map: dict[str, str],
        read_parameters: dict[str, object] | None = None,
        write_parameters: dict[str, object] | None = None,
        description: str | None = None,
    ) -> SchemaBuilder:
        """Add a semantic text field with integrated inference.

        :param name: The field name.
        :param model: The name of the embedding model to use.
        :param field_map: Maps field names in documents to the field used for embedding.
        :param read_parameters: Optional parameters for the model during queries.
        :param write_parameters: Optional parameters for the model during indexing.
        :param description: Optional description of the field.
        :returns: The builder instance for chaining.

        Example::

            schema = (SchemaBuilder()
                .semantic_text(
                    "content",
                    model="multilingual-e5-large",
                    field_map={"text": "content"},
                )
                .build())
        """
        self._fields[name] = SemanticTextField(
            model=model,
            field_map=field_map,
            read_parameters=read_parameters,
            write_parameters=write_parameters,
            description=description,
        )
        return self

    def build(self) -> dict[str, dict]:
        """Build and return the final schema dictionary.

        :returns: A dictionary mapping field names to their serialized configurations.
        :raises ValueError: If no fields have been added to the builder.

        Example::

            schema = (SchemaBuilder()
                .text("title", full_text_searchable=True)
                .dense_vector("embedding", dimension=1536, metric="cosine")
                .build())

            # Returns:
            # {
            #     "title": {"type": "string", "full_text_searchable": True},
            #     "embedding": {"type": "dense_vector", "dimension": 1536, "metric": "cosine"}
            # }
        """
        if not self._fields:
            raise ValueError("Cannot build empty schema. Add at least one field.")
        return {name: field.to_dict() for name, field in self._fields.items()}
