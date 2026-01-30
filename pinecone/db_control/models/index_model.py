"""IndexModel wrapper with backward compatibility shims.

This module provides the IndexModel class that wraps the generated OpenAPI
IndexModel and provides backward compatibility for code written against the
old API structure.
"""

from __future__ import annotations

import json
from typing import Literal, TYPE_CHECKING

from pinecone.core.openapi.db_control.model.index_model import IndexModel as OpenAPIIndexModel
from pinecone.utils.repr_overrides import custom_serializer

from .compatibility_spec import CompatibilitySpec

if TYPE_CHECKING:
    pass


class IndexModel:
    """Wrapper for OpenAPI IndexModel that provides compatibility shims.

    This class wraps the generated OpenAPI IndexModel and provides backward
    compatibility properties for code that was written against the old API
    structure (using ``spec``, ``dimension``, ``metric`` at the top level).

    The new alpha API uses ``deployment`` instead of ``spec``, and stores
    dimension/metric inside ``schema.fields``. This wrapper provides both
    access patterns transparently.

    :param index: The underlying OpenAPI IndexModel instance.

    Example usage::

        # New alpha API format with schema and deployment
        index = pc.describe_index("my-index")

        # Old-style access still works via compatibility shims
        print(index.spec.serverless.cloud)  # Via CompatibilitySpec
        print(index.dimension)              # Extracted from schema.fields
        print(index.metric)                 # Extracted from schema.fields

        # New-style access also works
        print(index.deployment.cloud)
        print(index.schema.fields)
    """

    def __init__(self, index: OpenAPIIndexModel):
        self.index = index
        self._spec_cache: CompatibilitySpec | None = None
        self._vector_info_cache: (
            tuple[int | None, str | None, Literal["dense", "sparse"] | None] | None
        ) = None

    def __str__(self) -> str:
        return str(self.index)

    def __getattr__(self, attr: str) -> object:
        if attr == "spec":
            return self._get_spec()
        if attr == "dimension":
            return self._get_dimension()
        if attr == "metric":
            return self._get_metric()
        if attr == "vector_type":
            return self._get_vector_type()
        return getattr(self.index, attr)

    def _get_spec(self) -> CompatibilitySpec | None:
        """Get spec with backward compatibility from deployment data.

        Builds a CompatibilitySpec from the deployment data, providing the old
        ``.spec.serverless`` / ``.spec.pod`` / ``.spec.byoc`` access patterns.

        :returns: CompatibilitySpec wrapper or None if no deployment.
        """
        if self._spec_cache is not None:
            return self._spec_cache

        deployment = self.index._data_store.get("deployment")
        if deployment is not None:
            self._spec_cache = CompatibilitySpec(deployment)
            return self._spec_cache

        return None

    def _get_vector_field_info(
        self,
    ) -> tuple[int | None, str | None, Literal["dense", "sparse"] | None]:
        """Extract dimension, metric, and vector_type from schema fields.

        Searches through the schema fields to find a vector field (dense_vector,
        sparse_vector, or semantic_text) and extracts its properties.

        :returns: Tuple of (dimension, metric, vector_type). All values may be None
            for FTS-only indexes.
        """
        if self._vector_info_cache is not None:
            return self._vector_info_cache

        schema = self.index._data_store.get("schema")
        if schema is None:
            result: tuple[int | None, str | None, Literal["dense", "sparse"] | None] = (
                None,
                None,
                None,
            )
            self._vector_info_cache = result
            return result

        fields = getattr(schema, "fields", None)
        if fields is None:
            result = (None, None, None)
            self._vector_info_cache = result
            return result

        # Look for vector fields in order of precedence
        for field_config in fields.values():
            field_type = getattr(field_config, "type", None)

            if field_type == "dense_vector":
                dimension: int | None = getattr(field_config, "dimension", None)
                metric: str | None = getattr(field_config, "metric", None)
                result = (dimension, metric, "dense")
                self._vector_info_cache = result
                return result
            elif field_type == "sparse_vector":
                metric = getattr(field_config, "metric", None)
                result = (None, metric, "sparse")
                self._vector_info_cache = result
                return result
            elif field_type == "semantic_text":
                metric = getattr(field_config, "metric", None)
                result = (None, metric, "dense")
                self._vector_info_cache = result
                return result

        # No vector fields found (FTS-only index)
        result = (None, None, None)
        self._vector_info_cache = result
        return result

    def _get_dimension(self) -> int | None:
        """Get the dimension of the index's vector field.

        Extracts dimension from schema.fields for dense vector fields.

        :returns: The dimension if a dense vector field exists, None otherwise.
        """
        dimension, _, _ = self._get_vector_field_info()
        return dimension

    def _get_metric(self) -> str | None:
        """Get the metric of the index's vector field.

        Extracts metric from schema.fields for vector fields.

        :returns: The metric if a vector field exists, None otherwise.
        """
        _, metric, _ = self._get_vector_field_info()
        return metric

    def _get_vector_type(self) -> Literal["dense", "sparse"] | None:
        """Get the vector type of the index.

        Derived from schema.fields based on the vector field type.

        :returns: "dense" for dense vectors, "sparse" for sparse vectors, None for FTS-only.
        """
        _, _, vector_type = self._get_vector_field_info()
        return vector_type

    def __getitem__(self, key: str) -> object:
        return self.__getattr__(key)

    def __repr__(self) -> str:
        return json.dumps(self.to_dict(), indent=4, default=custom_serializer)

    def to_dict(self) -> dict[str, object]:
        """Convert the IndexModel to a dictionary.

        :returns: Dictionary representation of the index.
        """
        result: dict[str, object] = self.index.to_dict()
        return result
