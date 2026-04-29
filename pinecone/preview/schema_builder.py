"""PreviewSchemaBuilder for constructing preview index schemas.

Returns a plain ``{"fields": {...}}`` dict (not a model) so forward-compatible
fields the SDK does not yet model can pass through unmodified.
"""

from __future__ import annotations

from typing import Any


class PreviewSchemaBuilder:
    """Fluent builder for preview index schema dicts.

    .. admonition:: Preview
       :class: warning

       Uses Pinecone API version ``2026-01.alpha``.
       Preview surface is not covered by SemVer — signatures and behavior
       may change in any minor SDK release. Pin your SDK version when
       relying on preview features.

    Each ``add_*`` method appends or replaces a field definition and returns
    ``self`` so calls can be chained.  Call :meth:`build` at the end to
    obtain the ``{"fields": {...}}`` dict.

    Adding a field whose name already exists silently replaces the previous
    definition (last writer wins).

    Examples:
        >>> from pinecone.preview import PreviewSchemaBuilder
        >>> schema = (
        ...     PreviewSchemaBuilder()
        ...     .add_dense_vector_field("embedding", dimension=768, metric="cosine")
        ...     .add_string_field("title", full_text_search={"language": "en"})
        ...     .add_string_field("category", filterable=True)
        ...     .add_string_list_field("tags", filterable=True)
        ...     .add_integer_field("year", filterable=True)
        ...     .build()
        ... )
    """

    def __init__(self) -> None:
        self._fields: dict[str, dict[str, Any]] = {}

    def add_dense_vector_field(
        self,
        name: str,
        *,
        dimension: int,
        metric: str,
        description: str | None = None,
        **additional_options: Any,
    ) -> PreviewSchemaBuilder:
        """Add a dense vector field for similarity search.

        .. admonition:: Preview
           :class: warning

           Uses Pinecone API version ``2026-01.alpha``.
           Preview surface is not covered by SemVer — signatures and behavior
           may change in any minor SDK release. Pin your SDK version when
           relying on preview features.

        Args:
            name: Field name. Replaces any existing field with the same name.
            dimension: Vector dimensionality (1–20 000).
            metric: Distance metric — ``"cosine"``, ``"euclidean"``, or
                ``"dotproduct"``.
            description: Optional human-readable description.
            **additional_options: Extra parameters merged into the field dict
                last, for forward compatibility with new API features.

        Returns:
            ``self`` for method chaining.
        """
        field: dict[str, Any] = {
            "type": "dense_vector",
            "dimension": dimension,
            "metric": metric,
        }
        if description is not None:
            field["description"] = description
        field.update(additional_options)
        self._fields[name] = field
        return self

    def add_sparse_vector_field(
        self,
        name: str,
        *,
        metric: str = "dotproduct",
        description: str | None = None,
        **additional_options: Any,
    ) -> PreviewSchemaBuilder:
        """Add a sparse vector field for keyword-weighted or learned-sparse search.

        .. admonition:: Preview
           :class: warning

           Uses Pinecone API version ``2026-01.alpha``.
           Preview surface is not covered by SemVer — signatures and behavior
           may change in any minor SDK release. Pin your SDK version when
           relying on preview features.

        Args:
            name: Field name. Replaces any existing field with the same name.
            metric: Distance metric. Defaults to ``"dotproduct"``.
            description: Optional human-readable description.
            **additional_options: Extra parameters merged into the field dict
                last, for forward compatibility with new API features.

        Returns:
            ``self`` for method chaining.
        """
        field: dict[str, Any] = {
            "type": "sparse_vector",
            "metric": metric,
        }
        if description is not None:
            field["description"] = description
        field.update(additional_options)
        self._fields[name] = field
        return self

    def add_string_field(
        self,
        name: str,
        *,
        full_text_search: dict[str, Any] | None = None,
        filterable: bool = False,
        description: str | None = None,
        **additional_options: Any,
    ) -> PreviewSchemaBuilder:
        """Add a string field for full-text search, metadata filtering, or both.

        .. admonition:: Preview
           :class: warning

           Uses Pinecone API version ``2026-01.alpha``.
           Preview surface is not covered by SemVer — signatures and behavior
           may change in any minor SDK release. Pin your SDK version when
           relying on preview features.

        Full-text search is enabled by passing a ``full_text_search`` dict —
        even an empty dict ``{}`` is a valid value and requests server
        defaults for all options. Omit (or pass ``None``) to indicate the
        field is not full-text searchable.

        Args:
            name: Field name. Replaces any existing field with the same name.
            full_text_search: Full-text search configuration. Pass ``{}`` for
                server defaults, or a dict with any of ``language``,
                ``stemming``, ``lowercase``, ``max_term_len``, ``stop_words``.
                ``None`` (default) means the field is not full-text
                searchable.
            filterable: Enable metadata-filter support. ``False`` values are
                omitted from the wire payload.
            description: Optional human-readable description.
            **additional_options: Extra parameters merged into the field dict
                last, for forward compatibility with new API features.

        Returns:
            ``self`` for method chaining.

        Examples:
            .. code-block:: python

                builder.add_string_field("title", full_text_search={})

            Enable FTS with an English language config::

                builder.add_string_field("title", full_text_search={"language": "en"})

            Filterable metadata field without FTS::

                builder.add_string_field("category", filterable=True)
        """
        field: dict[str, Any] = {"type": "string"}
        if full_text_search is not None:
            field["full_text_search"] = dict(full_text_search)
        if filterable:
            field["filterable"] = filterable
        if description is not None:
            field["description"] = description
        field.update(additional_options)
        self._fields[name] = field
        return self

    def add_string_list_field(
        self,
        name: str,
        *,
        filterable: bool = False,
        description: str | None = None,
        **additional_options: Any,
    ) -> PreviewSchemaBuilder:
        """Add a list-of-strings field for metadata filtering.

        .. admonition:: Preview
           :class: warning

           Uses Pinecone API version ``2026-01.alpha``.
           Preview surface is not covered by SemVer — signatures and behavior
           may change in any minor SDK release. Pin your SDK version when
           relying on preview features.

        String-list fields store a list of strings per row — useful for
        tag-style metadata (e.g. ``["sci-fi", "mystery"]``) that should be
        filterable against individual elements.

        The wire type is ``"string_list"``. ``filterable=False`` is omitted
        from the wire payload; ``None`` values are omitted as well.

        Args:
            name: Field name. Replaces any existing field with the same name.
            filterable: Enable metadata-filter support.
            description: Optional human-readable description.
            **additional_options: Extra parameters merged into the field dict
                last, for forward compatibility with new API features.

        Returns:
            ``self`` for method chaining.

        Examples:
            .. code-block:: python

                builder.add_string_list_field("tags", filterable=True)
        """
        field: dict[str, Any] = {"type": "string_list"}
        if filterable:
            field["filterable"] = filterable
        if description is not None:
            field["description"] = description
        field.update(additional_options)
        self._fields[name] = field
        return self

    def add_semantic_text_field(
        self,
        name: str,
        *,
        model: str,
        metric: str = "cosine",
        read_parameters: dict[str, Any] | None = None,
        write_parameters: dict[str, Any] | None = None,
        description: str | None = None,
        **additional_options: Any,
    ) -> PreviewSchemaBuilder:
        """Add a field with server-side embedding (integrated model).

        .. admonition:: Preview
           :class: warning

           Uses Pinecone API version ``2026-01.alpha``.
           Preview surface is not covered by SemVer — signatures and behavior
           may change in any minor SDK release. Pin your SDK version when
           relying on preview features.

        The server generates embeddings at write time and uses them at query
        time, so callers do not need to manage embeddings separately.

        Args:
            name: Field name. Replaces any existing field with the same name.
            model: Embedding model name (e.g. ``"multilingual-e5-large"``).
            metric: Distance metric for the generated embeddings. Defaults to
                ``"cosine"``.
            read_parameters: Query-time model parameters.
            write_parameters: Index-time model parameters.
            description: Optional human-readable description.
            **additional_options: Extra parameters merged into the field dict
                last, for forward compatibility with new API features.

        Returns:
            ``self`` for method chaining.
        """
        field: dict[str, Any] = {
            "type": "semantic_text",
            "model": model,
            "metric": metric,
        }
        if read_parameters is not None:
            field["read_parameters"] = read_parameters
        if write_parameters is not None:
            field["write_parameters"] = write_parameters
        if description is not None:
            field["description"] = description
        field.update(additional_options)
        self._fields[name] = field
        return self

    def add_integer_field(
        self,
        name: str,
        *,
        filterable: bool = False,
        description: str | None = None,
        **additional_options: Any,
    ) -> PreviewSchemaBuilder:
        """Add a numeric field for metadata filtering.

        .. admonition:: Preview
           :class: warning

           Uses Pinecone API version ``2026-01.alpha``.
           Preview surface is not covered by SemVer — signatures and behavior
           may change in any minor SDK release. Pin your SDK version when
           relying on preview features.

        Note:
            The wire type is ``"float"`` — the API normalises both
            ``"number"`` and ``"float"`` to ``"float"`` in responses.

        Args:
            name: Field name. Replaces any existing field with the same name.
            filterable: Enable filtering on this field. Defaults to ``False``.
            description: Optional human-readable description.
            **additional_options: Extra parameters merged into the field dict
                last, for forward compatibility with new API features.

        Returns:
            ``self`` for method chaining.
        """
        field: dict[str, Any] = {"type": "float"}
        if filterable:
            field["filterable"] = filterable
        if description is not None:
            field["description"] = description
        field.update(additional_options)
        self._fields[name] = field
        return self

    def add_custom_field(
        self,
        name: str,
        field_definition: dict[str, Any],
    ) -> PreviewSchemaBuilder:
        """Escape hatch — store a raw field dict verbatim.

        .. admonition:: Preview
           :class: warning

           Uses Pinecone API version ``2026-01.alpha``.
           Preview surface is not covered by SemVer — signatures and behavior
           may change in any minor SDK release. Pin your SDK version when
           relying on preview features.

        Use when you need a field type the SDK does not yet model, or when
        experimenting with new API features before the SDK adds support.

        Args:
            name: Field name. Replaces any existing field with the same name.
            field_definition: Complete field definition dict; stored as-is.

        Returns:
            ``self`` for method chaining.
        """
        self._fields[name] = field_definition
        return self

    def build(self) -> dict[str, dict[str, Any]]:
        """Return the completed schema dict.

        .. admonition:: Preview
           :class: warning

           Uses Pinecone API version ``2026-01.alpha``.
           Preview surface is not covered by SemVer — signatures and behavior
           may change in any minor SDK release. Pin your SDK version when
           relying on preview features.

        Returns a copy of the internal field dict so that subsequent
        ``add_*`` calls do not mutate a previously built result.

        Returns:
            ``{"fields": {name: field_dict, ...}}`` ready to pass to
            ``pc.preview.indexes.create(schema=...)``.
        """
        return {"fields": dict(self._fields)}


__all__ = ["PreviewSchemaBuilder"]
