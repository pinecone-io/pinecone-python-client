"""PreviewSchemaBuilder for constructing preview index schemas.

Returns a plain ``{"fields": {...}}`` dict (not a model) so forward-compatible
fields the SDK does not yet model can pass through unmodified.
"""

from __future__ import annotations

from typing import Any

_FTS_LANGUAGES_SHORT = frozenset(
    [
        "ar",
        "da",
        "de",
        "el",
        "en",
        "es",
        "fi",
        "fr",
        "hu",
        "it",
        "nl",
        "no",
        "pt",
        "ro",
        "ru",
        "sv",
        "ta",
        "tr",
    ]
)
_FTS_LANGUAGES_LONG_TO_SHORT: dict[str, str] = {
    "arabic": "ar",
    "danish": "da",
    "german": "de",
    "greek": "el",
    "english": "en",
    "spanish": "es",
    "finnish": "fi",
    "french": "fr",
    "hungarian": "hu",
    "italian": "it",
    "dutch": "nl",
    "norwegian": "no",
    "portuguese": "pt",
    "romanian": "ro",
    "russian": "ru",
    "swedish": "sv",
    "tamil": "ta",
    "turkish": "tr",
}


_FIELD_NAME_MAX_BYTES = 64
_DESCRIPTION_MAX_BYTES = 256


def _validate_field_name(name: str) -> None:
    from pinecone.errors.exceptions import PineconeValueError

    if not name:
        raise PineconeValueError("Field name must be a non-empty string")
    if name.startswith("$") or name.startswith("_"):
        raise PineconeValueError(
            f"Field name '{name}' is invalid: names cannot begin with '$' or '_'"
        )
    byte_len = len(name.encode("utf-8"))
    if byte_len > _FIELD_NAME_MAX_BYTES:
        raise PineconeValueError(
            f"Field name '{name}' is too long: {byte_len} bytes (max {_FIELD_NAME_MAX_BYTES})"
        )


def _validate_description(description: str | None) -> None:
    from pinecone.errors.exceptions import PineconeValueError

    if description is None:
        return
    byte_len = len(description.encode("utf-8"))
    if byte_len > _DESCRIPTION_MAX_BYTES:
        raise PineconeValueError(
            f"Description is too long: {byte_len} bytes (max {_DESCRIPTION_MAX_BYTES})"
        )


def _normalize_fts_language(language: str) -> str:
    """Return the canonical short-code form of a language input.

    Accepts ISO short codes (e.g. ``"en"``) and long-form aliases
    (e.g. ``"english"``). Raises PineconeValueError if neither matches.
    """
    from pinecone.errors.exceptions import PineconeValueError

    if language in _FTS_LANGUAGES_SHORT:
        return language
    lowered = language.lower()
    if lowered in _FTS_LANGUAGES_LONG_TO_SHORT:
        return _FTS_LANGUAGES_LONG_TO_SHORT[lowered]
    if lowered in _FTS_LANGUAGES_SHORT:
        return lowered
    raise PineconeValueError(
        f"Invalid language '{language}' provided as language for full_text_search field"
    )


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
        ...     .add_float_field("year", filterable=True)
        ...     .add_boolean_field("is_published", filterable=True)
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
        _validate_field_name(name)
        _validate_description(description)
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

        The wire type is ``"sparse_vector"``. The metric is fixed at
        ``"dotproduct"`` server-side and is not user-configurable.

        Args:
            name: Field name. Replaces any existing field with the same name.
            description: Optional human-readable description.
            **additional_options: Extra parameters merged into the field dict
                last, for forward compatibility with new API features.

        Returns:
            ``self`` for method chaining.
        """
        _validate_field_name(name)
        _validate_description(description)
        field: dict[str, Any] = {
            "type": "sparse_vector",
            "metric": "dotproduct",
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
        full_text_search: bool | dict[str, Any] | None = None,
        language: str | None = None,
        stemming: bool | None = None,
        stop_words: bool | None = None,
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

        Full-text search is enabled by passing ``full_text_search=True``, a
        ``full_text_search`` dict, or any of the typed FTS keyword arguments
        (``language``, ``stemming``, ``stop_words``).  Omit all of these (or
        pass ``full_text_search=None``) to indicate the field is not
        full-text searchable.

        When both ``full_text_search`` dict and keyword arguments are provided,
        the keyword arguments take precedence for the same key.

        ``lowercase`` and ``max_term_len`` are server-managed and cannot be
        configured via the SDK.

        Args:
            name: Field name. Replaces any existing field with the same name.
            full_text_search: ``True`` or ``{}`` to enable FTS with server
                defaults, a ``dict`` of FTS-config keys (``language``,
                ``stemming``, ``stop_words``), or ``None`` (default) to
                leave FTS disabled.  ``lowercase`` and ``max_term_len`` are
                server-managed and not user-configurable.
            language: Language for FTS tokenisation and analysis. Accepts
                ISO short codes or long-form aliases. Both ``"en"`` and
                ``"english"`` are valid; the SDK normalises to the short-code
                form on the wire. Supported codes: ``ar``, ``da``, ``de``,
                ``el``, ``en``, ``es``, ``fi``, ``fr``, ``hu``, ``it``,
                ``nl``, ``no``, ``pt``, ``ro``, ``ru``, ``sv``, ``ta``,
                ``tr`` (and their long-form aliases: ``arabic``, ``danish``,
                ``german``, ``greek``, ``english``, ``spanish``,
                ``finnish``, ``french``, ``hungarian``, ``italian``,
                ``dutch``, ``norwegian``, ``portuguese``, ``romanian``,
                ``russian``, ``swedish``, ``tamil``, ``turkish``).
            stemming: Enable word stemming. Required when ``stop_words=True``.
            stop_words: Enable stop-word filtering. Requires
                ``stemming=True``. Not all languages support stop words;
                the server will reject unsupported combinations — the SDK
                does not pre-validate that rule.
            filterable: Enable metadata-filter support. ``False`` values are
                omitted from the wire payload.
            description: Optional human-readable description.
            **additional_options: Extra parameters merged into the field dict
                last, for forward compatibility with new API features.

        Returns:
            ``self`` for method chaining.

        Raises:
            PineconeValueError: If ``language`` is not one of the 18 supported
                codes (or their long-form aliases), or if ``stop_words=True``
                is requested without ``stemming=True``.

        Examples:
            .. code-block:: python

                # Enable FTS with server defaults:
                builder.add_string_field("title", full_text_search=True)

                # Enable FTS with explicit kwargs:
                builder.add_string_field(
                    "title", language="en", stemming=True, stop_words=True
                )

                # FTS and filterable together:
                builder.add_string_field("title", language="en", filterable=True)
        """
        from pinecone.errors.exceptions import PineconeValueError

        _validate_field_name(name)
        _validate_description(description)
        # Determine whether FTS is enabled by ANY of the inputs.
        fts_kwargs_provided = language is not None or stemming is not None or stop_words is not None
        fts_enabled = (
            full_text_search is True or isinstance(full_text_search, dict) or fts_kwargs_provided
        )

        fts_config: dict[str, Any] = {}
        if isinstance(full_text_search, dict):
            fts_config.update(full_text_search)
        if language is not None:
            fts_config["language"] = _normalize_fts_language(language)
        if stemming is not None:
            fts_config["stemming"] = stemming
        if stop_words is not None:
            fts_config["stop_words"] = stop_words

        # Pre-validate cross-field rule. Run this AFTER merging so we see the
        # final value users intended (whether it came from the dict or a kwarg).
        if fts_config.get("stop_words") is True and fts_config.get("stemming") is not True:
            raise PineconeValueError("stop_words requires stemming to be enabled")

        # If the dict supplied a language string, normalize it too (kwarg path
        # already normalized above; the dict path may not have).
        if "language" in fts_config and isinstance(fts_config["language"], str):
            fts_config["language"] = _normalize_fts_language(fts_config["language"])

        field: dict[str, Any] = {"type": "string"}
        if fts_enabled:
            field["full_text_search"] = fts_config
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
        _validate_field_name(name)
        _validate_description(description)
        field: dict[str, Any] = {"type": "string_list"}
        if filterable:
            field["filterable"] = filterable
        if description is not None:
            field["description"] = description
        field.update(additional_options)
        self._fields[name] = field
        return self

    def add_boolean_field(
        self,
        name: str,
        *,
        filterable: bool = False,
        description: str | None = None,
        **additional_options: Any,
    ) -> PreviewSchemaBuilder:
        """Add a boolean field for metadata filtering.

        .. admonition:: Preview
           :class: warning

           Uses Pinecone API version ``2026-01.alpha``.
           Preview surface is not covered by SemVer — signatures and behavior
           may change in any minor SDK release. Pin your SDK version when
           relying on preview features.

        The wire type is ``"boolean"``. ``filterable=False`` is omitted from
        the wire payload; ``None`` description is omitted as well.

        Args:
            name: Field name. Replaces any existing field with the same name.
            filterable: Enable metadata-filter support on this field.
            description: Optional human-readable description.
            **additional_options: Extra parameters merged into the field dict
                last, for forward compatibility with new API features.

        Returns:
            ``self`` for method chaining.

        Examples:
            .. code-block:: python

                builder.add_boolean_field("is_published", filterable=True)
        """
        _validate_field_name(name)
        _validate_description(description)
        field: dict[str, Any] = {"type": "boolean"}
        if filterable:
            field["filterable"] = filterable
        if description is not None:
            field["description"] = description
        field.update(additional_options)
        self._fields[name] = field
        return self

    def add_float_field(
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

        The wire type is ``"float"``. The Pinecone API does not have a
        separate integer type; integers are stored and filtered as
        double-precision floats.

        Args:
            name: Field name. Replaces any existing field with the same name.
            filterable: Enable filtering on this field. ``False`` is omitted
                from the wire payload.
            description: Optional human-readable description.
            **additional_options: Extra parameters merged into the field dict
                last, for forward compatibility with new API features.

        Returns:
            ``self`` for method chaining.
        """
        _validate_field_name(name)
        _validate_description(description)
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
        _validate_field_name(name)
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
