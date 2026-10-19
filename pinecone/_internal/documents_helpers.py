"""Shared request-building and validation helpers for document operations.

Both the sync :class:`~pinecone.index.Index` and the async twin build their
document request bodies through these helpers, so the two transports emit
byte-identical wire bodies and raise byte-identical validation errors.

Validation and serialization are delegated to the typed request models in
:mod:`pinecone.models.documents.requests`; these helpers adapt their plain
:class:`ValueError` vocabulary into the SDK's :class:`PineconeValueError`
and add the cross-record checks (duplicate ``_id``, per-position error
context) the models cannot see.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any
from urllib.parse import quote

import msgspec

from pinecone.errors.exceptions import PineconeTypeError, ValidationError
from pinecone.models.documents.document import DocumentRecord, UpdateDocumentRecord
from pinecone.models.documents.requests import (
    DeleteDocumentsRequest,
    FetchDocumentsRequest,
    ListDocumentsRequest,
    SearchDocumentsRequest,
    UpdateDocumentsRequest,
    _NotAnIntegerError,
)
from pinecone.models.documents.score_by import DocumentScoringMethod
from pinecone.utils.filter_builder import Condition

_MAX_DOCUMENTS_PER_REQUEST = 1000


def _to_wire_fields(item: dict[str, Any]) -> dict[str, Any]:
    """Return *item* with every SDK struct value (e.g. ``SparseValues``) as a plain dict."""
    return {
        key: msgspec.to_builtins(value) if isinstance(value, msgspec.Struct) else value
        for key, value in item.items()
    }


def _encode_document_namespace(namespace: str) -> str:
    """Validate a documents-operation namespace and URL-encode it as a path segment.

    Document operations address the namespace in the URL path, so unlike the
    vector operations there is no empty-string default — the namespace must be
    a non-empty string, and characters that are not path-safe (``/``, spaces,
    ``%``, non-ASCII) are percent-encoded so they cannot change the route.

    Raises:
        :exc:`PineconeValueError`: If *namespace* is not a non-empty string.
    """
    if not isinstance(namespace, str):
        raise ValidationError("namespace must be a string")
    if not namespace or not namespace.strip():
        raise ValidationError("namespace must be a non-empty string")
    return quote(namespace, safe="")


def _validate_documents(
    documents: Sequence[Mapping[str, Any] | DocumentRecord],
    *,
    max_documents: int | None = _MAX_DOCUMENTS_PER_REQUEST,
) -> list[dict[str, Any]]:
    """Validate documents for upsert and normalize them to wire-shape dicts.

    A document list is accepted if and only if every element is a dict or
    :class:`DocumentRecord` whose ``_id`` is a non-empty ASCII string of at
    most 512 characters (``^[\\x01-\\x7F]+$``), unique within the list.

    Args:
        documents: The documents to validate.
        max_documents: Per-request cap (1000 for a single upsert request).
            ``None`` disables the cap for callers that split into batches.

    Returns:
        The documents as plain dicts, in input order.

    Raises:
        :exc:`PineconeValueError`: If the list is empty or over *max_documents*,
            or any document violates the ``_id`` contract — the message names
            the position of the offending document.
    """
    docs = list(documents)
    if not docs:
        raise ValidationError("documents must be a non-empty list")
    if max_documents is not None and len(docs) > max_documents:
        raise ValidationError(
            f"Number of documents ({len(docs)}) exceeds the maximum limit of "
            f"{max_documents} documents per request. Use index.documents.batch_upsert "
            "to split a larger list into multiple requests."
        )
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for position, doc in enumerate(docs):
        if isinstance(doc, DocumentRecord):
            item = doc.to_dict()
        elif isinstance(doc, Mapping):
            try:
                item = DocumentRecord(dict(doc)).to_dict()
            except ValueError as exc:
                raise ValidationError(f"Document at position {position}: {exc}") from exc
        else:
            raise ValidationError(
                f"Document at position {position} must be a dict or DocumentRecord, "
                f"got {type(doc).__name__}"
            )
        doc_id: str = item["_id"]
        if doc_id in seen:
            raise ValidationError(
                f"Document at position {position} has duplicate '_id' {doc_id!r}; "
                "document IDs must be unique within a request."
            )
        seen.add(doc_id)
        normalized.append(_to_wire_fields(item))
    return normalized


def _validate_update_documents(
    documents: Sequence[Mapping[str, Any] | UpdateDocumentRecord],
) -> list[dict[str, Any]]:
    """Validate per-ID document patches and normalize them to wire-shape dicts.

    A patch list is accepted if and only if every element is a dict or
    :class:`UpdateDocumentRecord` whose ``_id`` satisfies the same contract
    :func:`_validate_documents` enforces, whose ``_remove_fields`` (when
    present) is a list of strings naming no field the same patch also sets,
    and whose ``_id`` is unique within the list.

    Duplicate ``_id`` values are rejected rather than sent: the server
    applies the patches in a request independently, so two patches to one
    document would resolve in an unspecified order. This mirrors the
    duplicate rule ``upsert_documents`` already applies.

    Returns:
        The patches as plain dicts, in input order.

    Raises:
        :exc:`PineconeValueError`: If the list is empty or over 1000 entries,
            or any patch is malformed — the message names the position of the
            offending patch.
    """
    docs = list(documents)
    if not docs:
        raise ValidationError("documents must be a non-empty list")
    if len(docs) > _MAX_DOCUMENTS_PER_REQUEST:
        raise ValidationError(
            f"Number of documents ({len(docs)}) exceeds the maximum limit of "
            f"{_MAX_DOCUMENTS_PER_REQUEST} documents per request."
        )
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for position, doc in enumerate(docs):
        if isinstance(doc, UpdateDocumentRecord):
            item = doc.to_dict()
        elif isinstance(doc, Mapping):
            try:
                item = UpdateDocumentRecord(dict(doc)).to_dict()
            except ValueError as exc:
                raise ValidationError(f"Document at position {position}: {exc}") from exc
        else:
            raise ValidationError(
                f"Document at position {position} must be a dict or UpdateDocumentRecord, "
                f"got {type(doc).__name__}"
            )
        doc_id: str = item["_id"]
        if doc_id in seen:
            raise ValidationError(
                f"Document at position {position} has duplicate '_id' {doc_id!r}; "
                "document IDs must be unique within a request."
            )
        seen.add(doc_id)
        normalized.append(item)
    return normalized


def _check_filter(filter: object) -> None:
    """Reject a ``filter`` that is not a mapping, naming ``.to_dict()`` for a ``Condition``."""
    if filter is None or isinstance(filter, Mapping):
        return
    hint = (
        "call .to_dict() on a filter-builder Condition to get one"
        if isinstance(filter, Condition)
        else "pass a dict such as {'field': {'$eq': 'value'}}"
    )
    raise PineconeTypeError(f"filter must be a dict, got {type(filter).__name__}; {hint}")


def _check_score_by(score_by: object) -> None:
    """Reject a ``score_by`` that is a lone clause, a mapping, or a string."""
    if isinstance(score_by, (str, bytes, Mapping)) or not isinstance(score_by, Iterable):
        raise PineconeTypeError(
            f"score_by must be a list of scoring clauses, got {type(score_by).__name__}; "
            "wrap a single clause in a list, as in score_by=[clause]"
        )


def _names(name: str, value: Sequence[str] | None) -> list[str] | None:
    """Return ``value`` as a list, rejecting a bare ``str`` that would split into characters."""
    if value is None:
        return None
    if isinstance(value, str):
        raise PineconeTypeError(
            f"{name} must be a sequence of strings, not a single str; "
            f"pass [{value!r}] to name one entry."
        )
    return list(value)


def _build_search_documents_body(
    *,
    score_by: Sequence[DocumentScoringMethod | Mapping[str, Any]],
    top_k: int,
    include_fields: Sequence[str] | None,
    filter: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build and validate a ``search_documents`` request body.

    Raises:
        :exc:`PineconeTypeError`: If ``top_k`` is not an integer (a bool is not
            accepted).
        :exc:`PineconeValueError`: If ``score_by`` is empty, over 100 clauses,
            combines a ``dense_vector``/``sparse_vector`` clause with any other
            clause, or contains an invalid clause; or if ``top_k`` is outside
            [1, 10000].
        :exc:`PineconeTypeError`: If ``score_by`` is a single clause, mapping, or string
            rather than a list, holds an item that is neither a dict nor one of the
            six clause types, or ``filter`` is not a dict.
    """
    _check_score_by(score_by)
    _check_filter(filter)
    try:
        request = SearchDocumentsRequest(
            score_by=[dict(item) if isinstance(item, Mapping) else item for item in score_by],
            top_k=top_k,
            include_fields=_names("include_fields", include_fields),
            filter=dict(filter) if filter is not None else None,
        )
    except _NotAnIntegerError as exc:
        raise PineconeTypeError(str(exc)) from exc
    except (ValueError, msgspec.ValidationError) as exc:
        raise ValidationError(str(exc)) from exc
    body: dict[str, Any] = msgspec.to_builtins(request)
    return body


def _build_fetch_documents_body(
    *,
    ids: Sequence[str] | None,
    filter: Mapping[str, Any] | None,
    include_fields: Sequence[str] | None,
    pagination_token: str | None,
    limit: int | None,
) -> dict[str, Any]:
    """Build and validate a ``fetch_documents`` request body.

    Raises:
        :exc:`PineconeTypeError`: If ``limit`` is not an integer (a bool is not
            accepted).
        :exc:`PineconeValueError`: If both or neither of ``ids`` and ``filter``
            are provided, ``filter`` is empty, ``ids`` exceeds 1000 entries,
            ``pagination_token`` is passed without ``filter``, or ``limit``
            falls outside 1-10000.
    """
    _check_filter(filter)
    try:
        request = FetchDocumentsRequest(
            ids=_names("ids", ids),
            filter=dict(filter) if filter is not None else None,
            include_fields=_names("include_fields", include_fields),
            pagination_token=pagination_token,
            limit=limit,
        )
    except _NotAnIntegerError as exc:
        raise PineconeTypeError(str(exc)) from exc
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    body: dict[str, Any] = msgspec.to_builtins(request)
    return body


def _build_delete_documents_body(
    *,
    ids: Sequence[str] | None,
    filter: Mapping[str, Any] | None,
    delete_all: bool,
) -> dict[str, Any]:
    """Build and validate a ``delete_documents`` request body.

    Raises:
        :exc:`PineconeValueError`: If zero or more than one of ``ids``,
            ``filter``, and ``delete_all`` are provided, ``filter`` is empty,
            or ``ids`` exceeds 1000 entries.
        :exc:`PineconeTypeError`: If ``ids`` is a bare ``str``.
    """
    _check_filter(filter)
    try:
        request = DeleteDocumentsRequest(
            ids=_names("ids", ids),
            filter=dict(filter) if filter is not None else None,
            delete_all=True if delete_all else None,
        )
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    body: dict[str, Any] = msgspec.to_builtins(request)
    return body


def _build_update_documents_body(
    *,
    documents: Sequence[Mapping[str, Any] | UpdateDocumentRecord] | None,
    filter: Mapping[str, Any] | None,
    set_fields: Mapping[str, Any] | None,
    remove_fields: Sequence[str] | None,
) -> dict[str, Any]:
    """Build and validate an ``update_documents`` request body.

    Per-ID patches go through :func:`_validate_update_documents` first, so a
    malformed patch is reported with its position before the envelope's
    selector rules are applied.

    Raises:
        :exc:`PineconeValueError`: If ``documents`` is combined with any of
            the by-filter fields, neither ``documents`` nor ``filter`` is
            provided, ``set_fields``/``remove_fields`` are passed without
            ``filter``, ``filter`` is empty or carries no patch, or any
            per-ID patch is malformed.
    """
    _check_filter(filter)
    normalized = _validate_update_documents(documents) if documents is not None else None
    try:
        request = UpdateDocumentsRequest(
            documents=list(normalized) if normalized is not None else None,
            filter=dict(filter) if filter is not None else None,
            set_fields=dict(set_fields) if set_fields is not None else None,
            remove_fields=_names("remove_fields", remove_fields),
        )
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    body: dict[str, Any] = msgspec.to_builtins(request)
    return body


def _build_list_documents_body(
    *,
    prefix: str | None,
    limit: int | None,
    pagination_token: str | None,
) -> dict[str, Any]:
    """Build and validate a ``list_documents`` request body.

    Raises:
        :exc:`PineconeTypeError`: If ``limit`` is not an integer (a bool is not
            accepted).
        :exc:`PineconeValueError`: If ``prefix`` is over 512 characters or
            contains a character outside ``\\x01``-``\\x7F``, or ``limit``
            falls outside 1-100.
    """
    try:
        request = ListDocumentsRequest(
            prefix=prefix,
            limit=limit,
            pagination_token=pagination_token,
        )
    except _NotAnIntegerError as exc:
        raise PineconeTypeError(str(exc)) from exc
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    body: dict[str, Any] = msgspec.to_builtins(request)
    return body
