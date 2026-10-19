"""The request bodies behind the document operations, one per verb.

You rarely build these yourself — the methods on ``index.documents`` assemble them from
their arguments. They are worth reading when you want to know exactly which combinations
of arguments are legal, because each one checks its selectors and sizes on construction,
so an illegal combination is rejected at the call site rather than by the server.

An unset optional field is left off the request entirely rather than sent as null, which
is why "omitted" and "explicitly empty" can mean different things below.
"""

from __future__ import annotations

from typing import Any, get_args

import msgspec
from msgspec import Struct

from pinecone.errors.exceptions import PineconeTypeError
from pinecone.models.documents.document import DocumentRecord, UpdateDocumentRecord
from pinecone.models.documents.score_by import (
    DenseVectorQuery,
    DocumentScoringMethod,
    EmbedQuery,
    SparseEmbedQuery,
    SparseVectorQuery,
)
from pinecone.models.vectors.search import SearchQuery

__all__ = [
    "DeleteDocumentsRequest",
    "FetchDocumentsRequest",
    "ListDocumentsRequest",
    "SearchDocumentsRequest",
    "UpdateDocumentsRequest",
    "UpsertDocumentsRequest",
]

_MAX_DOCUMENTS_PER_REQUEST = 1000
_MAX_IDS_PER_REQUEST = 1000
_MAX_SCORING_METHODS = 100
_MAX_TOP_K = 10000
_CLAUSE_TYPES = ", ".join(t.__name__ for t in get_args(DocumentScoringMethod))


def _check_score_by_item(index: int, item: object) -> None:
    if isinstance(item, (dict, *get_args(DocumentScoringMethod))):
        return
    hint = (
        " SearchQuery belongs to the legacy records API; use TextQuery for full-text search "
        "or EmbedQuery for a field with integrated embedding."
        if isinstance(item, SearchQuery)
        else " Import the clause classes from pinecone, for example "
        "TextQuery(fields=['body'], query='...')."
    )
    raise PineconeTypeError(
        f"'score_by[{index}]' must be a dict or one of {_CLAUSE_TYPES}, "
        f"got {type(item).__name__}.{hint}"
    )


_MAX_FETCH_LIMIT = 10000
_MAX_LIST_LIMIT = 100
_MAX_PREFIX_LENGTH = 512


class _NotAnIntegerError(TypeError):
    """``top_k`` or ``limit`` was not an ``int``; re-raised as ``PineconeTypeError``."""


def _require_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _NotAnIntegerError(
            f"'{name}' must be an integer, not {type(value).__name__} ({value!r}); "
            "booleans are not accepted."
        )


_STANDALONE_CLAUSES = (DenseVectorQuery, SparseVectorQuery, EmbedQuery, SparseEmbedQuery)

_FILTER_EMPTY_ERR = (
    "The 'filter' field must be a non-empty object of filter predicates, "
    'for example {"category": {"$eq": "news"}}. '
    "Remove 'filter' to select documents by ID instead."
)


def _validate_filter_not_empty(filter_: dict[str, Any] | None) -> None:
    if filter_ is not None and len(filter_) == 0:
        raise ValueError(_FILTER_EMPTY_ERR)


def _validate_id_count(count: int) -> None:
    if count > _MAX_IDS_PER_REQUEST:
        raise ValueError(
            f"Number of ids ({count}) exceeds the maximum limit of "
            f"{_MAX_IDS_PER_REQUEST} ids per request"
        )


class UpsertDocumentsRequest(Struct, kw_only=True, omit_defaults=True):
    """The body of an ``upsert`` on ``index.documents``.

    Attributes:
        documents: The documents to write, 1 to 1000 of them. Each may be a
            :class:`~pinecone.models.documents.document.DocumentRecord` or a plain dict
            carrying the reserved ``_id`` key alongside your own fields; either way the
            ``_id`` is validated here, so a bad one is reported before anything is sent.
            An upsert replaces the whole document, so a field you leave out of a document
            you are rewriting does not survive.

    Raises:
        ValueError: If ``documents`` is empty, holds more than 1000 documents, or contains
            a document whose ``_id`` is missing or invalid.
    """

    documents: list[dict[str, Any] | DocumentRecord]

    def __post_init__(self) -> None:
        if not self.documents:
            raise ValueError("'documents' must contain at least one document to upsert.")
        if len(self.documents) > _MAX_DOCUMENTS_PER_REQUEST:
            raise ValueError(
                f"Number of documents ({len(self.documents)}) exceeds the maximum "
                f"limit of {_MAX_DOCUMENTS_PER_REQUEST} documents"
            )
        self.documents = [
            (item if isinstance(item, DocumentRecord) else DocumentRecord(item)).to_dict()
            for item in self.documents
        ]


class SearchDocumentsRequest(Struct, kw_only=True, omit_defaults=True):
    """The body of a ``search`` on ``index.documents``.

    Attributes:
        score_by: How to rank the documents, 1 to 100 clauses. Each may be a typed
            :data:`~pinecone.models.documents.score_by.DocumentScoringMethod` variant or a
            plain dict with a ``type`` key. Several ``text`` and ``query_string`` clauses
            can be combined to score on more than one signal at once; a
            ``dense_vector``, ``sparse_vector``, ``embed`` or ``sparse_embed`` clause has
            to stand alone, and combining one with anything else is rejected here.
        top_k: How many documents to return, 1 to 10000.
        include_fields: Which of your fields to return on each match. Omitting it, and
            passing ``[]``, both mean the same thing — only ``_id`` and ``_score`` come
            back. Pass ``["*"]`` for every field, or name the ones you need. Naming a
            field declared with ``store_text: false`` is rejected by the server with a
            ``400``, since its text is not stored; ``["*"]`` leaves such fields out.
        filter: A filter narrowing which documents are searched at all, or ``None`` to
            search the whole namespace. It restricts the candidates; it does not
            contribute to the score. A data field (a vector field, or a string the schema
            declares with ``full_text_search``, ``embed`` or ``sparse_embed``) rejects
            comparison operators such as ``$eq`` with a ``400``. A string whose text is stored
            and searchable takes the text-match operators ``$match_phrase``, ``$match_all`` and
            ``$match_any``: one declared with ``full_text_search``, or with ``embed`` or
            ``sparse_embed`` and the default ``store_text``. A ``store_text=False`` string and a
            vector field do not. A
            metadata field, one the schema does not declare or declares only for filtering,
            takes the comparison, set and logical operators and rejects a text-match
            operator with a ``400``. The two kinds combine under ``$and``.

    Raises:
        PineconeTypeError: If a ``score_by`` item is neither a dict nor a clause type.
        TypeError: If ``top_k`` is not an integer (a bool is not accepted).
        ValueError: If ``score_by`` is empty or holds over 100 clauses, if a vector clause
            is combined with another clause, or if ``top_k`` is outside 1 to 10000.
    """

    score_by: list[DocumentScoringMethod | dict[str, Any]]
    top_k: int
    include_fields: list[str] | None = None
    filter: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.score_by:
            raise ValueError("'score_by' must contain at least one scoring method.")
        if len(self.score_by) > _MAX_SCORING_METHODS:
            raise ValueError(
                f"Number of scoring methods exceeds the maximum limit of "
                f"{_MAX_SCORING_METHODS} methods"
            )
        for position, item in enumerate(self.score_by):
            _check_score_by_item(position, item)
        normalized: list[DocumentScoringMethod | dict[str, Any]] = [
            item
            if not isinstance(item, dict)
            else msgspec.convert(item, type=DocumentScoringMethod)
            for item in self.score_by
        ]
        if len(normalized) > 1 and any(
            isinstance(item, _STANDALONE_CLAUSES) for item in normalized
        ):
            raise ValueError(
                "'score_by' clauses of type 'text' and 'query_string' may be combined, "
                "but 'dense_vector', 'sparse_vector', 'embed', and 'sparse_embed' "
                "clauses must appear alone in 'score_by'"
            )
        self.score_by = normalized
        _require_int("top_k", self.top_k)
        if not 1 <= self.top_k <= _MAX_TOP_K:
            raise ValueError(f"'top_k' must be between 1 and {_MAX_TOP_K} (got {self.top_k}).")


class FetchDocumentsRequest(Struct, kw_only=True, omit_defaults=True):
    """The body of a ``fetch`` on ``index.documents``.

    Select the documents one way or the other: exactly one of ``ids`` and ``filter`` must
    be given. Only the ``filter`` form pages, since only it can match an unbounded number
    of documents.

    Attributes:
        ids: The document IDs to fetch, 1 to 1000. Mutually exclusive with ``filter``.
        filter: A non-empty filter selecting the documents to fetch. Mutually exclusive
            with ``ids``. A data field (a vector field, or a string the schema declares with
            ``full_text_search``, ``embed`` or ``sparse_embed``) rejects comparison operators
            such as ``$eq`` with a ``400``. A string whose text is stored and searchable takes
            the text-match operators ``$match_phrase``, ``$match_all`` and ``$match_any``: one
            declared with ``full_text_search``, or with ``embed`` or ``sparse_embed`` and the
            default ``store_text``. A ``store_text=False`` string and a vector field do not. A
            metadata field, one the schema does not declare or declares only for filtering,
            takes the comparison, set and logical operators and rejects a text-match
            operator with a ``400``. The two kinds combine under ``$and``.
        include_fields: Which of your fields to return. Omitting it, passing ``[]``, and
            passing ``["*"]`` all return every field — the opposite of the default on a
            search, which returns none of them. Every field here means every field the
            index stores: a field declared with ``store_text: false`` is left out, and
            naming one is rejected by the server with a ``400``.
        pagination_token: The token from a previous fetch response, to get the next page.
            Valid only with ``filter``.
        limit: How many documents per page, 1 to 10000, or ``None`` for the server default
            of 100. Only a ``filter`` fetch pages; an ``ids`` fetch ignores it.

    Raises:
        TypeError: If ``limit`` is not an integer (a bool is not accepted).
        ValueError: If both or neither of ``ids`` and ``filter`` are given, if ``filter``
            is an empty object, if ``ids`` holds over 1000 IDs, if ``pagination_token``
            is given without ``filter``, or if ``limit`` is outside 1 to 10000.
    """

    ids: list[str] | None = None
    filter: dict[str, Any] | None = None
    include_fields: list[str] | None = None
    pagination_token: str | None = None
    limit: int | None = None

    def __post_init__(self) -> None:
        if self.pagination_token is not None and self.filter is None:
            raise ValueError(
                "The 'pagination_token' field is only valid together with 'filter' in "
                "document fetch requests. Remove 'pagination_token', or replace 'ids' "
                "with a 'filter' to page through matching documents."
            )
        has_ids = bool(self.ids)
        if has_ids and self.filter is not None:
            raise ValueError(
                "The 'ids' and 'filter' fields are mutually exclusive in document fetch "
                "requests. Provide 'ids' to fetch specific documents, or 'filter' to "
                "fetch every document matching a metadata filter."
            )
        if not has_ids and self.filter is None:
            raise ValueError(
                "No 'ids' or 'filter' provided in the document fetch request. Provide "
                "at least one document ID in 'ids', or a metadata filter in 'filter'."
            )
        _validate_filter_not_empty(self.filter)
        if self.ids is not None:
            _validate_id_count(len(self.ids))
        if self.limit is not None:
            _require_int("limit", self.limit)
        if self.limit is not None and not 1 <= self.limit <= _MAX_FETCH_LIMIT:
            raise ValueError(
                f"'limit' must be between 1 and {_MAX_FETCH_LIMIT} (got {self.limit})."
            )


class DeleteDocumentsRequest(Struct, kw_only=True, omit_defaults=True):
    """The body of a ``delete`` on ``index.documents``.

    Exactly one of the three selectors must be given, so a delete always states its scope
    explicitly and there is no way to write one that means "everything" by omission.

    Attributes:
        ids: The document IDs to delete, 1 to 1000. Mutually exclusive with the others.
        filter: A non-empty metadata filter selecting the documents to delete. A text-match
            operator here is rejected rather than ignored, since evaluated in a filter it
            would widen the delete. Mutually exclusive with the others.
        delete_all: ``True`` deletes every document in the namespace. Mutually exclusive
            with the others.

    Raises:
        ValueError: If more than one selector is given, if none is, if ``filter`` is an
            empty object, or if ``ids`` holds over 1000 IDs.
    """

    ids: list[str] | None = None
    filter: dict[str, Any] | None = None
    delete_all: bool | None = None

    def __post_init__(self) -> None:
        has_ids = bool(self.ids)
        wants_delete_all = bool(self.delete_all)
        if self.filter is not None and (has_ids or wants_delete_all):
            raise ValueError(
                "The 'filter' field is mutually exclusive with 'ids' and 'delete_all' "
                "in document delete requests. Provide exactly one of 'ids', 'filter', "
                "or 'delete_all'."
            )
        if self.filter is None:
            if has_ids and wants_delete_all:
                raise ValueError(
                    "The 'ids' and 'delete_all' fields are mutually exclusive in "
                    "document delete requests. Provide 'ids' to delete specific "
                    "documents, or 'delete_all' to delete every document in the "
                    "namespace."
                )
            if not has_ids and not wants_delete_all:
                raise ValueError(
                    "No 'ids', 'filter', or 'delete_all' provided in the document "
                    "delete request. Provide exactly one of them."
                )
        _validate_filter_not_empty(self.filter)
        if self.ids is not None:
            _validate_id_count(len(self.ids))


class UpdateDocumentsRequest(Struct, kw_only=True, omit_defaults=True):
    """The body of an ``update`` on ``index.documents``, in either of its two forms.

    Patch named documents individually with ``documents``, or patch every document a
    filter matches with ``filter`` plus ``set_fields`` and/or ``remove_fields``. The two
    forms are mutually exclusive. Either way this is a partial update: fields you do not
    name survive.

    Attributes:
        documents: Per-document patches, 1 to 1000. Each may be an
            :class:`~pinecone.models.documents.document.UpdateDocumentRecord` or a plain
            dict; in the dict form ``_id`` and ``_remove_fields`` are reserved keys and
            every other key is a field being set. Mutually exclusive with the by-filter
            fields.
        filter: A non-empty metadata filter selecting the documents to patch. A text-match
            operator here is rejected rather than ignored, since evaluated in a filter it
            would widen the patch. Mutually exclusive with ``documents``.
        set_fields: Fields to set, and their new values, on every document ``filter``
            matches.
        remove_fields: Field names to delete from every document ``filter`` matches.

    Raises:
        ValueError: If ``documents`` is combined with any by-filter field, if neither
            selector is given, if ``set_fields`` or ``remove_fields`` is given without a
            ``filter``, if a ``filter`` is given with nothing to change, if ``filter`` is
            an empty object, or if ``documents`` is empty or holds over 1000 patches.
    """

    documents: list[dict[str, Any] | UpdateDocumentRecord] | None = None
    filter: dict[str, Any] | None = None
    set_fields: dict[str, Any] | None = None
    remove_fields: list[str] | None = None

    def __post_init__(self) -> None:
        has_patch = bool(self.set_fields) or bool(self.remove_fields)
        if self.documents is not None and (self.filter is not None or has_patch):
            raise ValueError(
                "The 'documents' field is mutually exclusive with the by-filter fields "
                "('filter', 'set_fields', 'remove_fields') in document update requests. "
                "Provide per-document patches in 'documents', or a 'filter' together "
                "with 'set_fields' and/or 'remove_fields'."
            )
        if self.documents is None and self.filter is None:
            if has_patch:
                raise ValueError(
                    "The 'set_fields' and 'remove_fields' fields are only valid "
                    "together with 'filter' in document update requests. Add a "
                    "'filter' to select the documents to patch, or move the changes "
                    "into per-document patches in 'documents'."
                )
            raise ValueError(
                "No 'documents' or 'filter' provided in the document update request. "
                "Provide per-document patches in 'documents', or a 'filter' together "
                "with 'set_fields' and/or 'remove_fields'."
            )
        if self.filter is not None:
            _validate_filter_not_empty(self.filter)
            if not has_patch:
                raise ValueError(
                    "A document update by 'filter' must change something. Provide "
                    "'set_fields' and/or 'remove_fields' alongside 'filter'."
                )
        if self.documents is not None:
            if not self.documents:
                raise ValueError("'documents' must contain at least one document update.")
            if len(self.documents) > _MAX_DOCUMENTS_PER_REQUEST:
                raise ValueError(
                    f"Number of documents ({len(self.documents)}) exceeds the maximum "
                    f"limit of {_MAX_DOCUMENTS_PER_REQUEST} documents"
                )
            self.documents = [
                (
                    item if isinstance(item, UpdateDocumentRecord) else UpdateDocumentRecord(item)
                ).to_dict()
                for item in self.documents
            ]


class ListDocumentsRequest(Struct, kw_only=True, omit_defaults=True):
    """The body of a ``list`` on ``index.documents``.

    A list walks IDs, not documents, so none of your fields come back — use it to
    enumerate a namespace, then fetch the IDs you want.

    Attributes:
        prefix: Return only IDs starting with this string, e.g. ``"article-"``. ASCII, at
            most 512 characters. ``None`` lists every ID.
        limit: How many IDs per page, 1 to 100, or ``None`` to let the server choose.
        pagination_token: The token from a previous list response, to get the next page.

    Raises:
        TypeError: If ``limit`` is not an integer (a bool is not accepted).
        ValueError: If ``prefix`` is over 512 characters or contains a non-ASCII character
            or NUL, or if ``limit`` is outside 1 to 100.
    """

    prefix: str | None = None
    limit: int | None = None
    pagination_token: str | None = None

    def __post_init__(self) -> None:
        if self.prefix is not None:
            if len(self.prefix) > _MAX_PREFIX_LENGTH:
                raise ValueError(
                    f"'prefix' exceeds the maximum length of {_MAX_PREFIX_LENGTH} "
                    f"characters (got {len(self.prefix)})."
                )
            if any(not ("\x01" <= char <= "\x7f") for char in self.prefix):
                raise ValueError(
                    "'prefix' must contain only ASCII characters in the range "
                    "\\x01-\\x7F (no NUL byte, no non-ASCII characters)."
                )
        if self.limit is not None:
            _require_int("limit", self.limit)
        if self.limit is not None and not 1 <= self.limit <= _MAX_LIST_LIMIT:
            raise ValueError(f"'limit' must be between 1 and {_MAX_LIST_LIMIT} (got {self.limit}).")
